use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod layout_ops;
mod matrix;
mod stack_parser;

pub use layout_ops::LayoutOps;
pub use matrix::MatrixOps;
pub use stack_parser::RustStackParser;

/// Parse state for the tokenizer state machine
#[derive(Debug, Clone, Copy, PartialEq)]
enum ParseState {
    Main,
    Comment,
    Literal,
    LiteralHex,
    Number,
    Float,
    Keyword,
    String,
    StringEscape,
    WOpen,
    WClose,
    HexString,
}

/// Token type returned to Python
#[derive(Debug, Clone)]
enum Token {
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Literal(Vec<u8>),
    Keyword(Vec<u8>),
}

/// Escape sequences for strings
fn escape_char(c: u8) -> Option<u8> {
    match c {
        b'b' => Some(8),
        b't' => Some(9),
        b'n' => Some(10),
        b'f' => Some(12),
        b'r' => Some(13),
        b'(' => Some(40),
        b')' => Some(41),
        b'\\' => Some(92),
        _ => None,
    }
}

/// Check if character is hex digit
#[inline]
fn is_hex(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

/// Check if character is octal digit
#[inline]
fn is_octal(c: u8) -> bool {
    c >= b'0' && c <= b'7'
}

/// Check if character ends a literal
#[inline]
fn ends_literal(c: u8) -> bool {
    matches!(c, b'#' | b'%' | b'/' | b'[' | b']' | b'(' | b')' | b'<' | b'>' | b'{' | b'}')
        || c.is_ascii_whitespace()
}

/// Check if character ends a keyword
#[inline]
fn ends_keyword(c: u8) -> bool {
    ends_literal(c)
}

/// Check if character ends a hex string (non-hex, non-whitespace)
#[inline]
fn ends_hexstring(c: u8) -> bool {
    !c.is_ascii_whitespace() && !is_hex(c)
}

/// Rust tokenizer for PDF/PostScript
#[pyclass]
pub struct RustTokenizer {
    state: ParseState,
    curtoken: Vec<u8>,
    curtokenpos: usize,
    paren_depth: i32,
    hex_chars: Vec<u8>,
    oct_chars: Vec<u8>,
}

#[pymethods]
impl RustTokenizer {
    #[new]
    pub fn new() -> Self {
        RustTokenizer {
            state: ParseState::Main,
            curtoken: Vec::with_capacity(256),
            curtokenpos: 0,
            paren_depth: 0,
            hex_chars: Vec::with_capacity(2),
            oct_chars: Vec::with_capacity(3),
        }
    }

    /// Reset the tokenizer state
    pub fn reset(&mut self) {
        self.state = ParseState::Main;
        self.curtoken.clear();
        self.curtokenpos = 0;
        self.paren_depth = 0;
        self.hex_chars.clear();
        self.oct_chars.clear();
    }

    /// Get the current token bytes (for Python access when handling incomplete tokens)
    pub fn get_curtoken<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.curtoken)
    }

    /// Check if we're in the middle of parsing a token
    pub fn has_partial_token(&self) -> bool {
        self.state != ParseState::Main || !self.curtoken.is_empty()
    }

    /// Tokenize a buffer starting at position i - returns all tokens
    /// Returns: (new_position, tokens_list)
    /// Each token is (token_pos, token_type, token_data)
    pub fn tokenize<'py>(
        &mut self,
        py: Python<'py>,
        buffer: &[u8],
        start: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Vec<(usize, String, PyObject)>)> {
        let mut i = start;
        let mut tokens: Vec<(usize, String, PyObject)> = Vec::new();

        while i < buffer.len() {
            let (new_i, maybe_token) = self.parse_step(py, buffer, i, bufpos)?;
            i = new_i;

            if let Some((pos, token)) = maybe_token {
                let (ttype, data) = self.token_to_python(py, token)?;
                tokens.push((pos, ttype, data));
            }
        }

        Ok((i, tokens))
    }

    /// Tokenize a buffer starting at position i - returns only one token
    /// Returns: (new_position, optional token)
    /// Token is (token_pos, token_type, token_data)
    pub fn tokenize_one<'py>(
        &mut self,
        py: Python<'py>,
        buffer: &[u8],
        start: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Option<(usize, String, PyObject)>)> {
        let mut i = start;

        while i < buffer.len() {
            let (new_i, maybe_token) = self.parse_step(py, buffer, i, bufpos)?;
            i = new_i;

            if let Some((pos, token)) = maybe_token {
                let (ttype, data) = self.token_to_python(py, token)?;
                return Ok((i, Some((pos, ttype, data))));
            }
        }

        Ok((i, None))
    }

    /// Process a single byte of input when handling EOF or buffer boundary
    pub fn tokenize_byte<'py>(
        &mut self,
        py: Python<'py>,
        c: u8,
        bufpos: usize,
    ) -> PyResult<Vec<(usize, String, PyObject)>> {
        let buffer = [c];
        let (_, tokens) = self.tokenize(py, &buffer, 0, bufpos)?;
        Ok(tokens)
    }
}

impl RustTokenizer {
    /// Convert internal token to Python objects
    fn token_to_python<'py>(
        &self,
        py: Python<'py>,
        token: Token,
    ) -> PyResult<(String, PyObject)> {
        match token {
            Token::Int(v) => Ok(("int".to_string(), v.into_pyobject(py)?.unbind().into_any())),
            Token::Float(v) => Ok(("float".to_string(), v.into_pyobject(py)?.unbind().into_any())),
            Token::Bool(v) => Ok(("bool".to_string(), v.into_pyobject(py)?.to_owned().unbind().into_any())),
            Token::Bytes(v) => Ok((
                "bytes".to_string(),
                PyBytes::new(py, &v).unbind().into_any(),
            )),
            Token::Literal(v) => Ok((
                "literal".to_string(),
                PyBytes::new(py, &v).unbind().into_any(),
            )),
            Token::Keyword(v) => Ok((
                "keyword".to_string(),
                PyBytes::new(py, &v).unbind().into_any(),
            )),
        }
    }

    /// Main parse step - dispatch to appropriate state handler
    fn parse_step(
        &mut self,
        _py: Python<'_>,
        s: &[u8],
        i: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        match self.state {
            ParseState::Main => self.parse_main(s, i, bufpos),
            ParseState::Comment => self.parse_comment(s, i),
            ParseState::Literal => self.parse_literal(s, i),
            ParseState::LiteralHex => self.parse_literal_hex(s, i),
            ParseState::Number => self.parse_number(s, i),
            ParseState::Float => self.parse_float(s, i),
            ParseState::Keyword => self.parse_keyword(s, i),
            ParseState::String => self.parse_string(s, i),
            ParseState::StringEscape => self.parse_string_escape(s, i),
            ParseState::WOpen => self.parse_wopen(s, i),
            ParseState::WClose => self.parse_wclose(s, i),
            ParseState::HexString => self.parse_hexstring(s, i),
        }
    }

    /// Main state - looking for start of next token
    fn parse_main(
        &mut self,
        s: &[u8],
        i: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        // Skip whitespace
        let mut j = i;
        while j < s.len() && s[j].is_ascii_whitespace() {
            j += 1;
        }

        if j >= s.len() {
            return Ok((j, None));
        }

        let c = s[j];
        self.curtokenpos = bufpos + j;

        match c {
            b'%' => {
                self.curtoken.clear();
                self.curtoken.push(b'%');
                self.state = ParseState::Comment;
                Ok((j + 1, None))
            }
            b'/' => {
                self.curtoken.clear();
                self.state = ParseState::Literal;
                Ok((j + 1, None))
            }
            b'-' | b'+' | b'0'..=b'9' => {
                self.curtoken.clear();
                self.curtoken.push(c);
                self.state = ParseState::Number;
                Ok((j + 1, None))
            }
            b'.' => {
                self.curtoken.clear();
                self.curtoken.push(c);
                self.state = ParseState::Float;
                Ok((j + 1, None))
            }
            b'a'..=b'z' | b'A'..=b'Z' => {
                self.curtoken.clear();
                self.curtoken.push(c);
                self.state = ParseState::Keyword;
                Ok((j + 1, None))
            }
            b'(' => {
                self.curtoken.clear();
                self.paren_depth = 1;
                self.state = ParseState::String;
                Ok((j + 1, None))
            }
            b'<' => {
                self.curtoken.clear();
                self.state = ParseState::WOpen;
                Ok((j + 1, None))
            }
            b'>' => {
                self.curtoken.clear();
                self.state = ParseState::WClose;
                Ok((j + 1, None))
            }
            b'\x00' => {
                // Null byte - skip it
                Ok((j + 1, None))
            }
            _ => {
                // Single-character keyword (e.g., '[', ']', '{', '}')
                let token = Token::Keyword(vec![c]);
                Ok((j + 1, Some((self.curtokenpos, token))))
            }
        }
    }

    /// Parse comment - consume until end of line
    fn parse_comment(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        // Find end of line
        let mut j = i;
        while j < s.len() && s[j] != b'\r' && s[j] != b'\n' {
            j += 1;
        }

        if j >= s.len() {
            // Haven't found end of line yet
            self.curtoken.extend_from_slice(&s[i..]);
            return Ok((j, None));
        }

        // Found end of line - comment is done
        self.curtoken.extend_from_slice(&s[i..j]);
        self.state = ParseState::Main;
        // Comments are ignored - don't emit a token
        Ok((j, None))
    }

    /// Parse literal name (e.g., /Name)
    fn parse_literal(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() && !ends_literal(s[j]) {
            j += 1;
        }

        if j >= s.len() {
            // Need more data
            self.curtoken.extend_from_slice(&s[i..]);
            return Ok((j, None));
        }

        self.curtoken.extend_from_slice(&s[i..j]);

        let c = s[j];
        if c == b'#' {
            // Hex escape in literal
            self.hex_chars.clear();
            self.state = ParseState::LiteralHex;
            return Ok((j + 1, None));
        }

        // Literal is complete
        let token = Token::Literal(std::mem::take(&mut self.curtoken));
        self.state = ParseState::Main;
        Ok((j, Some((self.curtokenpos, token))))
    }

    /// Parse hex escape in literal (e.g., #20 for space)
    fn parse_literal_hex(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        if i >= s.len() {
            return Ok((i, None));
        }

        let c = s[i];
        if is_hex(c) && self.hex_chars.len() < 2 {
            self.hex_chars.push(c);
            return Ok((i + 1, None));
        }

        // Convert hex chars to byte
        if !self.hex_chars.is_empty() {
            let hex_str = std::str::from_utf8(&self.hex_chars).unwrap_or("00");
            if let Ok(byte) = u8::from_str_radix(hex_str, 16) {
                self.curtoken.push(byte);
            }
            self.hex_chars.clear();
        }

        self.state = ParseState::Literal;
        Ok((i, None))
    }

    /// Parse integer number
    fn parse_number(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }

        if j >= s.len() {
            self.curtoken.extend_from_slice(&s[i..]);
            return Ok((j, None));
        }

        self.curtoken.extend_from_slice(&s[i..j]);

        let c = s[j];
        if c == b'.' {
            // It's a float
            self.curtoken.push(c);
            self.state = ParseState::Float;
            return Ok((j + 1, None));
        }

        // Parse as integer
        let token_str = std::str::from_utf8(&self.curtoken).unwrap_or("0");
        if let Ok(v) = token_str.parse::<i64>() {
            let token = Token::Int(v);
            self.curtoken.clear();
            self.state = ParseState::Main;
            return Ok((j, Some((self.curtokenpos, token))));
        }

        // Failed to parse - discard and continue
        self.curtoken.clear();
        self.state = ParseState::Main;
        Ok((j, None))
    }

    /// Parse float number
    fn parse_float(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }

        if j >= s.len() {
            self.curtoken.extend_from_slice(&s[i..]);
            return Ok((j, None));
        }

        self.curtoken.extend_from_slice(&s[i..j]);

        // Parse as float
        let token_str = std::str::from_utf8(&self.curtoken).unwrap_or("0");
        if let Ok(v) = token_str.parse::<f64>() {
            let token = Token::Float(v);
            self.curtoken.clear();
            self.state = ParseState::Main;
            return Ok((j, Some((self.curtokenpos, token))));
        }

        // Failed to parse - discard and continue
        self.curtoken.clear();
        self.state = ParseState::Main;
        Ok((j, None))
    }

    /// Parse keyword (e.g., true, false, stream, etc.)
    fn parse_keyword(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() && !ends_keyword(s[j]) {
            j += 1;
        }

        if j >= s.len() {
            self.curtoken.extend_from_slice(&s[i..]);
            return Ok((j, None));
        }

        self.curtoken.extend_from_slice(&s[i..j]);

        // Check for special keywords
        let token = if self.curtoken == b"true" {
            Token::Bool(true)
        } else if self.curtoken == b"false" {
            Token::Bool(false)
        } else {
            Token::Keyword(std::mem::take(&mut self.curtoken))
        };

        self.curtoken.clear();
        self.state = ParseState::Main;
        Ok((j, Some((self.curtokenpos, token))))
    }

    /// Parse string (parenthesized)
    fn parse_string(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() {
            let c = s[j];
            match c {
                b'\\' => {
                    self.curtoken.extend_from_slice(&s[i..j]);
                    self.oct_chars.clear();
                    self.state = ParseState::StringEscape;
                    return Ok((j + 1, None));
                }
                b'(' => {
                    self.paren_depth += 1;
                    j += 1;
                }
                b')' => {
                    self.paren_depth -= 1;
                    if self.paren_depth == 0 {
                        self.curtoken.extend_from_slice(&s[i..j]);
                        let token = Token::Bytes(std::mem::take(&mut self.curtoken));
                        self.state = ParseState::Main;
                        return Ok((j + 1, Some((self.curtokenpos, token))));
                    }
                    j += 1;
                }
                _ => {
                    j += 1;
                }
            }
        }

        // Need more data
        self.curtoken.extend_from_slice(&s[i..]);
        Ok((j, None))
    }

    /// Parse string escape sequence
    fn parse_string_escape(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        if i >= s.len() {
            return Ok((i, None));
        }

        let c = s[i];

        // Check for octal escape
        if is_octal(c) && self.oct_chars.len() < 3 {
            self.oct_chars.push(c);
            return Ok((i + 1, None));
        }

        // If we have accumulated octal chars, convert them
        if !self.oct_chars.is_empty() {
            let oct_str = std::str::from_utf8(&self.oct_chars).unwrap_or("0");
            if let Ok(byte) = u8::from_str_radix(oct_str, 8) {
                self.curtoken.push(byte);
            }
            self.oct_chars.clear();
            self.state = ParseState::String;
            return Ok((i, None));
        }

        // Check for escape sequence
        if let Some(escaped) = escape_char(c) {
            self.curtoken.push(escaped);
            self.state = ParseState::String;
            return Ok((i + 1, None));
        }

        // Handle \r\n -> ignore both
        if c == b'\r' && i + 1 < s.len() && s[i + 1] == b'\n' {
            self.state = ParseState::String;
            return Ok((i + 2, None));
        }

        // Handle \r or \n alone -> ignore
        if c == b'\r' || c == b'\n' {
            self.state = ParseState::String;
            return Ok((i + 1, None));
        }

        // Unknown escape - just include the character
        self.state = ParseState::String;
        Ok((i + 1, None))
    }

    /// Parse < - could be dict begin (<<) or hex string
    fn parse_wopen(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        if i >= s.len() {
            return Ok((i, None));
        }

        let c = s[i];
        if c == b'<' {
            // It's <<
            let token = Token::Keyword(b"<<".to_vec());
            self.state = ParseState::Main;
            return Ok((i + 1, Some((self.curtokenpos, token))));
        }

        // It's a hex string
        self.state = ParseState::HexString;
        Ok((i, None))
    }

    /// Parse > - could be dict end (>>) or error
    fn parse_wclose(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        if i >= s.len() {
            return Ok((i, None));
        }

        let c = s[i];
        if c == b'>' {
            // It's >>
            let token = Token::Keyword(b">>".to_vec());
            self.state = ParseState::Main;
            return Ok((i + 1, Some((self.curtokenpos, token))));
        }

        // Single > - shouldn't happen in valid PDF, just ignore
        self.state = ParseState::Main;
        Ok((i, None))
    }

    /// Parse hex string (e.g., <48454C4C4F>)
    fn parse_hexstring(
        &mut self,
        s: &[u8],
        i: usize,
    ) -> PyResult<(usize, Option<(usize, Token)>)> {
        let mut j = i;
        while j < s.len() && !ends_hexstring(s[j]) {
            // Skip whitespace, accumulate hex chars
            if !s[j].is_ascii_whitespace() {
                self.curtoken.push(s[j]);
            }
            j += 1;
        }

        if j >= s.len() {
            return Ok((j, None));
        }

        // Found end of hex string (should be '>')
        // Convert hex to bytes
        let hex_bytes = self.decode_hex(&self.curtoken);
        let token = Token::Bytes(hex_bytes);
        self.curtoken.clear();
        self.state = ParseState::Main;
        Ok((j, Some((self.curtokenpos, token))))
    }

    /// Decode hex string to bytes
    /// Matches Python behavior: pairs of hex digits, trailing single digit
    /// is treated as its value (not shifted to high nibble)
    fn decode_hex(&self, hex: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(hex.len() / 2 + 1);
        let mut i = 0;

        while i < hex.len() {
            if i + 1 < hex.len() {
                // Two hex digits - combine into one byte
                let high_val = Self::hex_to_nibble(hex[i]);
                let low_val = Self::hex_to_nibble(hex[i + 1]);
                result.push((high_val << 4) | low_val);
                i += 2;
            } else {
                // Single trailing hex digit - use as-is (Python compat)
                let val = Self::hex_to_nibble(hex[i]);
                result.push(val);
                i += 1;
            }
        }

        result
    }

    /// Convert hex char to nibble value
    #[inline]
    fn hex_to_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => 0,
        }
    }
}

/// Python module definition
#[pymodule]
fn pdfminer_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustTokenizer>()?;
    m.add_class::<RustStackParser>()?;
    m.add_class::<MatrixOps>()?;
    m.add_class::<LayoutOps>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hex() {
        assert!(is_hex(b'0'));
        assert!(is_hex(b'9'));
        assert!(is_hex(b'a'));
        assert!(is_hex(b'f'));
        assert!(is_hex(b'A'));
        assert!(is_hex(b'F'));
        assert!(!is_hex(b'g'));
        assert!(!is_hex(b'z'));
    }

    #[test]
    fn test_hex_to_nibble() {
        assert_eq!(RustTokenizer::hex_to_nibble(b'0'), 0);
        assert_eq!(RustTokenizer::hex_to_nibble(b'9'), 9);
        assert_eq!(RustTokenizer::hex_to_nibble(b'a'), 10);
        assert_eq!(RustTokenizer::hex_to_nibble(b'f'), 15);
        assert_eq!(RustTokenizer::hex_to_nibble(b'A'), 10);
        assert_eq!(RustTokenizer::hex_to_nibble(b'F'), 15);
    }
}
