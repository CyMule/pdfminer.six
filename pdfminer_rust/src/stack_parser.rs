//! Stack-based parser for PDF/PostScript objects.
//!
//! This module implements a stack parser that builds on top of the tokenizer
//! to construct complete PDF objects (arrays, dictionaries, procedures).
//!
//! The parser follows the PostScript/PDF object construction semantics:
//! - `[` begins an array, `]` ends it and collects items into a list
//! - `<<` begins a dictionary, `>>` ends it and collects key/value pairs
//! - `{` begins a procedure, `}` ends it and collects into a tuple
//! - Primitives (int, float, bool, bytes, literal, keyword) are pushed to stack

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::RustTokenizer;

/// Context type for nested object construction
#[derive(Debug, Clone, Copy, PartialEq)]
enum ContextType {
    Array,      // Started by `[`
    Dict,       // Started by `<<`
    Procedure,  // Started by `{`
}

/// Internal representation of parsed values before Python conversion
#[derive(Debug, Clone)]
enum ParsedValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Literal(Vec<u8>),
    Keyword(Vec<u8>),
    Array(Vec<(usize, ParsedValue)>),
    Dict(Vec<(usize, ParsedValue)>),
    Procedure(Vec<(usize, ParsedValue)>),
    /// PDF object reference - (objid, genno)
    ObjRef(i64, i64),
    /// PDF null value
    Null,
}

/// Context frame for nested object construction
#[derive(Debug)]
struct ContextFrame {
    pos: usize,
    context_type: ContextType,
    stack: Vec<(usize, ParsedValue)>,
}

/// Rust stack parser for PDF/PostScript objects.
///
/// Wraps a RustTokenizer and provides higher-level object parsing,
/// constructing arrays, dictionaries, and procedures from token streams.
#[pyclass]
pub struct RustStackParser {
    tokenizer: RustTokenizer,
    /// Current stack of parsed values (pos, value)
    curstack: Vec<(usize, ParsedValue)>,
    /// Context stack for nested constructions
    context: Vec<ContextFrame>,
    /// Results queue - completed top-level objects
    results: Vec<(usize, ParsedValue)>,
}

#[pymethods]
impl RustStackParser {
    /// Create a new stack parser
    #[new]
    pub fn new() -> Self {
        RustStackParser {
            tokenizer: RustTokenizer::new(),
            curstack: Vec::new(),
            context: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Reset the parser state
    pub fn reset(&mut self) {
        self.tokenizer.reset();
        self.curstack.clear();
        self.context.clear();
        self.results.clear();
    }

    /// Parse the next complete object from the buffer.
    ///
    /// Returns: (new_charpos, Option<(pos, object)>)
    ///
    /// The returned object can be:
    /// - int, float, bool, bytes for primitives
    /// - ("literal", bytes) tuple for literals (Python wraps with LIT())
    /// - ("keyword", bytes) tuple for keywords (Python wraps with KWD())
    /// - list for arrays
    /// - dict for dictionaries
    /// - tuple for procedures
    pub fn nextobject<'py>(
        &mut self,
        py: Python<'py>,
        buffer: &[u8],
        charpos: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Option<(usize, PyObject)>)> {
        let mut pos = charpos;

        // If we have results already, return one
        if let Some((obj_pos, value)) = self.results.pop() {
            let py_obj = self.value_to_python(py, value)?;
            return Ok((pos, Some((obj_pos, py_obj))));
        }

        // Process tokens until we have a result
        loop {
            // Get next token
            let (new_pos, maybe_token) =
                self.tokenizer.tokenize_one(py, buffer, pos, bufpos)?;
            pos = new_pos;

            let (token_pos, token_type, token_data) = match maybe_token {
                Some(t) => t,
                None => {
                    // No more tokens in buffer
                    return Ok((pos, None));
                }
            };

            // Convert token to internal value and process
            self.process_token(py, token_pos, &token_type, token_data)?;

            // If we're not in a context (not building array/dict/proc)
            // and we have items on the stack, flush to results
            if self.context.is_empty() {
                self.flush();
            }

            // If we have results, return one
            if let Some((obj_pos, value)) = self.results.pop() {
                let py_obj = self.value_to_python(py, value)?;
                return Ok((pos, Some((obj_pos, py_obj))));
            }
        }
    }

    /// Check if there's a partial token or object in progress
    pub fn has_partial(&self) -> bool {
        self.tokenizer.has_partial_token()
            || !self.curstack.is_empty()
            || !self.context.is_empty()
    }

    /// Parse ALL complete objects from the buffer in one call.
    ///
    /// This is more efficient than calling nextobject() repeatedly because
    /// it reduces FFI overhead by returning all objects at once.
    ///
    /// Returns: (new_charpos, list of (pos, object) tuples)
    pub fn parse_buffer<'py>(
        &mut self,
        py: Python<'py>,
        buffer: &[u8],
        charpos: usize,
        bufpos: usize,
    ) -> PyResult<(usize, Vec<(usize, PyObject)>)> {
        let mut pos = charpos;
        let mut all_results: Vec<(usize, PyObject)> = Vec::new();

        // First, drain any existing results (collect to avoid borrow issues)
        let existing: Vec<_> = self.results.drain(..).collect();
        for (obj_pos, value) in existing {
            let py_obj = self.value_to_python(py, value)?;
            all_results.push((obj_pos, py_obj));
        }

        // Process all tokens in the buffer
        loop {
            let (new_pos, maybe_token) =
                self.tokenizer.tokenize_one(py, buffer, pos, bufpos)?;
            pos = new_pos;

            let (token_pos, token_type, token_data) = match maybe_token {
                Some(t) => t,
                None => {
                    // No more tokens in buffer - flush and return all results
                    if self.context.is_empty() {
                        self.flush();
                    }
                    let final_results: Vec<_> = self.results.drain(..).collect();
                    for (obj_pos, value) in final_results {
                        let py_obj = self.value_to_python(py, value)?;
                        all_results.push((obj_pos, py_obj));
                    }
                    return Ok((pos, all_results));
                }
            };

            // Convert token to internal value and process
            self.process_token(py, token_pos, &token_type, token_data)?;

            // If we're not in a context, flush to results
            if self.context.is_empty() {
                self.flush();
            }
        }
    }
}

impl RustStackParser {
    /// Process a single token
    fn process_token(
        &mut self,
        py: Python<'_>,
        pos: usize,
        token_type: &str,
        token_data: PyObject,
    ) -> PyResult<()> {
        match token_type {
            "int" => {
                let value: i64 = token_data.extract(py)?;
                self.push(pos, ParsedValue::Int(value));
            }
            "float" => {
                let value: f64 = token_data.extract(py)?;
                self.push(pos, ParsedValue::Float(value));
            }
            "bool" => {
                let value: bool = token_data.extract(py)?;
                self.push(pos, ParsedValue::Bool(value));
            }
            "bytes" => {
                let bytes = token_data.downcast_bound::<PyBytes>(py)?;
                self.push(pos, ParsedValue::Bytes(bytes.as_bytes().to_vec()));
            }
            "literal" => {
                let bytes = token_data.downcast_bound::<PyBytes>(py)?;
                self.push(pos, ParsedValue::Literal(bytes.as_bytes().to_vec()));
            }
            "keyword" => {
                let bytes = token_data.downcast_bound::<PyBytes>(py)?;
                let kwd = bytes.as_bytes();
                self.handle_keyword(pos, kwd);
            }
            _ => {
                // Unknown token type - push as keyword
                if let Ok(bytes) = token_data.downcast_bound::<PyBytes>(py) {
                    self.push(pos, ParsedValue::Keyword(bytes.as_bytes().to_vec()));
                }
            }
        }
        Ok(())
    }

    /// Handle a keyword token
    fn handle_keyword(&mut self, pos: usize, kwd: &[u8]) {
        match kwd {
            b"[" | b"]" => {
                // Don't build arrays in Rust - let Python handle them
                // This allows Python's do_keyword() to process items inside arrays
                self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
            }
            b"<<" | b">>" => {
                // Don't build dicts in Rust - let Python handle them
                // This is necessary because arrays inside dicts would break
                // the key-value pairing if arrays aren't built in Rust
                self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
            }
            b"{" | b"}" => {
                // Don't build procedures in Rust - let Python handle them
                // This allows Python's do_keyword() to process items inside procedures
                self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
            }
            b"R" => {
                // PDF object reference: objid genno R
                // Pop the last 2 items (genno, objid) and push an ObjRef
                if self.curstack.len() >= 2 {
                    let (_, genno_val) = self.curstack.pop().unwrap();
                    let (objid_pos, objid_val) = self.curstack.pop().unwrap();

                    // Extract integers from the values
                    let objid = match objid_val {
                        ParsedValue::Int(v) => v,
                        _ => {
                            // Not an int, push everything back and treat R as keyword
                            self.curstack.push((objid_pos, objid_val));
                            self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
                            return;
                        }
                    };
                    let genno = match genno_val {
                        ParsedValue::Int(v) => v,
                        _ => {
                            // Not an int, push everything back and treat R as keyword
                            self.curstack.push((objid_pos, ParsedValue::Int(objid)));
                            self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
                            return;
                        }
                    };

                    self.push(objid_pos, ParsedValue::ObjRef(objid, genno));
                } else {
                    // Not enough items on stack, push as keyword
                    self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
                }
            }
            b"null" => {
                // PDF null value
                self.push(pos, ParsedValue::Null);
            }
            _ => {
                // Other keywords - push for do_keyword handling
                self.push(pos, ParsedValue::Keyword(kwd.to_vec()));
            }
        }
    }

    /// Push a value onto the current stack
    fn push(&mut self, pos: usize, value: ParsedValue) {
        self.curstack.push((pos, value));
    }

    /// Start a new context for nested object construction
    fn start_context(&mut self, pos: usize, context_type: ContextType) {
        let old_stack = std::mem::take(&mut self.curstack);
        self.context.push(ContextFrame {
            pos,
            context_type,
            stack: old_stack,
        });
    }

    /// End the current context and return collected items
    fn end_context(&mut self, expected_type: ContextType) -> Option<(usize, Vec<(usize, ParsedValue)>)> {
        if let Some(frame) = self.context.pop() {
            if frame.context_type != expected_type {
                // Type mismatch - restore context and return None
                self.context.push(frame);
                return None;
            }

            let items = std::mem::take(&mut self.curstack);
            self.curstack = frame.stack;
            Some((frame.pos, items))
        } else {
            None
        }
    }

    /// Flush current stack to results (when not in a context)
    fn flush(&mut self) {
        self.results.append(&mut self.curstack);
    }

    /// Convert internal ParsedValue to Python object
    fn value_to_python(&self, py: Python<'_>, value: ParsedValue) -> PyResult<PyObject> {
        match value {
            ParsedValue::Int(v) => Ok(v.into_pyobject(py)?.unbind().into_any()),
            ParsedValue::Float(v) => Ok(v.into_pyobject(py)?.unbind().into_any()),
            ParsedValue::Bool(v) => Ok(v.into_pyobject(py)?.to_owned().unbind().into_any()),
            ParsedValue::Bytes(v) => Ok(PyBytes::new(py, &v).unbind().into_any()),
            ParsedValue::Literal(v) => {
                // Return tuple ("literal", bytes) - Python wraps with LIT()
                let tuple = PyTuple::new(
                    py,
                    &[
                        "literal".into_pyobject(py)?.unbind().into_any(),
                        PyBytes::new(py, &v).unbind().into_any(),
                    ],
                )?;
                Ok(tuple.unbind().into_any())
            }
            ParsedValue::Keyword(v) => {
                // Return tuple ("keyword", bytes) - Python wraps with KWD()
                let tuple = PyTuple::new(
                    py,
                    &[
                        "keyword".into_pyobject(py)?.unbind().into_any(),
                        PyBytes::new(py, &v).unbind().into_any(),
                    ],
                )?;
                Ok(tuple.unbind().into_any())
            }
            ParsedValue::Array(items) => {
                // Convert to Python list (just the values, not positions)
                let list = PyList::empty(py);
                for (_, item) in items {
                    let py_item = self.value_to_python(py, item)?;
                    list.append(py_item)?;
                }
                Ok(list.unbind().into_any())
            }
            ParsedValue::Dict(items) => {
                // Convert to Python dict (key/value pairs, just values)
                // In PDF, dict keys are literals and values follow
                let dict = PyDict::new(py);
                let values: Vec<_> = items.into_iter().map(|(_, v)| v).collect();

                if values.len() % 2 != 0 {
                    // Invalid dict - odd number of items
                    // Return error marker for Python to raise PSSyntaxError
                    let list = PyList::empty(py);
                    for item in values {
                        let py_item = self.value_to_python(py, item)?;
                        list.append(py_item)?;
                    }
                    let tuple = PyTuple::new(
                        py,
                        &[
                            "dict_error".into_pyobject(py)?.unbind().into_any(),
                            list.unbind().into_any(),
                        ],
                    )?;
                    return Ok(tuple.unbind().into_any());
                }

                for chunk in values.chunks(2) {
                    if let [key, value] = chunk {
                        // Get the key name from literal
                        let key_name = match key {
                            ParsedValue::Literal(bytes) => {
                                // Try to decode as UTF-8, fall back to bytes
                                match std::str::from_utf8(bytes) {
                                    Ok(s) => s.into_pyobject(py)?.unbind().into_any(),
                                    Err(_) => PyBytes::new(py, bytes).unbind().into_any(),
                                }
                            }
                            _ => {
                                // Non-literal key - convert to Python object
                                self.value_to_python(py, key.clone())?
                            }
                        };

                        let py_value = self.value_to_python(py, value.clone())?;
                        dict.set_item(key_name, py_value)?;
                    }
                }
                Ok(dict.unbind().into_any())
            }
            ParsedValue::Procedure(items) => {
                // Convert to Python list (just the values, same as arrays)
                // In Python, procedures are also represented as lists
                let list = PyList::empty(py);
                for (_, item) in items {
                    let py_item = self.value_to_python(py, item)?;
                    list.append(py_item)?;
                }
                Ok(list.unbind().into_any())
            }
            ParsedValue::ObjRef(objid, genno) => {
                // Return tuple ("objref", objid, genno) - Python converts to PDFObjRef
                let tuple = PyTuple::new(
                    py,
                    &[
                        "objref".into_pyobject(py)?.unbind().into_any(),
                        objid.into_pyobject(py)?.unbind().into_any(),
                        genno.into_pyobject(py)?.unbind().into_any(),
                    ],
                )?;
                Ok(tuple.unbind().into_any())
            }
            ParsedValue::Null => {
                // Return Python None
                Ok(py.None().into_pyobject(py)?.unbind().into_any())
            }
        }
    }
}

impl Default for RustStackParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_type_equality() {
        assert_eq!(ContextType::Array, ContextType::Array);
        assert_ne!(ContextType::Array, ContextType::Dict);
        assert_ne!(ContextType::Dict, ContextType::Procedure);
    }

    #[test]
    fn test_parsed_value_debug() {
        let v = ParsedValue::Int(42);
        assert!(format!("{:?}", v).contains("42"));

        let v = ParsedValue::Float(3.14);
        assert!(format!("{:?}", v).contains("3.14"));

        let v = ParsedValue::Bool(true);
        assert!(format!("{:?}", v).contains("true"));

        let v = ParsedValue::Bytes(b"hello".to_vec());
        let debug_str = format!("{:?}", v);
        assert!(debug_str.contains("Bytes")); // Vec<u8> shows as bytes, not string

        let v = ParsedValue::Literal(b"Name".to_vec());
        assert!(format!("{:?}", v).contains("Literal"));

        let v = ParsedValue::Keyword(b"stream".to_vec());
        assert!(format!("{:?}", v).contains("Keyword"));
    }

    #[test]
    fn test_stack_parser_new() {
        let parser = RustStackParser::new();
        assert!(!parser.has_partial());
        assert!(parser.curstack.is_empty());
        assert!(parser.context.is_empty());
        assert!(parser.results.is_empty());
    }

    #[test]
    fn test_stack_parser_reset() {
        let mut parser = RustStackParser::new();
        parser.curstack.push((0, ParsedValue::Int(1)));
        parser.results.push((0, ParsedValue::Int(2)));

        parser.reset();

        assert!(parser.curstack.is_empty());
        assert!(parser.context.is_empty());
        assert!(parser.results.is_empty());
    }

    #[test]
    fn test_push_and_flush() {
        let mut parser = RustStackParser::new();

        parser.push(0, ParsedValue::Int(42));
        parser.push(5, ParsedValue::Float(3.14));
        parser.push(10, ParsedValue::Bool(true));

        assert_eq!(parser.curstack.len(), 3);
        assert!(parser.results.is_empty());

        parser.flush();

        assert!(parser.curstack.is_empty());
        assert_eq!(parser.results.len(), 3);
    }

    #[test]
    fn test_start_and_end_context_array() {
        let mut parser = RustStackParser::new();

        // Push some items first
        parser.push(0, ParsedValue::Int(1));

        // Start array context
        parser.start_context(5, ContextType::Array);

        // The old stack should be saved, curstack should be empty
        assert!(parser.curstack.is_empty());
        assert_eq!(parser.context.len(), 1);

        // Push items into array
        parser.push(6, ParsedValue::Int(2));
        parser.push(8, ParsedValue::Int(3));

        // End array context
        let result = parser.end_context(ContextType::Array);
        assert!(result.is_some());

        let (pos, items) = result.unwrap();
        assert_eq!(pos, 5);
        assert_eq!(items.len(), 2);

        // Original stack should be restored
        assert_eq!(parser.curstack.len(), 1);
        assert!(parser.context.is_empty());
    }

    #[test]
    fn test_nested_contexts() {
        let mut parser = RustStackParser::new();

        // Start array
        parser.start_context(0, ContextType::Array);
        parser.push(1, ParsedValue::Int(1));

        // Start nested dict
        parser.start_context(5, ContextType::Dict);
        parser.push(6, ParsedValue::Literal(b"key".to_vec()));
        parser.push(10, ParsedValue::Int(42));

        // End dict
        let dict_result = parser.end_context(ContextType::Dict);
        assert!(dict_result.is_some());
        let (dict_pos, dict_items) = dict_result.unwrap();
        assert_eq!(dict_pos, 5);
        assert_eq!(dict_items.len(), 2);

        // Push dict as array element
        parser.push(dict_pos, ParsedValue::Dict(dict_items));

        // End array
        let array_result = parser.end_context(ContextType::Array);
        assert!(array_result.is_some());
        let (array_pos, array_items) = array_result.unwrap();
        assert_eq!(array_pos, 0);
        assert_eq!(array_items.len(), 2); // Int(1) and Dict
    }

    #[test]
    fn test_end_context_type_mismatch() {
        let mut parser = RustStackParser::new();

        parser.start_context(0, ContextType::Array);

        // Try to end with wrong type
        let result = parser.end_context(ContextType::Dict);
        assert!(result.is_none());

        // Context should still be there
        assert_eq!(parser.context.len(), 1);
    }

    #[test]
    fn test_end_context_no_context() {
        let mut parser = RustStackParser::new();

        let result = parser.end_context(ContextType::Array);
        assert!(result.is_none());
    }

    #[test]
    fn test_handle_keyword_array() {
        let mut parser = RustStackParser::new();

        // [
        parser.handle_keyword(0, b"[");
        assert_eq!(parser.context.len(), 1);

        // Push some items
        parser.push(1, ParsedValue::Int(1));
        parser.push(3, ParsedValue::Int(2));

        // ]
        parser.handle_keyword(5, b"]");
        assert!(parser.context.is_empty());
        assert_eq!(parser.curstack.len(), 1);

        // Should have an array on the stack
        if let (_, ParsedValue::Array(items)) = &parser.curstack[0] {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected array on stack");
        }
    }

    #[test]
    fn test_handle_keyword_dict() {
        let mut parser = RustStackParser::new();

        // <<
        parser.handle_keyword(0, b"<<");
        assert_eq!(parser.context.len(), 1);

        // /Key 42
        parser.push(2, ParsedValue::Literal(b"Key".to_vec()));
        parser.push(6, ParsedValue::Int(42));

        // >>
        parser.handle_keyword(8, b">>");
        assert!(parser.context.is_empty());
        assert_eq!(parser.curstack.len(), 1);

        // Should have a dict on the stack
        if let (_, ParsedValue::Dict(items)) = &parser.curstack[0] {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected dict on stack");
        }
    }

    #[test]
    fn test_handle_keyword_procedure() {
        let mut parser = RustStackParser::new();

        // {
        parser.handle_keyword(0, b"{");
        assert_eq!(parser.context.len(), 1);

        // Push some items
        parser.push(1, ParsedValue::Keyword(b"add".to_vec()));
        parser.push(5, ParsedValue::Int(1));

        // }
        parser.handle_keyword(7, b"}");
        assert!(parser.context.is_empty());
        assert_eq!(parser.curstack.len(), 1);

        // Should have a procedure on the stack
        if let (_, ParsedValue::Procedure(items)) = &parser.curstack[0] {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected procedure on stack");
        }
    }

    #[test]
    fn test_handle_keyword_other() {
        let mut parser = RustStackParser::new();

        parser.handle_keyword(0, b"stream");
        assert_eq!(parser.curstack.len(), 1);

        if let (_, ParsedValue::Keyword(kwd)) = &parser.curstack[0] {
            assert_eq!(kwd, b"stream");
        } else {
            panic!("Expected keyword on stack");
        }
    }

    #[test]
    fn test_has_partial_empty() {
        let parser = RustStackParser::new();
        assert!(!parser.has_partial());
    }

    #[test]
    fn test_has_partial_with_stack() {
        let mut parser = RustStackParser::new();
        parser.push(0, ParsedValue::Int(1));
        assert!(parser.has_partial());
    }

    #[test]
    fn test_has_partial_with_context() {
        let mut parser = RustStackParser::new();
        parser.start_context(0, ContextType::Array);
        assert!(parser.has_partial());
    }

    #[test]
    fn test_default_trait() {
        let parser = RustStackParser::default();
        assert!(!parser.has_partial());
    }
}
