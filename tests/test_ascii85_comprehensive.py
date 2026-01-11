"""Comprehensive tests for pdfminer/ascii85.py

This module tests the ASCII85 and ASCIIHex decoders used for PDF stream decoding.
"""

import pytest

from pdfminer.ascii85 import ascii85decode, asciihexdecode


class TestAscii85DecodeBasic:
    """Basic tests for ascii85decode function"""

    def test_empty_input(self):
        result = ascii85decode(b"")
        assert result == b""

    def test_simple_decode(self):
        result = ascii85decode(b"87cUR")
        assert result == b"Hell"

    def test_decode_hello_world(self):
        result = ascii85decode(b'87cURD]i,"Ebo7')
        assert result == b"Hello World"

    def test_decode_with_trailing_tilde_greater(self):
        result = ascii85decode(b"87cUR~>")
        assert result == b"Hell"

    def test_decode_with_leading_less_tilde(self):
        result = ascii85decode(b"<~87cUR")
        assert result == b"Hell"

    def test_decode_with_full_markers(self):
        result = ascii85decode(b"<~87cUR~>")
        assert result == b"Hell"


class TestAscii85DecodeWhitespace:
    """Tests for whitespace handling in ascii85decode"""

    def test_decode_with_leading_whitespace(self):
        result = ascii85decode(b"   87cUR")
        assert result == b"Hell"

    def test_decode_with_trailing_whitespace(self):
        result = ascii85decode(b"87cUR   ")
        assert result == b"Hell"

    def test_decode_with_newlines(self):
        result = ascii85decode(b"87cUR\n")
        assert result == b"Hell"

    def test_decode_with_tabs(self):
        result = ascii85decode(b"\t87cUR")
        assert result == b"Hell"

    def test_decode_with_mixed_leading_trailing_whitespace(self):
        result = ascii85decode(b" \t\n 87cUR \t\n ")
        assert result == b"Hell"

    def test_whitespace_before_markers(self):
        result = ascii85decode(b"  <~  87cUR  ~>  ")
        assert result == b"Hell"


class TestAscii85DecodeMarkerVariations:
    """Tests for various marker combinations"""

    def test_only_trailing_tilde(self):
        result = ascii85decode(b"87cUR~")
        assert result == b"Hell"

    def test_tilde_with_spaces_in_ending(self):
        result = ascii85decode(b"87cUR ~ >")
        assert result == b"Hell"

    def test_leading_tilde_only(self):
        result = ascii85decode(b"~87cUR")
        assert result == b"Hell"

    def test_tilde_greater_with_spaces(self):
        result = ascii85decode(b"87cUR  ~  >")
        assert result == b"Hell"


class TestAscii85DecodeZShortcut:
    """Tests for the 'z' shortcut (encodes four null bytes)"""

    def test_z_shortcut(self):
        result = ascii85decode(b"z")
        assert result == b"\x00\x00\x00\x00"

    def test_multiple_z_shortcuts(self):
        result = ascii85decode(b"zz")
        assert result == b"\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_z_mixed_with_regular(self):
        result = ascii85decode(b"87cURz")
        assert result == b"Hell\x00\x00\x00\x00"


class TestAscii85DecodePadding:
    """Tests for padding behavior (non-multiple of 4 bytes)"""

    def test_one_byte_padding(self):
        result = ascii85decode(b"!!")
        assert len(result) == 1

    def test_two_byte_padding(self):
        result = ascii85decode(b"!!!")
        assert len(result) == 2

    def test_three_byte_padding(self):
        result = ascii85decode(b"!!!!")
        assert len(result) == 3

    def test_full_group_no_padding(self):
        result = ascii85decode(b"!!!!!")
        assert len(result) == 4


class TestAscii85DecodeBinaryData:
    """Tests for binary data encoding/decoding"""

    def test_all_zeros(self):
        result = ascii85decode(b"!!!!!")
        assert result == b"\x00\x00\x00\x00"

    def test_z_for_all_zeros(self):
        result = ascii85decode(b"z")
        assert result == b"\x00\x00\x00\x00"

    def test_all_ones(self):
        result = ascii85decode(b"s8W-!")
        assert result == b"\xff\xff\xff\xff"

    def test_mixed_binary(self):
        result = ascii85decode(b"!!!$!")
        assert result == b"\x00\x00\x00\xff"


class TestAscii85DecodeRealWorld:
    """Tests with real-world examples from PDFs"""

    def test_typical_pdf_stream(self):
        encoded = b'<~87cURD]i,"EW~>'
        result = ascii85decode(encoded)
        assert result == b"Hello Wor"

    def test_longer_string(self):
        encoded = b'87cURD_*#4DfTZ)+T'
        result = ascii85decode(encoded)
        assert result == b"Hello, World!"

    def test_long_encoded_string(self):
        encoded = (
            b"9jqo^BlbD-BleB1DJ+*+F(f,q/0JhKF<GL>Cj@.4Gp$d7F!,L7@<6@)/0JDEF<G%<"
            b"+EV:2F!,O<DJ+*.@<*K0@<6L(Df-\\0Ec5e;DffZ(EZee.Bl.9pF\"AGXBPCsi+DGm>"
            b"@3BB/F*&OCAfu2/AKYi(DIb:@FD,*)+C]U=@3BN#EcYr5Fh@K"
        )
        result = ascii85decode(encoded)
        assert len(result) > 0
        assert b"Man is" in result


class TestAsciihexDecodeBasic:
    """Basic tests for asciihexdecode function"""

    def test_empty_input(self):
        result = asciihexdecode(b"")
        assert result == b""

    def test_simple_hex(self):
        result = asciihexdecode(b"48656c6c6f")
        assert result == b"Hello"

    def test_uppercase_hex(self):
        result = asciihexdecode(b"48656C6C6F")
        assert result == b"Hello"

    def test_mixed_case_hex(self):
        result = asciihexdecode(b"48656C6c6F")
        assert result == b"Hello"


class TestAsciihexDecodeWhitespace:
    """Tests for whitespace handling in asciihexdecode"""

    def test_spaces_ignored(self):
        result = asciihexdecode(b"48 65 6c 6c 6f")
        assert result == b"Hello"

    def test_tabs_ignored(self):
        result = asciihexdecode(b"48\t65\t6c\t6c\t6f")
        assert result == b"Hello"

    def test_newlines_ignored(self):
        result = asciihexdecode(b"48\n65\n6c\n6c\n6f")
        assert result == b"Hello"

    def test_mixed_whitespace_ignored(self):
        result = asciihexdecode(b"48 \t\n 65 \t\n 6c 6c 6f")
        assert result == b"Hello"

    def test_leading_whitespace_ignored(self):
        result = asciihexdecode(b"   48656c6c6f")
        assert result == b"Hello"

    def test_trailing_whitespace_ignored(self):
        result = asciihexdecode(b"48656c6c6f   ")
        assert result == b"Hello"


class TestAsciihexDecodeEodMarker:
    """Tests for EOD (End-of-Data) marker handling"""

    def test_eod_marker_truncates(self):
        result = asciihexdecode(b"48656c6c6f>garbage")
        assert result == b"Hello"

    def test_eod_marker_only(self):
        result = asciihexdecode(b">")
        assert result == b""

    def test_eod_marker_with_leading_data(self):
        result = asciihexdecode(b"4865>")
        assert result == b"He"

    def test_eod_marker_after_whitespace(self):
        result = asciihexdecode(b"48 65 6c 6c 6f >")
        assert result == b"Hello"


class TestAsciihexDecodeOddDigits:
    """Tests for odd number of hex digits (padding with 0)"""

    def test_odd_digits_padded(self):
        result = asciihexdecode(b"F>")
        assert result == b"\xf0"

    def test_single_digit_with_eod(self):
        result = asciihexdecode(b"A>")
        assert result == b"\xa0"

    def test_three_digits_with_eod(self):
        result = asciihexdecode(b"123>")
        assert result == b"\x12\x30"

    def test_five_digits_with_eod(self):
        result = asciihexdecode(b"12345>")
        assert result == b"\x12\x34\x50"

    def test_odd_digits_no_eod_even_length(self):
        result = asciihexdecode(b"48656c6c6f00")
        assert result == b"Hello\x00"


class TestAsciihexDecodeBinaryData:
    """Tests for binary data encoding/decoding"""

    def test_all_zeros(self):
        result = asciihexdecode(b"00000000")
        assert result == b"\x00\x00\x00\x00"

    def test_all_ff(self):
        result = asciihexdecode(b"ffffffff")
        assert result == b"\xff\xff\xff\xff"

    def test_mixed_binary(self):
        result = asciihexdecode(b"000000ff")
        assert result == b"\x00\x00\x00\xff"

    def test_null_bytes(self):
        result = asciihexdecode(b"00")
        assert result == b"\x00"

    def test_ff_bytes(self):
        result = asciihexdecode(b"ff")
        assert result == b"\xff"


class TestAsciihexDecodeRealWorld:
    """Tests with real-world examples from PDFs"""

    def test_typical_pdf_hex_stream(self):
        encoded = b"48 65 6C 6C 6F 20 57 6F 72 6C 64>"
        result = asciihexdecode(encoded)
        assert result == b"Hello World"

    def test_long_hex_string(self):
        data = b"4D616E206973206469737469"
        result = asciihexdecode(data)
        assert result == b"Man is disti"


class TestAscii85RoundTrip:
    """Tests for encoding/decoding round-trip verification using base64.a85encode"""

    def test_roundtrip_simple_text(self):
        from base64 import a85encode

        original = b"Hello, World!"
        encoded = a85encode(original)
        decoded = ascii85decode(encoded)
        assert decoded == original

    def test_roundtrip_binary_data(self):
        from base64 import a85encode

        original = bytes(range(256))
        encoded = a85encode(original)
        decoded = ascii85decode(encoded)
        assert decoded == original

    def test_roundtrip_null_bytes(self):
        from base64 import a85encode

        original = b"\x00" * 100
        encoded = a85encode(original)
        decoded = ascii85decode(encoded)
        assert decoded == original

    def test_roundtrip_random_lengths(self):
        from base64 import a85encode

        for length in range(1, 20):
            original = bytes(range(length))
            encoded = a85encode(original)
            decoded = ascii85decode(encoded)
            assert decoded == original

    def test_roundtrip_with_markers(self):
        from base64 import a85encode

        original = b"Test data with markers"
        encoded = a85encode(original)
        decoded_with_markers = ascii85decode(b"<~" + encoded + b"~>")
        assert decoded_with_markers == original


class TestAsciihexRoundTrip:
    """Tests for hex encoding/decoding round-trip verification"""

    def test_roundtrip_simple_text(self):
        from binascii import hexlify

        original = b"Hello, World!"
        encoded = hexlify(original)
        decoded = asciihexdecode(encoded)
        assert decoded == original

    def test_roundtrip_binary_data(self):
        from binascii import hexlify

        original = bytes(range(256))
        encoded = hexlify(original)
        decoded = asciihexdecode(encoded)
        assert decoded == original

    def test_roundtrip_null_bytes(self):
        from binascii import hexlify

        original = b"\x00" * 100
        encoded = hexlify(original)
        decoded = asciihexdecode(encoded)
        assert decoded == original

    def test_roundtrip_with_eod_marker(self):
        from binascii import hexlify

        original = b"Test data"
        encoded = hexlify(original) + b">"
        decoded = asciihexdecode(encoded)
        assert decoded == original


class TestAscii85DecodeEdgeCases:
    """Edge case tests for ascii85decode"""

    def test_only_markers(self):
        result = ascii85decode(b"<~~>")
        assert result == b""

    def test_only_whitespace(self):
        result = ascii85decode(b"   ")
        assert result == b""

    def test_markers_with_whitespace_only(self):
        result = ascii85decode(b"<~   ~>")
        assert result == b""

    def test_special_chars_in_encoded_data(self):
        result = ascii85decode(b"@:E_W")
        assert len(result) > 0

    def test_minimum_valid_input(self):
        result = ascii85decode(b"!!")
        assert len(result) == 1

    def test_y_shortcut_for_spaces(self):
        from base64 import a85decode

        result = a85decode(b"y", foldspaces=True)
        assert result == b"    "


class TestAsciihexDecodeEdgeCases:
    """Edge case tests for asciihexdecode"""

    def test_only_eod_marker(self):
        result = asciihexdecode(b">")
        assert result == b""

    def test_only_whitespace(self):
        result = asciihexdecode(b"   ")
        assert result == b""

    def test_whitespace_before_eod(self):
        result = asciihexdecode(b"   >")
        assert result == b""

    def test_multiple_eod_markers(self):
        result = asciihexdecode(b"4865>>")
        assert result == b"He"

    def test_eod_in_middle(self):
        result = asciihexdecode(b"48>65")
        assert result == b"H"

    def test_single_byte(self):
        result = asciihexdecode(b"41")
        assert result == b"A"

    def test_boundary_hex_values(self):
        result = asciihexdecode(b"00FF")
        assert result == b"\x00\xff"


class TestAsciihexDecodeInvalidInput:
    """Tests for invalid hex input handling"""

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError):
            asciihexdecode(b"ZZ")

    def test_invalid_char_g_raises(self):
        with pytest.raises(ValueError):
            asciihexdecode(b"4g65")

    def test_valid_boundary_chars(self):
        result = asciihexdecode(b"09afAF")
        assert result == b"\x09\xaf\xaf"

    def test_invalid_char_before_eod(self):
        with pytest.raises(ValueError):
            asciihexdecode(b"4X>")


class TestAscii85DecodeInvalidInput:
    """Tests for invalid ASCII85 input handling"""

    def test_invalid_high_byte_raises(self):
        with pytest.raises(ValueError):
            ascii85decode(b"\xff\xff")

    def test_overflow_value_raises(self):
        with pytest.raises(ValueError):
            ascii85decode(b"uuuuu")


class TestRegexPatterns:
    """Tests for the regex patterns used in the module"""

    def test_start_pattern_matches_less_tilde(self):
        from pdfminer.ascii85 import start_re

        assert start_re.match(b"<~test") is not None

    def test_start_pattern_matches_tilde_only(self):
        from pdfminer.ascii85 import start_re

        assert start_re.match(b"~test") is not None

    def test_start_pattern_matches_with_whitespace(self):
        from pdfminer.ascii85 import start_re

        assert start_re.match(b"  <  ~  test") is not None

    def test_end_pattern_matches_tilde_greater(self):
        from pdfminer.ascii85 import end_re

        assert end_re.search(b"test~>") is not None

    def test_end_pattern_matches_tilde_only(self):
        from pdfminer.ascii85 import end_re

        assert end_re.search(b"test~") is not None

    def test_end_pattern_matches_with_whitespace(self):
        from pdfminer.ascii85 import end_re

        assert end_re.search(b"test  ~  >  ") is not None

    def test_bws_pattern_matches_spaces(self):
        from pdfminer.ascii85 import bws_re

        assert bws_re.search(b" ") is not None

    def test_bws_pattern_matches_tabs(self):
        from pdfminer.ascii85 import bws_re

        assert bws_re.search(b"\t") is not None

    def test_bws_pattern_matches_newlines(self):
        from pdfminer.ascii85 import bws_re

        assert bws_re.search(b"\n") is not None

    def test_bws_pattern_matches_carriage_return(self):
        from pdfminer.ascii85 import bws_re

        assert bws_re.search(b"\r") is not None

    def test_start_pattern_strips_correctly(self):
        from pdfminer.ascii85 import start_re

        result = start_re.sub(b"", b"<~ data")
        assert result == b"data"

    def test_end_pattern_strips_correctly(self):
        from pdfminer.ascii85 import end_re

        result = end_re.sub(b"", b"data ~>")
        assert result == b"data"

    def test_bws_pattern_removes_all_whitespace(self):
        from pdfminer.ascii85 import bws_re

        result = bws_re.sub(b"", b"a b\tc\nd\re")
        assert result == b"abcde"


class TestAscii85WithInternalWhitespace:
    """Tests for ASCII85 decoding with internal whitespace (handled by a85decode)"""

    def test_internal_space_handled(self):
        from base64 import a85encode

        original = b"Test"
        encoded = a85encode(original)
        spaced = encoded[:2] + b" " + encoded[2:]
        result = ascii85decode(spaced)
        assert result == original

    def test_internal_newline_handled(self):
        from base64 import a85encode

        original = b"Test"
        encoded = a85encode(original)
        with_newline = encoded[:2] + b"\n" + encoded[2:]
        result = ascii85decode(with_newline)
        assert result == original


class TestAsciihexDecodeCarriageReturn:
    """Tests for carriage return handling in asciihexdecode"""

    def test_carriage_return_ignored(self):
        result = asciihexdecode(b"48\r65\r6c\r6c\r6f")
        assert result == b"Hello"

    def test_crlf_ignored(self):
        result = asciihexdecode(b"48\r\n65\r\n6c6c6f")
        assert result == b"Hello"


class TestAsciihexDecodeLowerBoundary:
    """Tests for hex decoding at character boundaries"""

    def test_digit_0(self):
        result = asciihexdecode(b"30")
        assert result == b"0"

    def test_digit_9(self):
        result = asciihexdecode(b"39")
        assert result == b"9"

    def test_lowercase_a(self):
        result = asciihexdecode(b"61")
        assert result == b"a"

    def test_lowercase_f_value(self):
        result = asciihexdecode(b"0f")
        assert result == b"\x0f"

    def test_uppercase_a_value(self):
        result = asciihexdecode(b"0A")
        assert result == b"\x0a"

    def test_uppercase_f_value(self):
        result = asciihexdecode(b"0F")
        assert result == b"\x0f"
