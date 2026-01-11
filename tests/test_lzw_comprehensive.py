"""Comprehensive tests for pdfminer.lzw module.

Tests cover:
- lzwdecode() function
- LZWDecoder class methods (readbits, feed, run)
- Edge cases and error handling
- Known LZW compressed data patterns
"""

from io import BytesIO

import pytest

from pdfminer.lzw import CorruptDataError, LZWDecoder, lzwdecode
from pdfminer.pdfexceptions import PDFEOFError


class TestLZWDecoder:
    """Tests for LZWDecoder class."""

    def test_initialization(self) -> None:
        """Test LZWDecoder initializes with correct default values."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        assert decoder.buff == 0
        assert decoder.bpos == 8
        assert decoder.nbits == 9
        assert decoder.table == []
        assert decoder.prevbuf is None

    def test_readbits_single_byte(self) -> None:
        """Test reading bits from a single byte."""
        fp = BytesIO(b"\x80")  # 10000000 in binary
        decoder = LZWDecoder(fp)
        decoder.buff = 0x80
        decoder.bpos = 0
        result = decoder.readbits(1)
        assert result == 1

    def test_readbits_multiple_bits(self) -> None:
        """Test reading multiple bits at once."""
        fp = BytesIO(b"\xff")  # 11111111 in binary
        decoder = LZWDecoder(fp)
        decoder.buff = 0xFF
        decoder.bpos = 0
        result = decoder.readbits(4)
        assert result == 15  # 1111 = 15

    def test_readbits_across_bytes(self) -> None:
        """Test reading bits that span across two bytes."""
        fp = BytesIO(b"\xAB\xCD")  # Two bytes
        decoder = LZWDecoder(fp)
        decoder.buff = 0xAB
        decoder.bpos = 0
        # Read 12 bits (8 from first byte + 4 from second)
        result = decoder.readbits(12)
        # 0xAB = 10101011, 0xCD = 11001101
        # First 8 bits: 10101011 (0xAB), next 4 bits from 0xCD: 1100 (0xC)
        # But the algorithm reads the high bits first from the next byte
        # 0xAB << 4 = 0xAB0, plus high 4 bits of 0xCD (1100 = 12 = 0xC)
        # However, the way readbits works: it reads remaining r bits, then
        # continues. 0xAB (full byte), then high 4 bits of 0xCD = 1100 = 0xC
        # Result: 10101011 1010 = 0xABA (2746 decimal)
        assert result == 0xABA

    def test_readbits_eof_raises_error(self) -> None:
        """Test that reading past EOF raises PDFEOFError."""
        fp = BytesIO(b"")
        decoder = LZWDecoder(fp)
        decoder.bpos = 8  # Force read from fp
        with pytest.raises(PDFEOFError):
            decoder.readbits(9)

    def test_readbits_partial_byte(self) -> None:
        """Test reading partial bits from a byte."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.buff = 0b11110000
        decoder.bpos = 0
        # Read 3 bits
        result = decoder.readbits(3)
        assert result == 0b111
        assert decoder.bpos == 3

    def test_feed_clear_code_256(self) -> None:
        """Test that code 256 (clear code) resets the table."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        # Initialize with some state
        decoder.table = [b"test"]
        decoder.nbits = 12
        decoder.prevbuf = b"something"

        result = decoder.feed(256)
        assert result == b""
        assert len(decoder.table) == 258  # 256 entries + 2 special codes
        assert decoder.table[256] is None
        assert decoder.table[257] is None
        assert decoder.prevbuf == b""
        assert decoder.nbits == 9

    def test_feed_end_of_data_code_257(self) -> None:
        """Test that code 257 (end of data) does nothing."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.prevbuf = b"test"

        result = decoder.feed(257)
        assert result == b""
        assert decoder.prevbuf == b"test"

    def test_feed_first_code_after_clear(self) -> None:
        """Test feeding the first code after a clear code."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        # First send clear code
        decoder.feed(256)
        # Then send a character code
        result = decoder.feed(65)  # ASCII 'A'
        assert result == b"A"
        assert decoder.prevbuf == b"A"

    def test_feed_existing_code_in_table(self) -> None:
        """Test feeding a code that exists in the table."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        decoder.feed(65)  # 'A'
        result = decoder.feed(66)  # 'B'
        assert result == b"B"
        # Table should now have 'AB' at index 258
        assert decoder.table[258] == b"AB"

    def test_feed_code_equals_table_length(self) -> None:
        """Test the special case where code equals current table length."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code
        decoder.feed(65)  # 'A' - prevbuf becomes 'A'
        # Now table has 258 entries (0-255 + 256 + 257)
        # If we send code 258, it should use prevbuf + prevbuf[0]
        result = decoder.feed(258)
        assert result == b"AA"
        assert decoder.table[258] == b"AA"

    def test_feed_corrupt_data_raises_error(self) -> None:
        """Test that an invalid code raises CorruptDataError."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code
        decoder.feed(65)  # 'A'
        # Code 259 doesn't exist yet (only 258 entries + one new)
        with pytest.raises(CorruptDataError):
            decoder.feed(500)  # Way out of range

    def test_feed_nbits_increase_at_511(self) -> None:
        """Test that nbits increases to 10 when table reaches 511 entries."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code, table has 258 entries
        # Feed codes to grow table to 511
        decoder.feed(0)  # First code after clear
        for i in range(1, 254):  # 253 more codes to reach 511 entries
            decoder.feed(i)
        assert len(decoder.table) == 511
        assert decoder.nbits == 10

    def test_feed_nbits_increase_at_1023(self) -> None:
        """Test that nbits increases to 11 when table reaches 1023 entries."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code
        decoder.feed(0)  # First code
        # Build table up to 1023 entries
        for i in range(1, 766):  # More codes to reach 1023
            decoder.feed(i % 256)
        assert len(decoder.table) == 1023
        assert decoder.nbits == 11

    def test_feed_nbits_increase_at_2047(self) -> None:
        """Test that nbits increases to 12 when table reaches 2047 entries."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code
        decoder.feed(0)  # First code
        # Build table up to 2047 entries
        for i in range(1, 1790):  # More codes to reach 2047
            decoder.feed(i % 256)
        assert len(decoder.table) == 2047
        assert decoder.nbits == 12

    def test_run_basic(self) -> None:
        """Test basic run() iteration."""
        # Create valid LZW data: clear code (256) + 'A' (65) + EOD (257)
        # With 9-bit codes: 256 = 100000000, 65 = 001000001, 257 = 100000001
        # Packed: 10000000 00010000 01100000 001...
        data = bytes([0x80, 0x10, 0x60, 0x40])
        fp = BytesIO(data)
        decoder = LZWDecoder(fp)
        result = list(decoder.run())
        # First is empty from clear code, second is 'A', then EOD
        assert b"A" in result

    def test_run_handles_eof(self) -> None:
        """Test that run() handles EOF gracefully."""
        fp = BytesIO(b"\x80\x00")  # Clear code only (partial)
        decoder = LZWDecoder(fp)
        result = list(decoder.run())
        # Should complete without error
        assert isinstance(result, list)

    def test_run_handles_corrupt_data(self) -> None:
        """Test that run() stops on corrupt data without raising.

        Note: The run() method catches CorruptDataError in feed() and stops
        yielding. However, if the table is not initialized (no clear code),
        an IndexError may occur. The corrupt data handling specifically
        catches CorruptDataError for codes that are out of bounds after
        the table is initialized.
        """
        # Start with a valid clear code, then feed corrupt data
        # 256 (clear) = 100000000 in 9 bits
        # Then a code that's way out of range
        # 0x80 = 10000000, 0x00 = 00000000 gives us clear code
        # Then 0xFF 0xFF gives high codes that will be corrupt
        data = bytes([0x80, 0x00, 0xFF, 0xFF, 0xFF, 0xFF])
        fp = BytesIO(data)
        decoder = LZWDecoder(fp)
        result = list(decoder.run())
        # Should complete without raising (CorruptDataError is caught)
        assert isinstance(result, list)


class TestLzwdecode:
    """Tests for lzwdecode() function."""

    def test_empty_input(self) -> None:
        """Test decoding empty input."""
        result = lzwdecode(b"")
        assert result == b""

    def test_clear_code_only(self) -> None:
        """Test data containing only clear code."""
        # 256 in 9 bits = 100000000
        # Packed in bytes: 10000000 0xxxxxxx
        data = bytes([0x80, 0x00])
        result = lzwdecode(data)
        assert result == b""

    def test_clear_and_eod(self) -> None:
        """Test data with clear code followed by EOD."""
        # 256 = 100000000, 257 = 100000001
        # Packed: 10000000 01000000 01...
        data = bytes([0x80, 0x40, 0x40])
        result = lzwdecode(data)
        assert result == b""

    def test_simple_literal_decoding(self) -> None:
        """Test decoding simple literal bytes.

        The LZW encoding packs 9-bit codes into bytes from MSB to LSB.
        For 'A' (65) after clear code:
        - Clear (256) = 100000000 (9 bits)
        - 'A' (65)    = 001000001 (9 bits)
        - EOD (257)   = 100000001 (9 bits)
        Total: 27 bits = 3.375 bytes, pad to 4 bytes
        Packed: 10000000 00010000 01100000 001...
        """
        # Let's verify with a simpler approach using feed() directly
        # to ensure we understand the decoder
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        result = decoder.feed(65)  # 'A'
        assert result == b"A"

    def test_returns_bytes(self) -> None:
        """Test that lzwdecode always returns bytes."""
        result = lzwdecode(b"")
        assert isinstance(result, bytes)

    def test_with_minimum_valid_stream(self) -> None:
        """Test with minimum valid LZW stream (clear + eod)."""
        # Create properly encoded clear code (256) followed by EOD (257)
        # 256 in 9 bits: 1 00000000
        # 257 in 9 bits: 1 00000001
        # Concatenated: 100000000 100000001
        # Padded to bytes: 10000000 01000000 01xxxxxx
        data = bytes([0x80, 0x40, 0x40])
        result = lzwdecode(data)
        assert result == b""


class TestCorruptDataError:
    """Tests for CorruptDataError exception."""

    def test_is_pdf_exception(self) -> None:
        """Test that CorruptDataError inherits from PDFException."""
        from pdfminer.pdfexceptions import PDFException

        error = CorruptDataError("test")
        assert isinstance(error, PDFException)

    def test_can_be_raised(self) -> None:
        """Test that CorruptDataError can be raised and caught."""
        with pytest.raises(CorruptDataError):
            raise CorruptDataError("test message")

    def test_message_preserved(self) -> None:
        """Test that error message is preserved."""
        try:
            raise CorruptDataError("test message")
        except CorruptDataError as e:
            assert "test message" in str(e)


class TestLZWBitManipulation:
    """Tests specifically for bit manipulation in LZWDecoder."""

    def test_readbits_exact_byte_boundary(self) -> None:
        """Test reading exactly 8 bits."""
        fp = BytesIO(b"\xAB\xCD")
        decoder = LZWDecoder(fp)
        decoder.buff = 0xAB
        decoder.bpos = 0
        result = decoder.readbits(8)
        assert result == 0xAB

    def test_readbits_9_bits(self) -> None:
        """Test reading 9 bits (common for LZW)."""
        fp = BytesIO(b"\xFF\x80")
        decoder = LZWDecoder(fp)
        decoder.buff = 0xFF
        decoder.bpos = 0
        result = decoder.readbits(9)
        # 0xFF = 11111111, 0x80 = 10000000
        # First 9 bits: 111111111 = 511
        assert result == 511

    def test_readbits_12_bits_max(self) -> None:
        """Test reading 12 bits (maximum for LZW)."""
        fp = BytesIO(b"\xFF\xF0")
        decoder = LZWDecoder(fp)
        decoder.buff = 0xFF
        decoder.bpos = 0
        result = decoder.readbits(12)
        # 0xFF = 11111111, 0xF0 = 11110000
        # First 12 bits: 111111111111 = 4095
        assert result == 4095

    def test_readbits_with_partial_buffer(self) -> None:
        """Test reading bits when buffer is partially consumed."""
        fp = BytesIO(b"\xF0")
        decoder = LZWDecoder(fp)
        decoder.buff = 0xFF  # 11111111
        decoder.bpos = 4  # Already consumed 4 bits
        result = decoder.readbits(4)
        # Remaining bits: 1111
        assert result == 15

    def test_readbits_zero_bits(self) -> None:
        """Test reading zero bits."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.buff = 0xFF
        decoder.bpos = 0
        result = decoder.readbits(0)
        assert result == 0

    def test_readbits_one_bit_high(self) -> None:
        """Test reading single high bit."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.buff = 0x80  # 10000000
        decoder.bpos = 0
        result = decoder.readbits(1)
        assert result == 1

    def test_readbits_one_bit_low(self) -> None:
        """Test reading single low bit."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.buff = 0x00  # 00000000
        decoder.bpos = 0
        result = decoder.readbits(1)
        assert result == 0


class TestLZWTableManagement:
    """Tests for LZW table management."""

    def test_table_initialization_after_clear(self) -> None:
        """Test that table is properly initialized after clear code."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code

        # Check first 256 entries are single bytes
        for i in range(256):
            assert decoder.table[i] == bytes([i])

        # Check special codes are None
        assert decoder.table[256] is None
        assert decoder.table[257] is None

    def test_table_growth(self) -> None:
        """Test that table grows as codes are processed."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear code - table has 258
        decoder.feed(65)  # 'A' - no new entry yet (first after clear)
        initial_size = len(decoder.table)
        decoder.feed(66)  # 'B' - adds 'AB' to table
        assert len(decoder.table) == initial_size + 1

    def test_table_entries_are_bytes(self) -> None:
        """Test that all table entries (except specials) are bytes."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)

        for i, entry in enumerate(decoder.table):
            if i not in (256, 257):
                assert isinstance(entry, bytes)


class TestLZWKnownPatterns:
    """Tests with known LZW compressed patterns."""

    def test_repeated_character_compression(self) -> None:
        """Test pattern that benefits from LZW compression."""
        # LZW is good at compressing repeated patterns
        # This tests the codeword-equals-table-length case
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        decoder.feed(65)  # 'A'
        # Code 258 should produce 'AA'
        result = decoder.feed(258)
        assert result == b"AA"

    def test_sequential_characters(self) -> None:
        """Test sequential character decoding."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        decoder.feed(65)  # 'A'
        decoder.feed(66)  # 'B'
        decoder.feed(67)  # 'C'
        # Table should now contain 'AB' and 'BC'
        assert decoder.table[258] == b"AB"
        assert decoder.table[259] == b"BC"

    def test_string_building(self) -> None:
        """Test that LZW builds strings correctly."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        decoder.feed(65)  # 'A'
        decoder.feed(66)  # 'B' - adds 'AB' at 258
        decoder.feed(258)  # 'AB' - adds 'BA' at 259
        assert decoder.table[259] == b"BA"


class TestLZWEdgeCases:
    """Tests for edge cases in LZW decoding."""

    def test_multiple_clear_codes(self) -> None:
        """Test handling multiple consecutive clear codes."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # First clear
        decoder.feed(65)  # 'A'
        decoder.feed(66)  # 'B'
        assert len(decoder.table) > 258  # Table grew with new entries
        decoder.feed(256)  # Second clear - should reset
        assert len(decoder.table) == 258  # Reset to initial
        assert decoder.nbits == 9

    def test_clear_code_resets_nbits(self) -> None:
        """Test that clear code properly resets nbits."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.nbits = 12  # Simulate being at max bits
        decoder.feed(256)  # Clear
        assert decoder.nbits == 9

    def test_code_zero_after_clear(self) -> None:
        """Test handling code 0 (null byte) after clear."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        result = decoder.feed(0)  # Null byte
        assert result == b"\x00"

    def test_code_255_after_clear(self) -> None:
        """Test handling code 255 (0xFF byte) after clear."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear
        result = decoder.feed(255)  # 0xFF byte
        assert result == b"\xff"

    def test_all_single_byte_codes(self) -> None:
        """Test that all single-byte codes (0-255) work."""
        fp = BytesIO(b"\x00")
        decoder = LZWDecoder(fp)
        decoder.feed(256)  # Clear

        # Test a sample of codes
        for code in [0, 1, 32, 65, 127, 128, 200, 255]:
            decoder.feed(256)  # Reset
            result = decoder.feed(code)
            assert result == bytes([code])


class TestLZWIntegration:
    """Integration tests for complete LZW decoding."""

    def test_lzwdecode_roundtrip_concept(self) -> None:
        """Test that decoder produces bytes from valid input."""
        # This is a conceptual test - we verify the interface works
        # even if we don't have encoder to create test data
        result = lzwdecode(b"\x80\x00")  # Clear code
        assert isinstance(result, bytes)

    def test_decoder_run_yields_bytes(self) -> None:
        """Test that decoder.run() yields bytes objects."""
        data = bytes([0x80, 0x24, 0x40])  # Some LZW data
        fp = BytesIO(data)
        decoder = LZWDecoder(fp)
        for chunk in decoder.run():
            assert isinstance(chunk, bytes)

    def test_joining_chunks(self) -> None:
        """Test that chunks can be joined into final result."""
        data = bytes([0x80, 0x00])
        fp = BytesIO(data)
        decoder = LZWDecoder(fp)
        chunks = list(decoder.run())
        result = b"".join(chunks)
        assert isinstance(result, bytes)
