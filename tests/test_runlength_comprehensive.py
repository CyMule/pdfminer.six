"""Comprehensive tests for pdfminer/runlength.py

Tests the RunLength decoder (Adobe version) based on PDF Reference
version 1.4 section 3.3.4.

Run-length encoding format:
- Length byte 0-127: Copy the following (length + 1) bytes literally
- Length byte 129-255: Repeat the following byte (257 - length) times
- Length byte 128: End of data (EOD marker)
"""

import pytest

from pdfminer.runlength import rldecode


class TestRldecodeBasicLiteralCopy:
    """Tests for literal copy mode (length byte 0-127)"""

    def test_single_literal_byte(self):
        """Length 0 means copy 1 byte literally"""
        encoded = bytes([0, 0x41, 128])
        result = rldecode(encoded)
        assert result == b"A"

    def test_two_literal_bytes(self):
        """Length 1 means copy 2 bytes literally"""
        encoded = bytes([1, 0x41, 0x42, 128])
        result = rldecode(encoded)
        assert result == b"AB"

    def test_max_literal_bytes(self):
        """Length 127 means copy 128 bytes literally"""
        data = bytes(range(128))
        encoded = bytes([127]) + data + bytes([128])
        result = rldecode(encoded)
        assert result == data
        assert len(result) == 128

    def test_multiple_literal_runs(self):
        """Multiple consecutive literal copy runs"""
        encoded = bytes([2, 0x41, 0x42, 0x43, 1, 0x44, 0x45, 128])
        result = rldecode(encoded)
        assert result == b"ABCDE"

    def test_literal_with_zeros(self):
        """Literal copy containing zero bytes"""
        encoded = bytes([2, 0x00, 0x00, 0x00, 128])
        result = rldecode(encoded)
        assert result == bytes([0, 0, 0])

    def test_literal_with_high_bytes(self):
        """Literal copy containing high byte values (0x80-0xFF)"""
        encoded = bytes([3, 0x80, 0xFF, 0xFE, 0xFD, 128])
        result = rldecode(encoded)
        assert result == bytes([0x80, 0xFF, 0xFE, 0xFD])


class TestRldecodeRepeatMode:
    """Tests for repeat mode (length byte 129-255)"""

    def test_repeat_two_times(self):
        """Length 255 means repeat following byte 2 times (257 - 255 = 2)"""
        encoded = bytes([255, 0x41, 128])
        result = rldecode(encoded)
        assert result == b"AA"

    def test_repeat_three_times(self):
        """Length 254 means repeat following byte 3 times (257 - 254 = 3)"""
        encoded = bytes([254, 0x42, 128])
        result = rldecode(encoded)
        assert result == b"BBB"

    def test_repeat_max_128_times(self):
        """Length 129 means repeat following byte 128 times (257 - 129 = 128)"""
        encoded = bytes([129, 0x58, 128])
        result = rldecode(encoded)
        assert result == b"X" * 128
        assert len(result) == 128

    def test_repeat_zero_byte(self):
        """Repeat zero byte multiple times"""
        encoded = bytes([253, 0x00, 128])
        result = rldecode(encoded)
        assert result == bytes([0, 0, 0, 0])

    def test_repeat_high_byte(self):
        """Repeat high byte value (0xFF)"""
        encoded = bytes([252, 0xFF, 128])
        result = rldecode(encoded)
        assert result == bytes([0xFF] * 5)

    def test_multiple_repeat_runs(self):
        """Multiple consecutive repeat runs"""
        encoded = bytes([254, 0x41, 254, 0x42, 128])
        result = rldecode(encoded)
        assert result == b"AAABBB"


class TestRldecodeMixedModes:
    """Tests for mixed literal and repeat modes"""

    def test_literal_then_repeat(self):
        """Literal copy followed by repeat run"""
        encoded = bytes([2, 0x41, 0x42, 0x43, 254, 0x44, 128])
        result = rldecode(encoded)
        assert result == b"ABCDDD"

    def test_repeat_then_literal(self):
        """Repeat run followed by literal copy"""
        encoded = bytes([254, 0x41, 2, 0x42, 0x43, 0x44, 128])
        result = rldecode(encoded)
        assert result == b"AAABCD"

    def test_alternating_modes(self):
        """Alternating between literal and repeat modes"""
        encoded = bytes([
            1, 0x48, 0x45,
            254, 0x4C,
            0, 0x4F,
            128
        ])
        result = rldecode(encoded)
        assert result == b"HELLLO"

    def test_complex_mixed_sequence(self):
        """Complex sequence mixing literal and repeat modes"""
        encoded = bytes([
            3, 0x54, 0x45, 0x53, 0x54,
            253, 0x2D,
            2, 0x52, 0x55, 0x4E,
            255, 0x21,
            128
        ])
        result = rldecode(encoded)
        assert result == b"TEST----RUN!!"


class TestRldecodeEODMarker:
    """Tests for End of Data (EOD) marker (length byte 128)"""

    def test_immediate_eod(self):
        """EOD marker at start produces empty output"""
        encoded = bytes([128])
        result = rldecode(encoded)
        assert result == b""

    def test_eod_after_literal(self):
        """EOD marker after literal copy"""
        encoded = bytes([2, 0x41, 0x42, 0x43, 128])
        result = rldecode(encoded)
        assert result == b"ABC"

    def test_eod_after_repeat(self):
        """EOD marker after repeat run"""
        encoded = bytes([254, 0x58, 128])
        result = rldecode(encoded)
        assert result == b"XXX"

    def test_data_after_eod_ignored(self):
        """Data after EOD marker is ignored"""
        encoded = bytes([0, 0x41, 128, 0, 0x42, 254, 0x43, 128])
        result = rldecode(encoded)
        assert result == b"A"

    def test_multiple_eod_markers(self):
        """First EOD marker terminates decoding"""
        encoded = bytes([0, 0x41, 128, 128, 128])
        result = rldecode(encoded)
        assert result == b"A"


class TestRldecodeEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_input(self):
        """Empty input produces empty output (implicit EOD)"""
        result = rldecode(b"")
        assert result == b""

    def test_no_explicit_eod(self):
        """Input without explicit EOD marker (iterator exhaustion)"""
        encoded = bytes([1, 0x41, 0x42])
        result = rldecode(encoded)
        assert result == b"AB"

    def test_no_eod_after_repeat(self):
        """Repeat run without EOD marker"""
        encoded = bytes([254, 0x58])
        result = rldecode(encoded)
        assert result == b"XXX"

    def test_boundary_literal_length_127(self):
        """Boundary test: length 127 (max literal)"""
        data = bytes([i % 256 for i in range(128)])
        encoded = bytes([127]) + data + bytes([128])
        result = rldecode(encoded)
        assert result == data

    def test_boundary_repeat_length_129(self):
        """Boundary test: length 129 (max repeat count)"""
        encoded = bytes([129, 0x41, 128])
        result = rldecode(encoded)
        assert result == b"A" * 128

    def test_boundary_repeat_length_255(self):
        """Boundary test: length 255 (min repeat count)"""
        encoded = bytes([255, 0x41, 128])
        result = rldecode(encoded)
        assert result == b"AA"

    def test_single_byte_value_128_data(self):
        """Literal copy of the byte value 128 (same as EOD marker)"""
        encoded = bytes([0, 128, 128])
        result = rldecode(encoded)
        assert result == bytes([128])

    def test_repeat_byte_value_128(self):
        """Repeat the byte value 128"""
        encoded = bytes([254, 128, 128])
        result = rldecode(encoded)
        assert result == bytes([128, 128, 128])


class TestRldecodeLargeData:
    """Tests for large data handling"""

    def test_large_literal_sequence(self):
        """Multiple max-length literal runs"""
        data_chunk = bytes(range(128))
        encoded = bytes([127]) + data_chunk + bytes([127]) + data_chunk + bytes([128])
        result = rldecode(encoded)
        assert result == data_chunk + data_chunk
        assert len(result) == 256

    def test_large_repeat_sequence(self):
        """Multiple max-length repeat runs"""
        encoded = bytes([129, 0x41, 129, 0x42, 128])
        result = rldecode(encoded)
        assert result == b"A" * 128 + b"B" * 128
        assert len(result) == 256

    def test_many_small_runs(self):
        """Many small runs (stress test for iterator)"""
        runs = []
        for i in range(100):
            runs.extend([0, i % 256])
        runs.append(128)
        encoded = bytes(runs)
        result = rldecode(encoded)
        assert len(result) == 100

    def test_alternating_large_runs(self):
        """Alternating large literal and repeat runs"""
        literal_data = bytes([0x55] * 128)
        encoded = (
            bytes([127]) + literal_data +
            bytes([129, 0xAA]) +
            bytes([127]) + literal_data +
            bytes([129, 0xBB]) +
            bytes([128])
        )
        result = rldecode(encoded)
        expected = (
            literal_data + bytes([0xAA] * 128) + literal_data + bytes([0xBB] * 128)
        )
        assert result == expected
        assert len(result) == 512


class TestRldecodeBinaryData:
    """Tests for various binary data patterns"""

    def test_all_zero_bytes(self):
        """Data consisting entirely of zero bytes"""
        encoded = bytes([129, 0x00, 128])
        result = rldecode(encoded)
        assert result == bytes([0] * 128)

    def test_all_0xff_bytes(self):
        """Data consisting entirely of 0xFF bytes"""
        encoded = bytes([129, 0xFF, 128])
        result = rldecode(encoded)
        assert result == bytes([0xFF] * 128)

    def test_increasing_byte_pattern(self):
        """Literal copy of increasing byte pattern"""
        data = bytes(range(10))
        encoded = bytes([9]) + data + bytes([128])
        result = rldecode(encoded)
        assert result == data

    def test_decreasing_byte_pattern(self):
        """Literal copy of decreasing byte pattern"""
        data = bytes(range(9, -1, -1))
        encoded = bytes([9]) + data + bytes([128])
        result = rldecode(encoded)
        assert result == data

    def test_alternating_byte_pattern(self):
        """Literal copy of alternating bytes"""
        data = bytes([0x55, 0xAA] * 5)
        encoded = bytes([9]) + data + bytes([128])
        result = rldecode(encoded)
        assert result == data


class TestRldecodeRealWorldPatterns:
    """Tests simulating real-world run-length encoded data patterns"""

    def test_image_scanline_simulation(self):
        """Simulate a simple grayscale image scanline with repeated pixels"""
        encoded = bytes([
            129, 0x00,
            129, 0xFF,
            129, 0x00,
            128
        ])
        result = rldecode(encoded)
        assert result == bytes([0x00] * 128 + [0xFF] * 128 + [0x00] * 128)

    def test_sparse_data_pattern(self):
        """Sparse data: mostly zeros with occasional non-zero values"""
        encoded = bytes([
            253, 0x00,
            0, 0xFF,
            253, 0x00,
            0, 0xFE,
            253, 0x00,
            128
        ])
        result = rldecode(encoded)
        expected = bytes([0, 0, 0, 0, 0xFF, 0, 0, 0, 0, 0xFE, 0, 0, 0, 0])
        assert result == expected

    def test_text_like_data(self):
        """Text-like data with repeated spaces"""
        encoded = bytes([
            4, ord('H'), ord('e'), ord('l'), ord('l'), ord('o'),
            253, ord(' '),
            4, ord('W'), ord('o'), ord('r'), ord('l'), ord('d'),
            128
        ])
        result = rldecode(encoded)
        assert result == b"Hello    World"


class TestRldecodeReturnType:
    """Tests verifying return type consistency"""

    def test_returns_bytes_type(self):
        """rldecode always returns bytes type"""
        result = rldecode(bytes([128]))
        assert isinstance(result, bytes)

    def test_empty_result_is_bytes(self):
        """Empty result is bytes type"""
        result = rldecode(b"")
        assert isinstance(result, bytes)
        assert result == b""

    def test_result_from_literal_is_bytes(self):
        """Result from literal copy is bytes type"""
        result = rldecode(bytes([0, 0x41, 128]))
        assert isinstance(result, bytes)

    def test_result_from_repeat_is_bytes(self):
        """Result from repeat run is bytes type"""
        result = rldecode(bytes([255, 0x41, 128]))
        assert isinstance(result, bytes)


class TestRldecodeInputTypes:
    """Tests for various input types"""

    def test_bytes_input(self):
        """Standard bytes input"""
        result = rldecode(bytes([0, 0x41, 128]))
        assert result == b"A"

    def test_bytearray_input(self):
        """Bytearray input (should work as iterable)"""
        result = rldecode(bytearray([0, 0x41, 128]))
        assert result == b"A"

    def test_memoryview_input(self):
        """Memoryview input (should work as iterable)"""
        data = bytes([0, 0x41, 128])
        result = rldecode(memoryview(data))
        assert result == b"A"


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (bytes([128]), b""),
        (bytes([0, 0x41, 128]), b"A"),
        (bytes([1, 0x41, 0x42, 128]), b"AB"),
        (bytes([255, 0x41, 128]), b"AA"),
        (bytes([254, 0x41, 128]), b"AAA"),
        (bytes([1, 0x41, 0x42, 255, 0x43, 128]), b"ABCC"),
    ],
)
def test_rldecode_parametrized(encoded: bytes, expected: bytes) -> None:
    """Parametrized tests for common decode patterns"""
    assert rldecode(encoded) == expected


@pytest.mark.parametrize(
    ("length_byte", "expected_repeat_count"),
    [
        (255, 2),
        (254, 3),
        (253, 4),
        (200, 57),
        (150, 107),
        (130, 127),
        (129, 128),
    ],
)
def test_repeat_count_formula(length_byte: int, expected_repeat_count: int) -> None:
    """Verify the repeat count formula: 257 - length"""
    encoded = bytes([length_byte, 0x58, 128])
    result = rldecode(encoded)
    assert len(result) == expected_repeat_count
    assert result == bytes([0x58] * expected_repeat_count)


@pytest.mark.parametrize(
    ("length_byte", "expected_literal_count"),
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (50, 51),
        (100, 101),
        (126, 127),
        (127, 128),
    ],
)
def test_literal_count_formula(length_byte: int, expected_literal_count: int) -> None:
    """Verify the literal count formula: length + 1"""
    data = bytes([0x41] * expected_literal_count)
    encoded = bytes([length_byte]) + data + bytes([128])
    result = rldecode(encoded)
    assert len(result) == expected_literal_count
    assert result == data


class TestRldecodeTruncatedInput:
    """Tests for handling truncated or incomplete input data.

    The rldecode function expects well-formed input and raises exceptions
    when the input is truncated. This documents the current behavior.
    """

    def test_truncated_literal_run_raises(self):
        """Literal run that expects more bytes than available raises RuntimeError.

        When a literal copy run expects more bytes than available, the generator
        expression raises StopIteration which becomes RuntimeError in Python 3.7+.
        """
        encoded = bytes([5, 0x41, 0x42])
        with pytest.raises(RuntimeError):
            rldecode(encoded)

    def test_truncated_after_length_byte_raises(self):
        """Input ending after literal length byte raises RuntimeError.

        When a literal length byte is given but no data follows, the generator
        expression raises StopIteration which becomes RuntimeError in Python 3.7+.
        """
        encoded = bytes([5])
        with pytest.raises(RuntimeError):
            rldecode(encoded)

    def test_truncated_repeat_run_raises(self):
        """Repeat run without the byte to repeat raises StopIteration.

        When a repeat length byte is given but no byte to repeat follows,
        next() raises StopIteration directly (not wrapped in RuntimeError
        since it's not in a generator context).
        """
        encoded = bytes([255])
        with pytest.raises(StopIteration):
            rldecode(encoded)

    def test_length_byte_only_literal_raises(self):
        """Only a literal length byte with no data raises RuntimeError.

        When length byte 0 is given (expect 1 byte) but no data follows,
        the generator expression raises StopIteration which becomes RuntimeError.
        """
        encoded = bytes([0])
        with pytest.raises(RuntimeError):
            rldecode(encoded)
