"""Comprehensive tests for pdfminer/utils.py"""

import io
import pathlib

import pytest

from pdfminer.layout import LTComponent
from pdfminer.pdfexceptions import PDFTypeError, PDFValueError
from pdfminer.utils import (
    INF,
    MATRIX_IDENTITY,
    Plane,
    apply_matrix_norm,
    apply_matrix_pt,
    apply_matrix_rect,
    apply_png_predictor,
    apply_tiff_predictor,
    bbox2str,
    choplist,
    compatible_encode_method,
    decode_text,
    drange,
    enc,
    format_int_alpha,
    format_int_roman,
    fsplit,
    get_bound,
    isnumber,
    make_compat_bytes,
    make_compat_str,
    matrix2str,
    mult_matrix,
    nunpack,
    open_filename,
    paeth_predictor,
    parse_rect,
    pick,
    shorten_str,
    translate_matrix,
    uniq,
    unpad_aes,
)
from tests.helpers import absolute_sample_path


class TestOpenFilenameContextManager:
    """Tests for open_filename context manager"""

    def test_string_path_opens_and_closes(self):
        filename = absolute_sample_path("simple1.pdf")
        with open_filename(filename, "rb") as f:
            assert hasattr(f, "read")
            content = f.read(10)
            assert len(content) == 10

    def test_pathlib_path_opens_and_closes(self):
        filename = pathlib.Path(absolute_sample_path("simple1.pdf"))
        with open_filename(filename, "rb") as f:
            assert hasattr(f, "read")
            content = f.read(10)
            assert len(content) == 10

    def test_file_like_object_not_closed_on_exit(self):
        filename = absolute_sample_path("simple1.pdf")
        with open(filename, "rb") as real_file:
            with open_filename(real_file) as f:
                assert f is real_file
            assert not real_file.closed

    def test_bytesio_as_file_like_object(self):
        data = b"test content for bytesio"
        bytesio = io.BytesIO(data)
        with open_filename(bytesio) as f:
            assert f is bytesio
            assert f.read() == data
        assert not bytesio.closed

    def test_unsupported_type_raises_pdftypeerror(self):
        with pytest.raises(PDFTypeError):
            open_filename(12345)

    def test_unsupported_type_list_raises_pdftypeerror(self):
        with pytest.raises(PDFTypeError):
            open_filename(["not", "a", "file"])

    def test_closing_flag_true_for_string_path(self):
        filename = absolute_sample_path("simple1.pdf")
        ctx = open_filename(filename, "rb")
        assert ctx.closing is True
        ctx.file_handler.close()

    def test_closing_flag_false_for_file_object(self):
        filename = absolute_sample_path("simple1.pdf")
        with open(filename, "rb") as real_file:
            ctx = open_filename(real_file)
            assert ctx.closing is False


class TestMakeCompatBytes:
    """Tests for make_compat_bytes function"""

    def test_ascii_string_to_bytes(self):
        result = make_compat_bytes("hello")
        assert result == b"hello"

    def test_unicode_string_to_bytes(self):
        result = make_compat_bytes("hello\xe9")
        assert result == "hello\xe9".encode()

    def test_empty_string_to_bytes(self):
        result = make_compat_bytes("")
        assert result == b""

    def test_special_characters(self):
        result = make_compat_bytes("<>&")
        assert result == b"<>&"


class TestMakeCompatStr:
    """Tests for make_compat_str function"""

    def test_bytes_to_string_utf8(self):
        result = make_compat_str(b"hello")
        assert result == "hello"

    def test_string_passthrough(self):
        result = make_compat_str("already a string")
        assert result == "already a string"

    def test_integer_to_string(self):
        result = make_compat_str(42)
        assert result == "42"

    def test_float_to_string(self):
        result = make_compat_str(3.14)
        assert result == "3.14"

    def test_none_to_string(self):
        result = make_compat_str(None)
        assert result == "None"

    def test_empty_bytes_to_string(self):
        result = make_compat_str(b"")
        assert result == ""


class TestShortenStr:
    """Tests for shorten_str function"""

    def test_short_string_unchanged(self):
        s = "hello"
        assert shorten_str(s, 20) == s

    def test_exact_length_unchanged(self):
        s = "hello"
        assert shorten_str(s, 5) == s

    def test_long_string_shortened(self):
        s = "Hello there World"
        result = shorten_str(s, 15)
        assert " ... " in result
        assert len(result) <= 15

    def test_very_short_size_no_ellipsis(self):
        s = "Hello World"
        result = shorten_str(s, 5)
        assert result == "Hello"

    def test_size_6_no_ellipsis(self):
        s = "Hello World"
        result = shorten_str(s, 6)
        assert result == "Hello "

    def test_empty_string(self):
        assert shorten_str("", 10) == ""


class TestCompatibleEncodeMethod:
    """Tests for compatible_encode_method function"""

    def test_string_passthrough(self):
        result = compatible_encode_method("hello")
        assert result == "hello"

    def test_bytes_decode_utf8(self):
        result = compatible_encode_method(b"hello", "utf-8")
        assert result == "hello"

    def test_bytes_decode_latin1(self):
        result = compatible_encode_method(b"\xe9", "latin-1")
        assert result == "\xe9"

    def test_bytes_with_ignore_errors(self):
        invalid_utf8 = b"\xff\xfe"
        result = compatible_encode_method(invalid_utf8, "utf-8", "ignore")
        assert isinstance(result, str)

    def test_bytes_with_replace_errors(self):
        invalid_utf8 = b"\xff\xfe"
        result = compatible_encode_method(invalid_utf8, "utf-8", "replace")
        assert isinstance(result, str)


class TestMatrixMultiplication:
    """Tests for mult_matrix function"""

    def test_identity_matrix_multiplication(self):
        identity = MATRIX_IDENTITY
        m = (2, 0, 0, 2, 10, 20)
        result = mult_matrix(m, identity)
        assert result == m

    def test_identity_matrix_left(self):
        identity = MATRIX_IDENTITY
        m = (2, 0, 0, 2, 10, 20)
        result = mult_matrix(identity, m)
        assert result == m

    def test_scale_matrix_multiplication(self):
        scale_2x = (2, 0, 0, 2, 0, 0)
        result = mult_matrix(scale_2x, scale_2x)
        assert result == (4, 0, 0, 4, 0, 0)

    def test_translation_composition(self):
        t1 = (1, 0, 0, 1, 10, 0)
        t2 = (1, 0, 0, 1, 0, 20)
        result = mult_matrix(t1, t2)
        assert result == (1, 0, 0, 1, 10, 20)


class TestTranslateMatrix:
    """Tests for translate_matrix function"""

    def test_translate_identity_by_zero(self):
        result = translate_matrix(MATRIX_IDENTITY, (0, 0))
        assert result == MATRIX_IDENTITY

    def test_translate_identity_by_point(self):
        result = translate_matrix(MATRIX_IDENTITY, (10, 20))
        assert result == (1, 0, 0, 1, 10, 20)

    def test_translate_scaled_matrix(self):
        scale_2x = (2, 0, 0, 2, 0, 0)
        result = translate_matrix(scale_2x, (5, 10))
        assert result == (2, 0, 0, 2, 10, 20)

    def test_translate_existing_translation(self):
        m = (1, 0, 0, 1, 100, 200)
        result = translate_matrix(m, (10, 20))
        assert result == (1, 0, 0, 1, 110, 220)


class TestApplyMatrixPt:
    """Tests for apply_matrix_pt function"""

    def test_identity_matrix(self):
        result = apply_matrix_pt(MATRIX_IDENTITY, (10, 20))
        assert result == (10, 20)

    def test_translation(self):
        m = (1, 0, 0, 1, 100, 200)
        result = apply_matrix_pt(m, (10, 20))
        assert result == (110, 220)

    def test_scale(self):
        m = (2, 0, 0, 3, 0, 0)
        result = apply_matrix_pt(m, (10, 20))
        assert result == (20, 60)

    def test_origin_point(self):
        m = (1, 2, 3, 4, 5, 6)
        result = apply_matrix_pt(m, (0, 0))
        assert result == (5, 6)


class TestApplyMatrixNorm:
    """Tests for apply_matrix_norm function"""

    def test_identity_matrix(self):
        result = apply_matrix_norm(MATRIX_IDENTITY, (10, 20))
        assert result == (10, 20)

    def test_ignores_translation(self):
        m = (1, 0, 0, 1, 100, 200)
        result = apply_matrix_norm(m, (10, 20))
        assert result == (10, 20)

    def test_scale(self):
        m = (2, 0, 0, 3, 0, 0)
        result = apply_matrix_norm(m, (10, 20))
        assert result == (20, 60)

    def test_origin_vector(self):
        m = (2, 3, 4, 5, 100, 200)
        result = apply_matrix_norm(m, (0, 0))
        assert result == (0, 0)


class TestPlaneClass:
    """Tests for Plane spatial indexing class"""

    def test_empty_plane(self):
        plane = Plane((0, 0, 100, 100))
        assert len(plane) == 0
        assert list(plane) == []

    def test_add_single_object(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        assert len(plane) == 1
        assert obj in plane

    def test_add_multiple_objects(self):
        plane = Plane((0, 0, 100, 100))
        obj1 = LTComponent((10, 10, 20, 20))
        obj2 = LTComponent((50, 50, 60, 60))
        plane.add(obj1)
        plane.add(obj2)
        assert len(plane) == 2

    def test_extend_objects(self):
        plane = Plane((0, 0, 100, 100))
        objs = [LTComponent((i * 10, i * 10, i * 10 + 5, i * 10 + 5)) for i in range(5)]
        plane.extend(objs)
        assert len(plane) == 5

    def test_remove_object(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        plane.remove(obj)
        assert len(plane) == 0
        assert obj not in plane

    def test_find_overlapping_object(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        found = list(plane.find((15, 15, 25, 25)))
        assert obj in found

    def test_find_no_overlap(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        found = list(plane.find((50, 50, 60, 60)))
        assert found == []

    def test_find_outside_plane_bbox(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        found = list(plane.find((200, 200, 300, 300)))
        assert found == []

    def test_contains_after_remove(self):
        plane = Plane((0, 0, 100, 100))
        obj = LTComponent((10, 10, 20, 20))
        plane.add(obj)
        plane.remove(obj)
        assert obj not in plane

    def test_iteration_preserves_order(self):
        plane = Plane((0, 0, 100, 100))
        objs = [LTComponent((i * 10, i * 10, i * 10 + 5, i * 10 + 5)) for i in range(3)]
        for obj in objs:
            plane.add(obj)
        result = list(plane)
        assert result == objs

    def test_repr(self):
        plane = Plane((0, 0, 100, 100))
        repr_str = repr(plane)
        assert "Plane" in repr_str
        assert "objs=" in repr_str

    def test_custom_gridsize(self):
        plane = Plane((0, 0, 100, 100), gridsize=10)
        obj = LTComponent((5, 5, 15, 15))
        plane.add(obj)
        found = list(plane.find((0, 0, 20, 20)))
        assert obj in found


class TestEncFunction:
    """Tests for enc (HTML entity encoding) function"""

    def test_plain_text(self):
        result = enc("hello world")
        assert result == "hello world"

    def test_less_than(self):
        result = enc("<")
        assert result == "&lt;"

    def test_greater_than(self):
        result = enc(">")
        assert result == "&gt;"

    def test_ampersand(self):
        result = enc("&")
        assert result == "&amp;"

    def test_double_quote(self):
        result = enc('"')
        assert result == "&quot;"

    def test_empty_string(self):
        result = enc("")
        assert result == ""

    def test_bytes_returns_empty(self):
        result = enc(b"bytes input")
        assert result == ""


class TestBbox2str:
    """Tests for bbox2str function"""

    def test_integer_bbox(self):
        result = bbox2str((0, 0, 100, 200))
        assert result == "0.000,0.000,100.000,200.000"

    def test_float_bbox(self):
        result = bbox2str((1.5, 2.5, 3.5, 4.5))
        assert result == "1.500,2.500,3.500,4.500"

    def test_negative_bbox(self):
        result = bbox2str((-10.0, -20.0, 10.0, 20.0))
        assert result == "-10.000,-20.000,10.000,20.000"

    def test_precision(self):
        result = bbox2str((1.123456789, 2.123456789, 3.123456789, 4.123456789))
        assert result == "1.123,2.123,3.123,4.123"


class TestMatrix2str:
    """Tests for matrix2str function"""

    def test_identity_matrix(self):
        result = matrix2str(MATRIX_IDENTITY)
        assert result == "[1.00,0.00,0.00,1.00, (0.00,0.00)]"

    def test_scaled_matrix(self):
        result = matrix2str((2, 0, 0, 2, 10, 20))
        assert result == "[2.00,0.00,0.00,2.00, (10.00,20.00)]"

    def test_negative_values(self):
        result = matrix2str((-1, 0, 0, -1, -10, -20))
        assert "-1.00" in result


class TestFormatIntRoman:
    """Tests for format_int_roman function"""

    def test_one(self):
        assert format_int_roman(1) == "i"

    def test_four(self):
        assert format_int_roman(4) == "iv"

    def test_nine(self):
        assert format_int_roman(9) == "ix"

    def test_forty(self):
        assert format_int_roman(40) == "xl"

    def test_ninety(self):
        assert format_int_roman(90) == "xc"

    def test_four_hundred(self):
        assert format_int_roman(400) == "cd"

    def test_nine_hundred(self):
        assert format_int_roman(900) == "cm"

    def test_complex_number(self):
        assert format_int_roman(1999) == "mcmxcix"

    def test_max_value(self):
        assert format_int_roman(3999) == "mmmcmxcix"


class TestFormatIntAlpha:
    """Tests for format_int_alpha function"""

    def test_one_is_a(self):
        assert format_int_alpha(1) == "a"

    def test_twenty_six_is_z(self):
        assert format_int_alpha(26) == "z"

    def test_twenty_seven_is_aa(self):
        assert format_int_alpha(27) == "aa"

    def test_fifty_two_is_az(self):
        assert format_int_alpha(52) == "az"

    def test_large_number(self):
        result = format_int_alpha(26 * 27 + 1)
        assert result == "aaa"


class TestChoplist:
    """Tests for choplist function"""

    def test_exact_chunks(self):
        result = list(choplist(2, [1, 2, 3, 4]))
        assert result == [(1, 2), (3, 4)]

    def test_incomplete_chunk_dropped(self):
        result = list(choplist(2, [1, 2, 3, 4, 5]))
        assert result == [(1, 2), (3, 4)]

    def test_chunk_size_one(self):
        result = list(choplist(1, [1, 2, 3]))
        assert result == [(1,), (2,), (3,)]

    def test_empty_input(self):
        result = list(choplist(2, []))
        assert result == []

    def test_chunk_larger_than_input(self):
        result = list(choplist(10, [1, 2, 3]))
        assert result == []

    def test_generator_input(self):
        result = list(choplist(2, (x for x in range(6))))
        assert result == [(0, 1), (2, 3), (4, 5)]


class TestNunpack:
    """Tests for nunpack function"""

    def test_empty_bytes_returns_default(self):
        assert nunpack(b"") == 0
        assert nunpack(b"", 42) == 42

    def test_single_byte(self):
        assert nunpack(b"\x01") == 1
        assert nunpack(b"\xff") == 255

    def test_two_bytes(self):
        assert nunpack(b"\x01\x00") == 256
        assert nunpack(b"\xff\xff") == 65535

    def test_three_bytes(self):
        assert nunpack(b"\x01\x00\x00") == 65536

    def test_four_bytes(self):
        assert nunpack(b"\x00\x00\x00\x01") == 1
        assert nunpack(b"\x00\x00\x01\x00") == 256

    def test_big_endian(self):
        assert nunpack(b"\x12\x34") == 0x1234


class TestFsplit:
    """Tests for fsplit function"""

    def test_split_even_odd(self):
        truthy, falsy = fsplit(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
        assert truthy == [2, 4, 6]
        assert falsy == [1, 3, 5]

    def test_all_truthy(self):
        truthy, falsy = fsplit(lambda x: True, [1, 2, 3])
        assert truthy == [1, 2, 3]
        assert falsy == []

    def test_all_falsy(self):
        truthy, falsy = fsplit(lambda x: False, [1, 2, 3])
        assert truthy == []
        assert falsy == [1, 2, 3]

    def test_empty_input(self):
        truthy, falsy = fsplit(lambda x: True, [])
        assert truthy == []
        assert falsy == []

    def test_string_predicate(self):
        truthy, falsy = fsplit(lambda s: s.startswith("a"), ["apple", "banana", "apricot"])
        assert truthy == ["apple", "apricot"]
        assert falsy == ["banana"]


class TestUniq:
    """Tests for uniq function"""

    def test_removes_duplicates(self):
        result = list(uniq([1, 2, 2, 3, 3, 3, 4]))
        assert result == [1, 2, 3, 4]

    def test_preserves_order(self):
        result = list(uniq([3, 1, 2, 1, 3]))
        assert result == [3, 1, 2]

    def test_no_duplicates(self):
        result = list(uniq([1, 2, 3, 4]))
        assert result == [1, 2, 3, 4]

    def test_all_duplicates(self):
        result = list(uniq([1, 1, 1, 1]))
        assert result == [1]

    def test_empty_input(self):
        result = list(uniq([]))
        assert result == []

    def test_string_elements(self):
        result = list(uniq(["a", "b", "a", "c"]))
        assert result == ["a", "b", "c"]

    def test_generator_input(self):
        result = list(uniq(x for x in [1, 2, 1, 3]))
        assert result == [1, 2, 3]


class TestIsnumber:
    """Tests for isnumber function"""

    def test_int_is_number(self):
        assert isnumber(42) is True

    def test_float_is_number(self):
        assert isnumber(3.14) is True

    def test_negative_is_number(self):
        assert isnumber(-10) is True

    def test_zero_is_number(self):
        assert isnumber(0) is True

    def test_string_is_not_number(self):
        assert isnumber("42") is False

    def test_none_is_not_number(self):
        assert isnumber(None) is False

    def test_list_is_not_number(self):
        assert isnumber([1, 2, 3]) is False


class TestDrange:
    """Tests for drange function"""

    def test_simple_range(self):
        result = list(drange(0, 100, 10))
        assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_fractional_values(self):
        result = list(drange(0.5, 10.5, 5))
        assert 0 in result
        assert 2 in result

    def test_single_cell(self):
        result = list(drange(0, 10, 100))
        assert result == [0]


class TestGetBound:
    """Tests for get_bound function"""

    def test_single_point(self):
        result = get_bound([(5, 10)])
        assert result == (5, 10, 5, 10)

    def test_multiple_points(self):
        result = get_bound([(0, 0), (10, 20), (5, 15)])
        assert result == (0, 0, 10, 20)

    def test_negative_points(self):
        result = get_bound([(-5, -10), (5, 10)])
        assert result == (-5, -10, 5, 10)

    def test_empty_returns_inf(self):
        result = get_bound([])
        assert result == (INF, INF, -INF, -INF)


class TestPick:
    """Tests for pick function"""

    def test_pick_max(self):
        result = pick([1, 5, 3, 2], lambda x: x)
        assert result == 5

    def test_pick_min_negated(self):
        result = pick([1, 5, 3, 2], lambda x: -x)
        assert result == 1

    def test_pick_empty_returns_default(self):
        result = pick([], lambda x: x, maxobj=42)
        assert result == 42

    def test_pick_with_objects(self):
        items = [{"value": 1}, {"value": 5}, {"value": 3}]
        result = pick(items, lambda x: x["value"])
        assert result == {"value": 5}


class TestParseRect:
    """Tests for parse_rect function"""

    def test_valid_rect(self):
        result = parse_rect([0, 0, 100, 200])
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_tuple_input(self):
        result = parse_rect((10, 20, 30, 40))
        assert result == (10.0, 20.0, 30.0, 40.0)

    def test_string_numbers(self):
        result = parse_rect(["1", "2", "3", "4"])
        assert result == (1.0, 2.0, 3.0, 4.0)

    def test_invalid_raises(self):
        with pytest.raises(PDFValueError):
            parse_rect([1, 2, 3])


class TestDecodeText:
    """Tests for decode_text function"""

    def test_simple_ascii(self):
        result = decode_text(b"hello")
        assert "h" in result
        assert "e" in result

    def test_utf16be_with_bom(self):
        result = decode_text(b"\xfe\xff\x00H\x00e\x00l\x00l\x00o")
        assert result == "Hello"

    def test_empty_bytes(self):
        result = decode_text(b"")
        assert result == ""


class TestPaethPredictor:
    """Tests for paeth_predictor function"""

    def test_all_zeros(self):
        result = paeth_predictor(0, 0, 0)
        assert result == 0

    def test_left_nearest(self):
        result = paeth_predictor(10, 5, 0)
        assert result == 10

    def test_above_nearest(self):
        result = paeth_predictor(5, 10, 0)
        assert result == 10

    def test_upper_left_nearest(self):
        result = paeth_predictor(10, 10, 10)
        assert result == 10


class TestApplyTiffPredictor:
    """Tests for apply_tiff_predictor function"""

    def test_unsupported_bits_raises(self):
        with pytest.raises(PDFValueError):
            apply_tiff_predictor(1, 1, 16, b"\x00\x00")

    def test_simple_data(self):
        result = apply_tiff_predictor(1, 2, 8, b"\x01\x02")
        assert len(result) == 2


class TestApplyPngPredictor:
    """Tests for apply_png_predictor function"""

    def test_unsupported_bits_raises(self):
        with pytest.raises(PDFValueError):
            apply_png_predictor(10, 1, 1, 16, b"\x00\x00")

    def test_unsupported_filter_type_raises(self):
        with pytest.raises(PDFValueError):
            apply_png_predictor(10, 1, 1, 8, b"\x99\x00")

    def test_filter_type_none(self):
        result = apply_png_predictor(10, 1, 2, 8, b"\x00\x01\x02")
        assert result == b"\x01\x02"


class TestUnpadAes:
    """Tests for unpad_aes function"""

    def test_empty_input(self):
        result = unpad_aes(b"")
        assert result == b""

    def test_valid_padding(self):
        padded = b"hello\x03\x03\x03"
        result = unpad_aes(padded)
        assert result == b"hello"

    def test_invalid_padding_value(self):
        invalid = b"hello\x20"
        result = unpad_aes(invalid)
        assert result == invalid

    def test_padding_too_large(self):
        invalid = b"\x11"
        result = unpad_aes(invalid)
        assert result == invalid

    def test_full_block_padding(self):
        padded = b"helloworld123456" + bytes([16] * 16)
        result = unpad_aes(padded)
        assert result == b"helloworld123456"


class TestApplyMatrixRect:
    """Tests for apply_matrix_rect function"""

    def test_identity_preserves_rect(self):
        rect = (10, 20, 30, 40)
        result = apply_matrix_rect(MATRIX_IDENTITY, rect)
        assert result == rect

    def test_translation(self):
        m = (1, 0, 0, 1, 100, 200)
        rect = (10, 20, 30, 40)
        result = apply_matrix_rect(m, rect)
        assert result == (110, 220, 130, 240)

    def test_scale(self):
        m = (2, 0, 0, 2, 0, 0)
        rect = (10, 20, 30, 40)
        result = apply_matrix_rect(m, rect)
        assert result == (20, 40, 60, 80)


class TestINFConstant:
    """Tests for INF constant"""

    def test_inf_is_32bit_max(self):
        assert INF == (1 << 31) - 1
        assert INF == 2147483647


class TestMatrixIdentity:
    """Tests for MATRIX_IDENTITY constant"""

    def test_identity_values(self):
        assert MATRIX_IDENTITY == (1, 0, 0, 1, 0, 0)
