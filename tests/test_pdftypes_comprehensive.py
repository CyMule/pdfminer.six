"""Comprehensive tests for pdfminer/pdftypes.py"""

from __future__ import annotations

import warnings
import zlib
from typing import Any

import pytest

from pdfminer import settings
from pdfminer.pdfexceptions import (
    PDFException,
    PDFKeyError,
    PDFNotImplementedError,
    PDFObjectNotFound,
    PDFTypeError,
    PDFValueError,
)
from pdfminer.pdftypes import (
    LITERAL_CRYPT,
    LITERALS_ASCII85_DECODE,
    LITERALS_ASCIIHEX_DECODE,
    LITERALS_CCITTFAX_DECODE,
    LITERALS_DCT_DECODE,
    LITERALS_FLATE_DECODE,
    LITERALS_JBIG2_DECODE,
    LITERALS_JPX_DECODE,
    LITERALS_LZW_DECODE,
    LITERALS_RUNLENGTH_DECODE,
    PDFObject,
    PDFObjRef,
    PDFStream,
    decipher_all,
    decompress_corrupted,
    dict_value,
    float_value,
    int_value,
    list_value,
    num_value,
    resolve1,
    resolve_all,
    str_value,
    stream_value,
    uint_value,
)
from pdfminer.psparser import LIT


class MockDocument:
    """Mock PDFDocument for testing.

    This class mimics PDFDocument interface for test purposes.
    Using Any type to avoid complex type stubs for tests.
    """

    def __init__(self, objects: dict[int, Any] | None = None) -> None:
        self.objects = objects or {}
        self.decipher = None

    def getobj(self, objid: int) -> Any:
        if objid in self.objects:
            return self.objects[objid]
        raise PDFObjectNotFound(f"Object {objid} not found")


class MockDocumentWithDecipher(MockDocument):
    """Mock PDFDocument with a decipher function."""

    def __init__(
        self,
        objects: dict[int, Any] | None = None,
        decipher_func: Any = None,
    ) -> None:
        super().__init__(objects)
        self.decipher = decipher_func


def mock_doc(objects: dict[int, Any] | None = None) -> Any:
    """Create a mock document. Returns Any to satisfy type checker in tests."""
    return MockDocument(objects)


class TestLiteralConstants:
    """Tests for the literal constants defined in pdftypes.py"""

    def test_literal_crypt(self) -> None:
        assert LIT("Crypt") == LITERAL_CRYPT

    def test_literals_flate_decode(self) -> None:
        assert LIT("FlateDecode") in LITERALS_FLATE_DECODE
        assert LIT("Fl") in LITERALS_FLATE_DECODE
        assert len(LITERALS_FLATE_DECODE) == 2

    def test_literals_lzw_decode(self) -> None:
        assert LIT("LZWDecode") in LITERALS_LZW_DECODE
        assert LIT("LZW") in LITERALS_LZW_DECODE
        assert len(LITERALS_LZW_DECODE) == 2

    def test_literals_ascii85_decode(self) -> None:
        assert LIT("ASCII85Decode") in LITERALS_ASCII85_DECODE
        assert LIT("A85") in LITERALS_ASCII85_DECODE
        assert len(LITERALS_ASCII85_DECODE) == 2

    def test_literals_asciihex_decode(self) -> None:
        assert LIT("ASCIIHexDecode") in LITERALS_ASCIIHEX_DECODE
        assert LIT("AHx") in LITERALS_ASCIIHEX_DECODE
        assert len(LITERALS_ASCIIHEX_DECODE) == 2

    def test_literals_runlength_decode(self) -> None:
        assert LIT("RunLengthDecode") in LITERALS_RUNLENGTH_DECODE
        assert LIT("RL") in LITERALS_RUNLENGTH_DECODE
        assert len(LITERALS_RUNLENGTH_DECODE) == 2

    def test_literals_ccittfax_decode(self) -> None:
        assert LIT("CCITTFaxDecode") in LITERALS_CCITTFAX_DECODE
        assert LIT("CCF") in LITERALS_CCITTFAX_DECODE
        assert len(LITERALS_CCITTFAX_DECODE) == 2

    def test_literals_dct_decode(self) -> None:
        assert LIT("DCTDecode") in LITERALS_DCT_DECODE
        assert LIT("DCT") in LITERALS_DCT_DECODE
        assert len(LITERALS_DCT_DECODE) == 2

    def test_literals_jbig2_decode(self) -> None:
        assert LIT("JBIG2Decode") in LITERALS_JBIG2_DECODE
        assert len(LITERALS_JBIG2_DECODE) == 1

    def test_literals_jpx_decode(self) -> None:
        assert LIT("JPXDecode") in LITERALS_JPX_DECODE
        assert len(LITERALS_JPX_DECODE) == 1


class TestPDFObject:
    """Tests for PDFObject base class"""

    def test_pdfobject_inheritance(self) -> None:
        from pdfminer.psparser import PSObject

        assert issubclass(PDFObject, PSObject)

    def test_pdfobject_instantiation(self) -> None:
        obj = PDFObject()
        assert obj is not None


class TestPDFObjRef:
    """Tests for PDFObjRef class"""

    def test_pdfobjref_init(self) -> None:
        doc = mock_doc()
        ref = PDFObjRef(doc, 42)
        assert ref.doc is doc
        assert ref.objid == 42

    def test_pdfobjref_with_none_doc(self) -> None:
        ref = PDFObjRef(None, 42)
        assert ref.doc is None
        assert ref.objid == 42

    def test_pdfobjref_repr(self) -> None:
        doc = mock_doc()
        ref = PDFObjRef(doc, 42)
        assert repr(ref) == "<PDFObjRef:42>"

    def test_pdfobjref_repr_different_objids(self) -> None:
        doc = mock_doc()
        ref1 = PDFObjRef(doc, 1)
        ref2 = PDFObjRef(doc, 999)
        assert repr(ref1) == "<PDFObjRef:1>"
        assert repr(ref2) == "<PDFObjRef:999>"

    def test_pdfobjref_resolve_existing_object(self) -> None:
        doc = mock_doc({42: "test_value"})
        ref = PDFObjRef(doc, 42)
        assert ref.resolve() == "test_value"

    def test_pdfobjref_resolve_missing_object_returns_default(self) -> None:
        doc = mock_doc()
        ref = PDFObjRef(doc, 999)
        assert ref.resolve() is None
        assert ref.resolve(default="fallback") == "fallback"

    def test_pdfobjref_resolve_nested_reference(self) -> None:
        doc = mock_doc()
        inner_ref = PDFObjRef(doc, 2)
        doc.objects = {1: inner_ref, 2: "final_value"}
        outer_ref = PDFObjRef(doc, 1)
        assert outer_ref.resolve() is inner_ref

    def test_pdfobjref_zero_objid_strict_mode(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            doc = mock_doc()
            with pytest.raises(PDFValueError, match="cannot be 0"):
                PDFObjRef(doc, 0)
        finally:
            settings.STRICT = original_strict

    def test_pdfobjref_zero_objid_non_strict_mode(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            doc = mock_doc()
            ref = PDFObjRef(doc, 0)
            assert ref.objid == 0
        finally:
            settings.STRICT = original_strict

    def test_pdfobjref_deprecated_third_argument(self) -> None:
        doc = mock_doc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ref = PDFObjRef(doc, 42, "unused_argument")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "third argument" in str(w[0].message).lower()
            assert ref.objid == 42

    def test_pdfobjref_inheritance(self) -> None:
        assert issubclass(PDFObjRef, PDFObject)


class TestResolve1:
    """Tests for resolve1() function"""

    def test_resolve1_non_reference(self) -> None:
        assert resolve1(42) == 42
        assert resolve1("test") == "test"
        assert resolve1([1, 2, 3]) == [1, 2, 3]
        assert resolve1({"key": "value"}) == {"key": "value"}
        assert resolve1(None) is None

    def test_resolve1_single_reference(self) -> None:
        doc = mock_doc({1: "resolved_value"})
        ref = PDFObjRef(doc, 1)
        assert resolve1(ref) == "resolved_value"

    def test_resolve1_chained_references(self) -> None:
        doc = mock_doc()
        ref2 = PDFObjRef(doc, 2)
        doc.objects = {1: ref2, 2: "final_value"}
        ref1 = PDFObjRef(doc, 1)
        assert resolve1(ref1) == "final_value"

    def test_resolve1_missing_object_returns_default(self) -> None:
        doc = mock_doc()
        ref = PDFObjRef(doc, 999)
        assert resolve1(ref) is None
        assert resolve1(ref, default="custom_default") == "custom_default"

    def test_resolve1_does_not_resolve_nested_refs_in_containers(self) -> None:
        doc = mock_doc({2: "value"})
        inner_ref = PDFObjRef(doc, 2)
        doc.objects[1] = [inner_ref]
        outer_ref = PDFObjRef(doc, 1)
        result = resolve1(outer_ref)
        assert isinstance(result, list)
        assert isinstance(result[0], PDFObjRef)


class TestResolveAll:
    """Tests for resolve_all() function"""

    def test_resolve_all_non_reference(self) -> None:
        assert resolve_all(42) == 42
        assert resolve_all("test") == "test"
        assert resolve_all(None) is None

    def test_resolve_all_single_reference(self) -> None:
        doc = mock_doc({1: "resolved_value"})
        ref = PDFObjRef(doc, 1)
        assert resolve_all(ref) == "resolved_value"

    def test_resolve_all_nested_list(self) -> None:
        doc = mock_doc({1: "value1", 2: "value2"})
        ref1 = PDFObjRef(doc, 1)
        ref2 = PDFObjRef(doc, 2)
        result = resolve_all([ref1, ref2, "direct"])
        assert result == ["value1", "value2", "direct"]

    def test_resolve_all_nested_dict(self) -> None:
        doc = mock_doc({1: "value1", 2: "value2"})
        ref1 = PDFObjRef(doc, 1)
        ref2 = PDFObjRef(doc, 2)
        result = resolve_all({"a": ref1, "b": ref2, "c": "direct"})
        assert result == {"a": "value1", "b": "value2", "c": "direct"}

    def test_resolve_all_deeply_nested(self) -> None:
        doc = mock_doc({1: "inner_value"})
        ref = PDFObjRef(doc, 1)
        nested = {"level1": [{"level2": ref}]}
        result = resolve_all(nested)
        assert result == {"level1": [{"level2": "inner_value"}]}

    def test_resolve_all_chained_references(self) -> None:
        doc = mock_doc()
        ref2 = PDFObjRef(doc, 2)
        doc.objects = {1: ref2, 2: "final_value"}
        ref1 = PDFObjRef(doc, 1)
        assert resolve_all(ref1) == "final_value"

    def test_resolve_all_missing_object_returns_default(self) -> None:
        doc = mock_doc()
        ref = PDFObjRef(doc, 999)
        assert resolve_all(ref) is None
        assert resolve_all(ref, default="fallback") == "fallback"

    def test_resolve_all_list_with_missing_refs(self) -> None:
        doc = mock_doc({1: "exists"})
        ref1 = PDFObjRef(doc, 1)
        ref2 = PDFObjRef(doc, 999)
        result = resolve_all([ref1, ref2], default="missing")
        assert result == ["exists", "missing"]


class TestDecipherAll:
    """Tests for decipher_all() function"""

    def test_decipher_all_bytes(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data[::-1]

        result = decipher_all(mock_decipher, 1, 0, b"hello")
        assert result == b"olleh"

    def test_decipher_all_empty_bytes(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return b"should_not_be_called"

        result = decipher_all(mock_decipher, 1, 0, b"")
        assert result == b""

    def test_decipher_all_list(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data.upper()

        result = decipher_all(mock_decipher, 1, 0, [b"hello", b"world"])
        assert result == [b"HELLO", b"WORLD"]

    def test_decipher_all_dict(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data.upper()

        result = decipher_all(mock_decipher, 1, 0, {"a": b"test", "b": b"data"})
        assert result == {"a": b"TEST", "b": b"DATA"}

    def test_decipher_all_nested(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data.upper()

        nested = {"list": [b"hello", {"nested": b"world"}]}
        result = decipher_all(mock_decipher, 1, 0, nested)
        assert result == {"list": [b"HELLO", {"nested": b"WORLD"}]}

    def test_decipher_all_non_bytes_passthrough(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            raise AssertionError("Should not be called for non-bytes")

        assert decipher_all(mock_decipher, 1, 0, 42) == 42
        assert decipher_all(mock_decipher, 1, 0, "string") == "string"
        assert decipher_all(mock_decipher, 1, 0, None) is None


class TestIntValue:
    """Tests for int_value() function"""

    def test_int_value_integer(self) -> None:
        assert int_value(42) == 42
        assert int_value(-10) == -10
        assert int_value(0) == 0

    def test_int_value_reference(self) -> None:
        doc = mock_doc({1: 42})
        ref = PDFObjRef(doc, 1)
        assert int_value(ref) == 42

    def test_int_value_non_integer_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="Integer required"):
                int_value("not_an_int")
            with pytest.raises(PDFTypeError, match="Integer required"):
                int_value(3.14)
        finally:
            settings.STRICT = original_strict

    def test_int_value_non_integer_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert int_value("not_an_int") == 0
            assert int_value(3.14) == 0
            assert int_value(None) == 0
        finally:
            settings.STRICT = original_strict


class TestFloatValue:
    """Tests for float_value() function"""

    def test_float_value_float(self) -> None:
        assert float_value(3.14) == 3.14
        assert float_value(-2.5) == -2.5
        assert float_value(0.0) == 0.0

    def test_float_value_reference(self) -> None:
        doc = mock_doc({1: 3.14})
        ref = PDFObjRef(doc, 1)
        assert float_value(ref) == 3.14

    def test_float_value_integer_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="Float required"):
                float_value(42)
        finally:
            settings.STRICT = original_strict

    def test_float_value_non_float_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert float_value(42) == 0.0
            assert float_value("not_a_float") == 0.0
            assert float_value(None) == 0.0
        finally:
            settings.STRICT = original_strict


class TestNumValue:
    """Tests for num_value() function"""

    def test_num_value_integer(self) -> None:
        assert num_value(42) == 42
        assert num_value(-10) == -10

    def test_num_value_float(self) -> None:
        assert num_value(3.14) == 3.14
        assert num_value(-2.5) == -2.5

    def test_num_value_reference(self) -> None:
        doc = mock_doc({1: 42, 2: 3.14})
        ref_int = PDFObjRef(doc, 1)
        ref_float = PDFObjRef(doc, 2)
        assert num_value(ref_int) == 42
        assert num_value(ref_float) == 3.14

    def test_num_value_non_number_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="Int or Float required"):
                num_value("not_a_number")
        finally:
            settings.STRICT = original_strict

    def test_num_value_non_number_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert num_value("not_a_number") == 0
            assert num_value(None) == 0
        finally:
            settings.STRICT = original_strict


class TestUintValue:
    """Tests for uint_value() function"""

    def test_uint_value_positive(self) -> None:
        assert uint_value(42, 8) == 42
        assert uint_value(255, 8) == 255

    def test_uint_value_zero(self) -> None:
        assert uint_value(0, 8) == 256

    def test_uint_value_negative(self) -> None:
        assert uint_value(-1, 8) == 255
        assert uint_value(-128, 8) == 128

    def test_uint_value_different_bit_widths(self) -> None:
        assert uint_value(-1, 16) == 65535
        assert uint_value(-1, 32) == 4294967295

    def test_uint_value_reference(self) -> None:
        doc = mock_doc({1: 42})
        ref = PDFObjRef(doc, 1)
        assert uint_value(ref, 8) == 42


class TestStrValue:
    """Tests for str_value() function"""

    def test_str_value_bytes(self) -> None:
        assert str_value(b"hello") == b"hello"
        assert str_value(b"") == b""

    def test_str_value_reference(self) -> None:
        doc = mock_doc({1: b"test_string"})
        ref = PDFObjRef(doc, 1)
        assert str_value(ref) == b"test_string"

    def test_str_value_non_bytes_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="String required"):
                str_value("not_bytes")
            with pytest.raises(PDFTypeError, match="String required"):
                str_value(42)
        finally:
            settings.STRICT = original_strict

    def test_str_value_non_bytes_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert str_value("not_bytes") == b""
            assert str_value(42) == b""
            assert str_value(None) == b""
        finally:
            settings.STRICT = original_strict


class TestListValue:
    """Tests for list_value() function"""

    def test_list_value_list(self) -> None:
        assert list_value([1, 2, 3]) == [1, 2, 3]
        assert list_value([]) == []

    def test_list_value_tuple(self) -> None:
        assert list_value((1, 2, 3)) == (1, 2, 3)
        assert list_value(()) == ()

    def test_list_value_reference(self) -> None:
        doc = mock_doc({1: [1, 2, 3]})
        ref = PDFObjRef(doc, 1)
        assert list_value(ref) == [1, 2, 3]

    def test_list_value_non_list_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="List required"):
                list_value("not_a_list")
            with pytest.raises(PDFTypeError, match="List required"):
                list_value(42)
        finally:
            settings.STRICT = original_strict

    def test_list_value_non_list_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert list_value("not_a_list") == []
            assert list_value(42) == []
            assert list_value(None) == []
        finally:
            settings.STRICT = original_strict


class TestDictValue:
    """Tests for dict_value() function"""

    def test_dict_value_dict(self) -> None:
        assert dict_value({"key": "value"}) == {"key": "value"}
        assert dict_value({}) == {}

    def test_dict_value_reference(self) -> None:
        doc = mock_doc({1: {"key": "value"}})
        ref = PDFObjRef(doc, 1)
        assert dict_value(ref) == {"key": "value"}

    def test_dict_value_non_dict_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="Dict required"):
                dict_value("not_a_dict")
            with pytest.raises(PDFTypeError, match="Dict required"):
                dict_value([1, 2, 3])
        finally:
            settings.STRICT = original_strict

    def test_dict_value_non_dict_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            assert dict_value("not_a_dict") == {}
            assert dict_value([1, 2, 3]) == {}
            assert dict_value(None) == {}
        finally:
            settings.STRICT = original_strict


class TestStreamValue:
    """Tests for stream_value() function"""

    def test_stream_value_pdfstream(self) -> None:
        stream = PDFStream({"Length": 5}, b"hello")
        result = stream_value(stream)
        assert result is stream

    def test_stream_value_reference(self) -> None:
        stream = PDFStream({"Length": 5}, b"hello")
        doc = mock_doc({1: stream})
        ref = PDFObjRef(doc, 1)
        assert stream_value(ref) is stream

    def test_stream_value_non_stream_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError, match="PDFStream required"):
                stream_value("not_a_stream")
            with pytest.raises(PDFTypeError, match="PDFStream required"):
                stream_value(42)
        finally:
            settings.STRICT = original_strict

    def test_stream_value_non_stream_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            result = stream_value("not_a_stream")
            assert isinstance(result, PDFStream)
            assert result.attrs == {}
            assert result.rawdata == b""
        finally:
            settings.STRICT = original_strict


class TestDecompressCorrupted:
    """Tests for decompress_corrupted() function"""

    def test_decompress_corrupted_valid_data(self) -> None:
        original = b"hello world"
        compressed = zlib.compress(original)
        result = decompress_corrupted(compressed)
        assert result == original

    def test_decompress_corrupted_with_bad_checksum(self) -> None:
        original = b"test data for decompression"
        compressed = zlib.compress(original)
        corrupted = compressed[:-2] + b"\x00\x00"
        result = decompress_corrupted(corrupted)
        assert result == original

    def test_decompress_corrupted_early_error(self) -> None:
        bad_data = b"\x78\x9c\xff\xff\xff\xff"
        with pytest.raises(zlib.error):
            decompress_corrupted(bad_data)


class TestPDFStream:
    """Tests for PDFStream class"""

    def test_pdfstream_init(self) -> None:
        attrs = {"Length": 5, "Type": LIT("XObject")}
        rawdata = b"hello"
        stream = PDFStream(attrs, rawdata)
        assert stream.attrs == attrs
        assert stream.rawdata == rawdata
        assert stream.data is None
        assert stream.decipher is None
        assert stream.objid is None
        assert stream.genno is None

    def test_pdfstream_init_with_decipher(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data

        stream = PDFStream({}, b"test", decipher=mock_decipher)
        assert stream.decipher is mock_decipher

    def test_pdfstream_set_objid(self) -> None:
        stream = PDFStream({}, b"test")
        stream.set_objid(42, 0)
        assert stream.objid == 42
        assert stream.genno == 0

    def test_pdfstream_repr_raw(self) -> None:
        stream = PDFStream({"Length": 5}, b"hello")
        stream.set_objid(42, 0)
        repr_str = repr(stream)
        assert "42" in repr_str
        assert "raw=5" in repr_str

    def test_pdfstream_repr_decoded(self) -> None:
        stream = PDFStream({}, b"hello")
        stream.set_objid(42, 0)
        stream.data = b"hello"
        stream.rawdata = None
        repr_str = repr(stream)
        assert "42" in repr_str
        assert "len=5" in repr_str

    def test_pdfstream_contains(self) -> None:
        stream = PDFStream({"Length": 5, "Type": LIT("XObject")}, b"hello")
        assert "Length" in stream
        assert "Type" in stream
        assert "NonExistent" not in stream

    def test_pdfstream_getitem(self) -> None:
        stream = PDFStream({"Length": 5, "Type": LIT("XObject")}, b"hello")
        stream.set_objid(42, 0)
        assert stream["Length"] == 5
        assert stream["Type"] == LIT("XObject")

    def test_pdfstream_getitem_missing_key(self) -> None:
        stream = PDFStream({}, b"hello")
        stream.set_objid(42, 0)
        with pytest.raises(PDFKeyError, match="does not have attribute"):
            _ = stream["NonExistent"]

    def test_pdfstream_get(self) -> None:
        stream = PDFStream({"Length": 5}, b"hello")
        assert stream.get("Length") == 5
        assert stream.get("NonExistent") is None
        assert stream.get("NonExistent", "default") == "default"

    def test_pdfstream_get_any(self) -> None:
        stream = PDFStream({"Filter": LIT("FlateDecode")}, b"data")
        result = stream.get_any(("F", "Filter"), None)
        assert result == LIT("FlateDecode")

    def test_pdfstream_get_any_first_match(self) -> None:
        stream = PDFStream({"F": LIT("LZW"), "Filter": LIT("FlateDecode")}, b"data")
        result = stream.get_any(("F", "Filter"), None)
        assert result == LIT("LZW")

    def test_pdfstream_get_any_no_match(self) -> None:
        stream = PDFStream({}, b"data")
        result = stream.get_any(("F", "Filter"), "default")
        assert result == "default"

    def test_pdfstream_get_filters_no_filter(self) -> None:
        stream = PDFStream({}, b"data")
        assert stream.get_filters() == []

    def test_pdfstream_get_filters_single_filter(self) -> None:
        stream = PDFStream({"Filter": LIT("FlateDecode")}, b"data")
        filters = stream.get_filters()
        assert len(filters) == 1
        assert filters[0][0] == LIT("FlateDecode")

    def test_pdfstream_get_filters_multiple_filters(self) -> None:
        stream = PDFStream(
            {"Filter": [LIT("ASCII85Decode"), LIT("FlateDecode")]},
            b"data",
        )
        filters = stream.get_filters()
        assert len(filters) == 2
        assert filters[0][0] == LIT("ASCII85Decode")
        assert filters[1][0] == LIT("FlateDecode")

    def test_pdfstream_get_filters_with_params(self) -> None:
        stream = PDFStream(
            {
                "Filter": LIT("FlateDecode"),
                "DecodeParms": {"Predictor": 12, "Columns": 100},
            },
            b"data",
        )
        filters = stream.get_filters()
        assert len(filters) == 1
        assert filters[0][0] == LIT("FlateDecode")
        assert filters[0][1]["Predictor"] == 12

    def test_pdfstream_get_filters_multiple_params(self) -> None:
        stream = PDFStream(
            {
                "Filter": [LIT("ASCII85Decode"), LIT("FlateDecode")],
                "DecodeParms": [{}, {"Predictor": 12}],
            },
            b"data",
        )
        filters = stream.get_filters()
        assert len(filters) == 2
        assert filters[0][1] == {}
        assert filters[1][1]["Predictor"] == 12

    def test_pdfstream_get_filters_param_length_mismatch_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            stream = PDFStream(
                {
                    "Filter": [LIT("ASCII85Decode"), LIT("FlateDecode")],
                    "DecodeParms": [{"Predictor": 12}],
                },
                b"data",
            )
            with pytest.raises(PDFException, match="Parameters len filter mismatch"):
                stream.get_filters()
        finally:
            settings.STRICT = original_strict

    def test_pdfstream_decode_no_filter(self) -> None:
        stream = PDFStream({}, b"hello")
        stream.decode()
        assert stream.data == b"hello"
        assert stream.rawdata is None

    def test_pdfstream_decode_flate(self) -> None:
        original = b"hello world test data"
        compressed = zlib.compress(original)
        stream = PDFStream({"Filter": LIT("FlateDecode")}, compressed)
        stream.decode()
        assert stream.data == original

    def test_pdfstream_decode_flate_abbreviated(self) -> None:
        original = b"test data"
        compressed = zlib.compress(original)
        stream = PDFStream({"Filter": LIT("Fl")}, compressed)
        stream.decode()
        assert stream.data == original

    def test_pdfstream_decode_ascii85(self) -> None:
        stream = PDFStream({"Filter": LIT("ASCII85Decode")}, b"<~BOu!rDZ~>")
        stream.decode()
        assert stream.data == b"hello"

    def test_pdfstream_decode_asciihex(self) -> None:
        stream = PDFStream({"Filter": LIT("ASCIIHexDecode")}, b"48656C6C6F>")
        stream.decode()
        assert stream.data == b"Hello"

    def test_pdfstream_decode_runlength(self) -> None:
        encoded = bytes([0, ord("A"), 128])
        stream = PDFStream({"Filter": LIT("RunLengthDecode")}, encoded)
        stream.decode()
        assert stream.data == b"A"

    def test_pdfstream_decode_dct_passthrough(self) -> None:
        jpg_marker = b"\xff\xd8\xff"
        stream = PDFStream({"Filter": LIT("DCTDecode")}, jpg_marker)
        stream.decode()
        assert stream.data == jpg_marker

    def test_pdfstream_decode_jpx_passthrough(self) -> None:
        jpx_data = b"\x00\x00\x00\x0cjP"
        stream = PDFStream({"Filter": LIT("JPXDecode")}, jpx_data)
        stream.decode()
        assert stream.data == jpx_data

    def test_pdfstream_decode_jbig2_passthrough(self) -> None:
        jbig2_data = b"\x97\x4a\x42\x32"
        stream = PDFStream({"Filter": LIT("JBIG2Decode")}, jbig2_data)
        stream.decode()
        assert stream.data == jbig2_data

    def test_pdfstream_decode_crypt_not_implemented(self) -> None:
        stream = PDFStream({"Filter": LIT("Crypt")}, b"data")
        with pytest.raises(PDFNotImplementedError, match="/Crypt filter"):
            stream.decode()

    def test_pdfstream_decode_unsupported_filter(self) -> None:
        stream = PDFStream({"Filter": LIT("UnknownFilter")}, b"data")
        with pytest.raises(PDFNotImplementedError, match="Unsupported filter"):
            stream.decode()

    def test_pdfstream_decode_invalid_zlib_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            stream = PDFStream({"Filter": LIT("FlateDecode")}, b"invalid_zlib_data")
            with pytest.raises(PDFException, match="Invalid zlib bytes"):
                stream.decode()
        finally:
            settings.STRICT = original_strict

    def test_pdfstream_decode_invalid_zlib_non_strict(self) -> None:
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            stream = PDFStream({"Filter": LIT("FlateDecode")}, b"invalid")
            stream.decode()
            assert stream.data == b""
        finally:
            settings.STRICT = original_strict

    def test_pdfstream_decode_with_decipher(self) -> None:
        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data[::-1]

        original = b"hello"
        stream = PDFStream({}, original[::-1], decipher=mock_decipher)
        stream.set_objid(1, 0)
        stream.decode()
        assert stream.data == original

    def test_pdfstream_decode_predictor_1_no_change(self) -> None:
        original = b"test data"
        compressed = zlib.compress(original)
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "DecodeParms": {"Predictor": 1}},
            compressed,
        )
        stream.decode()
        assert stream.data == original

    def test_pdfstream_decode_unsupported_predictor(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "DecodeParms": {"Predictor": 5}},
            compressed,
        )
        with pytest.raises(PDFNotImplementedError, match="Unsupported predictor"):
            stream.decode()

    def test_pdfstream_get_data(self) -> None:
        stream = PDFStream({}, b"hello")
        assert stream.data is None
        data = stream.get_data()
        assert data == b"hello"
        assert stream.data == b"hello"

    def test_pdfstream_get_data_cached(self) -> None:
        stream = PDFStream({}, b"hello")
        data1 = stream.get_data()
        data2 = stream.get_data()
        assert data1 is data2

    def test_pdfstream_get_rawdata(self) -> None:
        stream = PDFStream({}, b"hello")
        assert stream.get_rawdata() == b"hello"
        stream.decode()
        assert stream.get_rawdata() is None

    def test_pdfstream_multiple_filters(self) -> None:
        original = b"hello world"
        compressed = zlib.compress(original)
        from base64 import a85encode

        ascii85_encoded = b"<~" + a85encode(compressed) + b"~>"
        stream = PDFStream(
            {"Filter": [LIT("ASCII85Decode"), LIT("FlateDecode")]},
            ascii85_encoded,
        )
        stream.decode()
        assert stream.data == original

    def test_pdfstream_inheritance(self) -> None:
        assert issubclass(PDFStream, PDFObject)


class TestPDFStreamPredictor:
    """Tests for PDFStream predictor handling"""

    def test_png_predictor_10(self) -> None:
        columns = 4
        colors = 1
        bpc = 8
        row1 = bytes([0]) + bytes([10, 20, 30, 40])
        row2 = bytes([0]) + bytes([50, 60, 70, 80])
        raw_with_predictor = row1 + row2
        compressed = zlib.compress(raw_with_predictor)
        stream = PDFStream(
            {
                "Filter": LIT("FlateDecode"),
                "DecodeParms": {
                    "Predictor": 10,
                    "Colors": colors,
                    "Columns": columns,
                    "BitsPerComponent": bpc,
                },
            },
            compressed,
        )
        stream.decode()
        expected = bytes([10, 20, 30, 40, 50, 60, 70, 80])
        assert stream.data == expected

    def test_tiff_predictor_2(self) -> None:
        columns = 4
        colors = 1
        bpc = 8
        row1 = bytes([10, 10, 10, 10])
        row2 = bytes([20, 20, 20, 20])
        raw_with_predictor = row1 + row2
        compressed = zlib.compress(raw_with_predictor)
        stream = PDFStream(
            {
                "Filter": LIT("FlateDecode"),
                "DecodeParms": {
                    "Predictor": 2,
                    "Colors": colors,
                    "Columns": columns,
                    "BitsPerComponent": bpc,
                },
            },
            compressed,
        )
        stream.decode()
        expected = bytes([10, 20, 30, 40, 20, 40, 60, 80])
        assert stream.data == expected


class TestExceptionAliases:
    """Tests for exception aliases in pdftypes.py"""

    def test_pdfexception_alias(self) -> None:
        from pdfminer import pdftypes

        assert pdftypes.PDFException is PDFException

    def test_pdftypeerror_alias(self) -> None:
        from pdfminer import pdftypes

        assert pdftypes.PDFTypeError is PDFTypeError

    def test_pdfvalueerror_alias(self) -> None:
        from pdfminer import pdftypes

        assert pdftypes.PDFValueError is PDFValueError

    def test_pdfobjectnotfound_alias(self) -> None:
        from pdfminer import pdftypes

        assert pdftypes.PDFObjectNotFound is PDFObjectNotFound

    def test_pdfnotimplementederror_alias(self) -> None:
        from pdfminer import pdftypes

        assert pdftypes.PDFNotImplementedError is PDFNotImplementedError


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_resolve1_with_none(self) -> None:
        assert resolve1(None) is None

    def test_resolve_all_empty_list(self) -> None:
        assert resolve_all([]) == []

    def test_resolve_all_empty_dict(self) -> None:
        assert resolve_all({}) == {}

    def test_pdfstream_empty_rawdata(self) -> None:
        stream = PDFStream({}, b"")
        stream.decode()
        assert stream.data == b""

    def test_pdfstream_filter_with_empty_params(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "DecodeParms": {}},
            compressed,
        )
        stream.decode()
        assert stream.data == original

    def test_pdfstream_abbreviated_filter_names(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream({"F": LIT("Fl")}, compressed)
        stream.decode()
        assert stream.data == original

    def test_pdfstream_dp_abbreviation(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "DP": {"Predictor": 1}},
            compressed,
        )
        stream.decode()
        assert stream.data == original

    def test_large_objid_pdfobjref(self) -> None:
        doc = mock_doc({999999: "large_id_value"})
        ref = PDFObjRef(doc, 999999)
        assert ref.resolve() == "large_id_value"
        assert repr(ref) == "<PDFObjRef:999999>"

    def test_pdfstream_filter_reference(self) -> None:
        doc = mock_doc({1: LIT("FlateDecode")})
        filter_ref = PDFObjRef(doc, 1)
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream({"Filter": filter_ref}, compressed)
        stream.decode()
        assert stream.data == original

    def test_decipher_all_preserves_objid_genno(self) -> None:
        received_params: list[tuple[int, int]] = []

        def mock_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            received_params.append((objid, genno))
            return data

        decipher_all(mock_decipher, 42, 7, b"test")
        assert received_params == [(42, 7)]

    def test_pdfstream_single_param_replicated_for_multiple_filters(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        from base64 import a85encode

        ascii85_encoded = b"<~" + a85encode(compressed) + b"~>"
        stream = PDFStream(
            {
                "Filter": [LIT("ASCII85Decode"), LIT("FlateDecode")],
                "DecodeParms": {"Predictor": 1},
            },
            ascii85_encoded,
        )
        filters = stream.get_filters()
        assert len(filters) == 2
        assert filters[0][1] == {"Predictor": 1}
        assert filters[1][1] == {"Predictor": 1}


class TestLZWDecode:
    """Tests for LZW decoding in PDFStream"""

    def test_lzw_decode(self) -> None:
        lzw_encoded = bytes([0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01])
        stream = PDFStream({"Filter": LIT("LZWDecode")}, lzw_encoded)
        stream.decode()
        assert stream.data is not None

    def test_lzw_abbreviated(self) -> None:
        lzw_encoded = bytes([0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01])
        stream = PDFStream({"Filter": LIT("LZW")}, lzw_encoded)
        stream.decode()
        assert stream.data is not None


class TestDecipherCallableProtocol:
    """Tests for DecipherCallable protocol"""

    def test_decipher_callable_signature(self) -> None:
        def valid_decipher(
            objid: int,
            genno: int,
            data: bytes,
            attrs: dict[str, Any] | None = None,
        ) -> bytes:
            return data

        stream = PDFStream({}, b"test", decipher=valid_decipher)
        stream.set_objid(1, 0)
        stream.decode()
        assert stream.data == b"test"


class TestPDFStreamFDecodeParms:
    """Tests for FDecodeParms attribute handling"""

    def test_fdecode_parms(self) -> None:
        original = b"test"
        compressed = zlib.compress(original)
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "FDecodeParms": {"Predictor": 1}},
            compressed,
        )
        stream.decode()
        assert stream.data == original
