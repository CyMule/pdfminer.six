"""Comprehensive tests for pdfminer/cmapdb.py

Tests for:
- CMapBase class
- CMap class
- IdentityCMap and IdentityCMapByte classes
- UnicodeMap class
- IdentityUnicodeMap class
- FileCMap class
- FileUnicodeMap class
- PyCMap and PyUnicodeMap classes
- CMapDB class (singleton pattern, caching, loading)
- CMapParser class
- Character code to CID mapping
- CID to Unicode mapping
"""

import io
import struct
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from pdfminer.cmapdb import (
    CMap,
    CMapBase,
    CMapDB,
    CMapError,
    CMapParser,
    FileCMap,
    FileUnicodeMap,
    IdentityCMap,
    IdentityCMapByte,
    IdentityUnicodeMap,
    PyCMap,
    PyUnicodeMap,
    UnicodeMap,
)
from pdfminer.pdfexceptions import PDFTypeError
from pdfminer.psparser import PSLiteral


class TestCMapError:
    """Tests for CMapError exception class"""

    def test_cmap_error_instantiation(self):
        error = CMapError("Test error message")
        assert str(error) == "Test error message"

    def test_cmap_error_is_pdf_exception(self):
        from pdfminer.pdfexceptions import PDFException

        error = CMapError("Test")
        assert isinstance(error, PDFException)


class TestCMapBase:
    """Tests for CMapBase abstract base class"""

    def test_init_with_no_kwargs(self):
        cmap = CMapBase()
        assert cmap.attrs == {}

    def test_init_with_kwargs(self):
        cmap = CMapBase(CMapName="TestMap", WMode=1)
        assert cmap.attrs["CMapName"] == "TestMap"
        assert cmap.attrs["WMode"] == 1

    def test_is_vertical_default(self):
        cmap = CMapBase()
        assert cmap.is_vertical() is False

    def test_is_vertical_with_wmode_0(self):
        cmap = CMapBase(WMode=0)
        assert cmap.is_vertical() is False

    def test_is_vertical_with_wmode_1(self):
        cmap = CMapBase(WMode=1)
        assert cmap.is_vertical() is True

    def test_is_vertical_with_nonzero_wmode(self):
        cmap = CMapBase(WMode=2)
        assert cmap.is_vertical() is True

    def test_set_attr(self):
        cmap = CMapBase()
        cmap.set_attr("CMapName", "NewName")
        assert cmap.attrs["CMapName"] == "NewName"

    def test_set_attr_overwrite(self):
        cmap = CMapBase(CMapName="OldName")
        cmap.set_attr("CMapName", "NewName")
        assert cmap.attrs["CMapName"] == "NewName"

    def test_add_code2cid_noop(self):
        cmap = CMapBase()
        cmap.add_code2cid("test", 123)

    def test_add_cid2unichr_noop(self):
        cmap = CMapBase()
        cmap.add_cid2unichr(123, b"\x00A")

    def test_use_cmap_noop(self):
        cmap1 = CMapBase()
        cmap2 = CMapBase()
        cmap1.use_cmap(cmap2)

    def test_decode_not_implemented(self):
        cmap = CMapBase()
        with pytest.raises(NotImplementedError):
            list(cmap.decode(b"\x00\x01"))

    def test_debug_class_variable(self):
        assert CMapBase.debug == 0


class TestCMap:
    """Tests for CMap class"""

    def test_init_with_no_kwargs(self):
        cmap = CMap()
        assert cmap.code2cid == {}
        assert cmap.attrs == {}

    def test_init_with_kwargs(self):
        cmap = CMap(CMapName="TestCMap", WMode=0)
        assert cmap.attrs["CMapName"] == "TestCMap"

    def test_repr_with_name(self):
        cmap = CMap(CMapName="MyCMap")
        assert repr(cmap) == "<CMap: MyCMap>"

    def test_repr_without_name(self):
        cmap = CMap()
        assert repr(cmap) == "<CMap: None>"

    def test_decode_empty_code2cid(self):
        cmap = CMap()
        result = list(cmap.decode(b"\x00\x01\x02"))
        assert result == []

    def test_decode_simple_mapping(self):
        cmap = CMap()
        cmap.code2cid[0x41] = 100
        result = list(cmap.decode(b"A"))
        assert result == [100]

    def test_decode_multi_byte(self):
        cmap = CMap()
        cmap.code2cid[0x00] = {0x41: 200}
        result = list(cmap.decode(b"\x00A"))
        assert result == [200]

    def test_decode_nested_mapping(self):
        cmap = CMap()
        cmap.code2cid[0x00] = {0x01: {0x02: 300}}
        result = list(cmap.decode(b"\x00\x01\x02"))
        assert result == [300]

    def test_decode_mixed_mapping(self):
        cmap = CMap()
        cmap.code2cid[0x41] = 100
        cmap.code2cid[0x42] = 101
        cmap.code2cid[0x00] = {0x43: 200}
        result = list(cmap.decode(b"AB\x00C"))
        assert result == [100, 101, 200]

    def test_decode_unknown_byte_skipped(self):
        cmap = CMap()
        cmap.code2cid[0x41] = 100
        result = list(cmap.decode(b"AXA"))
        assert result == [100, 100]

    def test_decode_incomplete_multi_byte(self):
        cmap = CMap()
        cmap.code2cid[0x00] = {0x41: 200}
        result = list(cmap.decode(b"\x00\x42"))
        assert result == []

    def test_use_cmap_simple(self):
        cmap1 = CMap()
        cmap2 = CMap()
        cmap2.code2cid[0x41] = 100
        cmap2.code2cid[0x42] = 101
        cmap1.use_cmap(cmap2)
        assert cmap1.code2cid == {0x41: 100, 0x42: 101}

    def test_use_cmap_nested(self):
        cmap1 = CMap()
        cmap2 = CMap()
        cmap2.code2cid[0x00] = {0x41: 200, 0x42: 201}
        cmap1.use_cmap(cmap2)
        assert 0x00 in cmap1.code2cid
        assert cmap1.code2cid[0x00][0x41] == 200

    def test_use_cmap_preserves_existing(self):
        cmap1 = CMap()
        cmap1.code2cid[0x43] = 300
        cmap2 = CMap()
        cmap2.code2cid[0x41] = 100
        cmap1.use_cmap(cmap2)
        assert cmap1.code2cid[0x43] == 300
        assert cmap1.code2cid[0x41] == 100

    def test_use_cmap_assert_type(self):
        cmap1 = CMap()
        cmap2 = CMapBase()
        with pytest.raises(AssertionError):
            cmap1.use_cmap(cmap2)

    def test_dump_to_stdout(self):
        cmap = CMap()
        cmap.code2cid[0x41] = 100
        cmap.code2cid[0x42] = 101
        output = io.StringIO()
        cmap.dump(out=output)
        result = output.getvalue()
        assert "code (65,) = cid 100" in result
        assert "code (66,) = cid 101" in result

    def test_dump_nested(self):
        cmap = CMap()
        cmap.code2cid[0x00] = {0x41: 200}
        output = io.StringIO()
        cmap.dump(out=output)
        result = output.getvalue()
        assert "code (0, 65) = cid 200" in result


class TestIdentityCMap:
    """Tests for IdentityCMap class"""

    def test_init_default(self):
        cmap = IdentityCMap()
        assert cmap.is_vertical() is False

    def test_init_horizontal(self):
        cmap = IdentityCMap(WMode=0)
        assert cmap.is_vertical() is False

    def test_init_vertical(self):
        cmap = IdentityCMap(WMode=1)
        assert cmap.is_vertical() is True

    def test_decode_empty_buffer(self):
        cmap = IdentityCMap()
        result = cmap.decode(b"")
        assert result == ()

    def test_decode_single_short(self):
        cmap = IdentityCMap()
        result = cmap.decode(b"\x00A")
        assert result == (0x0041,)

    def test_decode_multiple_shorts(self):
        cmap = IdentityCMap()
        buffer = struct.pack(">3H", 0x0041, 0x0042, 0x0043)
        result = cmap.decode(buffer)
        assert result == (0x0041, 0x0042, 0x0043)

    def test_decode_odd_length_buffer(self):
        cmap = IdentityCMap()
        buffer = struct.pack(">3H", 0x0041, 0x0042, 0x0043) + b"\x00"
        result = cmap.decode(buffer)
        assert result == (0x0041, 0x0042, 0x0043)

    def test_decode_single_byte_returns_empty(self):
        cmap = IdentityCMap()
        result = cmap.decode(b"\x41")
        assert result == ()

    def test_decode_max_values(self):
        cmap = IdentityCMap()
        buffer = struct.pack(">H", 0xFFFF)
        result = cmap.decode(buffer)
        assert result == (0xFFFF,)


class TestIdentityCMapByte:
    """Tests for IdentityCMapByte class"""

    def test_decode_empty_buffer(self):
        cmap = IdentityCMapByte()
        result = cmap.decode(b"")
        assert result == ()

    def test_decode_single_byte(self):
        cmap = IdentityCMapByte()
        result = cmap.decode(b"A")
        assert result == (0x41,)

    def test_decode_multiple_bytes(self):
        cmap = IdentityCMapByte()
        result = cmap.decode(b"ABC")
        assert result == (0x41, 0x42, 0x43)

    def test_decode_full_byte_range(self):
        cmap = IdentityCMapByte()
        buffer = bytes(range(256))
        result = cmap.decode(buffer)
        assert result == tuple(range(256))


class TestUnicodeMap:
    """Tests for UnicodeMap class"""

    def test_init_default(self):
        umap = UnicodeMap()
        assert umap.cid2unichr == {}

    def test_init_with_kwargs(self):
        umap = UnicodeMap(CMapName="TestUMap", WMode=1)
        assert umap.attrs["CMapName"] == "TestUMap"
        assert umap.is_vertical() is True

    def test_repr_with_name(self):
        umap = UnicodeMap(CMapName="MyUMap")
        assert repr(umap) == "<UnicodeMap: MyUMap>"

    def test_repr_without_name(self):
        umap = UnicodeMap()
        assert repr(umap) == "<UnicodeMap: None>"

    def test_get_unichr(self):
        umap = UnicodeMap()
        umap.cid2unichr[100] = "A"
        assert umap.get_unichr(100) == "A"

    def test_get_unichr_key_error(self):
        umap = UnicodeMap()
        with pytest.raises(KeyError):
            umap.get_unichr(999)

    def test_dump(self):
        umap = UnicodeMap()
        umap.cid2unichr[100] = "A"
        umap.cid2unichr[101] = "B"
        output = io.StringIO()
        umap.dump(out=output)
        result = output.getvalue()
        assert "cid 100 = unicode 'A'" in result
        assert "cid 101 = unicode 'B'" in result


class TestIdentityUnicodeMap:
    """Tests for IdentityUnicodeMap class"""

    def test_get_unichr_ascii(self):
        umap = IdentityUnicodeMap()
        assert umap.get_unichr(65) == "A"
        assert umap.get_unichr(97) == "a"

    def test_get_unichr_unicode(self):
        umap = IdentityUnicodeMap()
        assert umap.get_unichr(0x4E2D) == "\u4e2d"

    def test_get_unichr_zero(self):
        umap = IdentityUnicodeMap()
        assert umap.get_unichr(0) == "\x00"

    def test_get_unichr_max_bmp(self):
        umap = IdentityUnicodeMap()
        assert umap.get_unichr(0xFFFF) == "\uffff"


class TestFileCMap:
    """Tests for FileCMap class"""

    def test_add_code2cid_single_char(self):
        cmap = FileCMap()
        cmap.add_code2cid("A", 100)
        assert cmap.code2cid[0x41] == 100

    def test_add_code2cid_multi_char(self):
        cmap = FileCMap()
        cmap.add_code2cid("AB", 200)
        assert 0x41 in cmap.code2cid
        assert cmap.code2cid[0x41][0x42] == 200

    def test_add_code2cid_three_char(self):
        cmap = FileCMap()
        cmap.add_code2cid("ABC", 300)
        assert cmap.code2cid[0x41][0x42][0x43] == 300

    def test_add_code2cid_extend_existing(self):
        cmap = FileCMap()
        cmap.add_code2cid("AB", 200)
        cmap.add_code2cid("AC", 201)
        assert cmap.code2cid[0x41][0x42] == 200
        assert cmap.code2cid[0x41][0x43] == 201

    def test_decode_after_add(self):
        cmap = FileCMap()
        cmap.add_code2cid("A", 100)
        cmap.add_code2cid("BC", 200)
        result = list(cmap.decode(b"ABC"))
        assert result == [100, 200]


class TestFileUnicodeMap:
    """Tests for FileUnicodeMap class"""

    def test_add_cid2unichr_with_bytes(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, b"\x00A")
        assert umap.cid2unichr[100] == "A"

    def test_add_cid2unichr_with_utf16be(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, b"\x4e\x2d")
        assert umap.cid2unichr[100] == "\u4e2d"

    def test_add_cid2unichr_with_int(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, 0x41)
        assert umap.cid2unichr[100] == "A"

    def test_add_cid2unichr_with_psliteral(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, PSLiteral("space"))
        assert umap.cid2unichr[100] == " "

    def test_add_cid2unichr_invalid_type(self):
        umap = FileUnicodeMap()
        with pytest.raises(PDFTypeError):
            umap.add_cid2unichr(100, [1, 2, 3])

    def test_add_cid2unichr_nbsp_collision(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, 0x20)
        umap.add_cid2unichr(100, 0xA0)
        assert umap.cid2unichr[100] == " "

    def test_add_cid2unichr_nbsp_no_collision(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, 0xA0)
        assert umap.cid2unichr[100] == "\u00a0"


class TestPyCMap:
    """Tests for PyCMap class"""

    def test_init(self):
        module = MagicMock()
        module.CODE2CID = {0x41: 100}
        module.IS_VERTICAL = False
        cmap = PyCMap("TestPyCMap", module)
        assert cmap.attrs["CMapName"] == "TestPyCMap"
        assert cmap.code2cid == {0x41: 100}
        assert cmap.is_vertical() is False

    def test_init_vertical(self):
        module = MagicMock()
        module.CODE2CID = {}
        module.IS_VERTICAL = True
        cmap = PyCMap("VertPyCMap", module)
        assert cmap.is_vertical() is True


class TestPyUnicodeMap:
    """Tests for PyUnicodeMap class"""

    def test_init_horizontal(self):
        module = MagicMock()
        module.CID2UNICHR_H = {100: "A"}
        module.CID2UNICHR_V = {100: "B"}
        umap = PyUnicodeMap("TestPyUMap", module, vertical=False)
        assert umap.cid2unichr == {100: "A"}
        assert umap.is_vertical() is False

    def test_init_vertical(self):
        module = MagicMock()
        module.CID2UNICHR_H = {100: "A"}
        module.CID2UNICHR_V = {100: "B"}
        umap = PyUnicodeMap("TestPyUMap", module, vertical=True)
        assert umap.cid2unichr == {100: "B"}
        assert umap.is_vertical() is True


class TestCMapDB:
    """Tests for CMapDB singleton class"""

    def test_get_cmap_identity_h(self):
        cmap = CMapDB.get_cmap("Identity-H")
        assert isinstance(cmap, IdentityCMap)
        assert cmap.is_vertical() is False

    def test_get_cmap_identity_v(self):
        cmap = CMapDB.get_cmap("Identity-V")
        assert isinstance(cmap, IdentityCMap)
        assert cmap.is_vertical() is True

    def test_get_cmap_onebyte_identity_h(self):
        cmap = CMapDB.get_cmap("OneByteIdentityH")
        assert isinstance(cmap, IdentityCMapByte)
        assert cmap.is_vertical() is False

    def test_get_cmap_onebyte_identity_v(self):
        cmap = CMapDB.get_cmap("OneByteIdentityV")
        assert isinstance(cmap, IdentityCMapByte)
        assert cmap.is_vertical() is True

    def test_get_cmap_h(self):
        cmap = CMapDB.get_cmap("H")
        assert isinstance(cmap, PyCMap)
        assert str(cmap) == "<CMap: H>"

    def test_get_cmap_caching(self):
        cmap1 = CMapDB.get_cmap("H")
        cmap2 = CMapDB.get_cmap("H")
        assert cmap1 is cmap2

    def test_get_cmap_not_found(self):
        with pytest.raises(CMapDB.CMapNotFound):
            CMapDB.get_cmap("NonExistentCMap12345")

    def test_get_unicode_map(self):
        umap = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        assert isinstance(umap, PyUnicodeMap)
        assert str(umap) == "<UnicodeMap: Adobe-Japan1>"

    def test_get_unicode_map_vertical(self):
        umap_h = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        umap_v = CMapDB.get_unicode_map("Adobe-Japan1", vertical=True)
        assert umap_h.is_vertical() is False
        assert umap_v.is_vertical() is True

    def test_get_unicode_map_caching(self):
        umap1 = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        umap2 = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        assert umap1 is umap2

    def test_get_unicode_map_not_found(self):
        with pytest.raises(CMapDB.CMapNotFound):
            CMapDB.get_unicode_map("NonExistentUMap12345")

    def test_convert_code2cid_keys_simple(self):
        d = {"0": 100, "1": 101}
        result = CMapDB._convert_code2cid_keys(d)
        assert result == {0: 100, 1: 101}

    def test_convert_code2cid_keys_nested(self):
        d = {"0": {"65": 100, "66": 101}}
        result = CMapDB._convert_code2cid_keys(d)
        assert result == {0: {65: 100, 66: 101}}

    def test_convert_code2cid_keys_int_passthrough(self):
        result = CMapDB._convert_code2cid_keys(42)
        assert result == 42

    def test_cmap_not_found_is_cmap_error(self):
        assert issubclass(CMapDB.CMapNotFound, CMapError)


class TestCMapParser:
    """Tests for CMapParser class"""

    def test_parser_init(self):
        cmap = FileCMap()
        fp = BytesIO(b"")
        parser = CMapParser(cmap, fp)
        assert parser.cmap is cmap
        assert parser._in_cmap is True

    def test_parser_begincmap_endcmap(self):
        cmap = FileCMap()
        data = b"""
        %!PS-Adobe-3.0 Resource-CMap
        begincmap
        /CMapName /TestCMap def
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()
        cmap_name = cmap.attrs.get("CMapName")
        assert isinstance(cmap_name, PSLiteral)
        assert cmap_name.name == "TestCMap"

    def test_parser_def_keyword(self):
        cmap = FileCMap()
        data = b"""
        begincmap
        /CMapName /MyCMap def
        /CMapVersion 1 def
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()
        cmap_name = cmap.attrs.get("CMapName")
        assert isinstance(cmap_name, PSLiteral)
        assert cmap_name.name == "MyCMap"
        assert cmap.attrs.get("CMapVersion") == 1

    def test_parser_beginbfchar_endbfchar(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfchar
        <0041> <0042>
        endbfchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(0x41) == "B"

    def test_parser_multiple_bfchar(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        3 beginbfchar
        <0001> <0041>
        <0002> <0042>
        <0003> <0043>
        endbfchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(1) == "A"
        assert umap.cid2unichr.get(2) == "B"
        assert umap.cid2unichr.get(3) == "C"

    def test_parser_beginbfrange_endbfrange_bytes(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        <0001> <0003> <0041>
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(1) == "A"
        assert umap.cid2unichr.get(2) == "B"
        assert umap.cid2unichr.get(3) == "C"

    def test_parser_beginbfrange_endbfrange_array(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        <0001> <0003> [<0058> <0059> <005A>]
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(1) == "X"
        assert umap.cid2unichr.get(2) == "Y"
        assert umap.cid2unichr.get(3) == "Z"

    def test_parser_begincidchar_endcidchar(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        2 begincidchar
        100 <0041>
        101 <0042>
        endcidchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(100) == "A"
        assert umap.cid2unichr.get(101) == "B"

    def test_parser_begincidrange_endcidrange(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        <0041> <0043> 100
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(100) == "A"
        assert umap.cid2unichr.get(101) == "B"
        assert umap.cid2unichr.get(102) == "C"

    def test_parser_begincodespacerange(self):
        cmap = FileCMap()
        data = b"""
        begincmap
        1 begincodespacerange
        <00> <FF>
        endcodespacerange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()

    def test_parser_beginnotdefrange(self):
        cmap = FileCMap()
        data = b"""
        begincmap
        1 beginnotdefrange
        <0000> <001f> 0
        endnotdefrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()

    def test_parser_without_begincmap(self):
        umap = FileUnicodeMap()
        data = b"""
        1 beginbfchar
        <0041> <0042>
        endbfchar
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(0x41) == "B"

    def test_parser_usecmap_not_found(self):
        cmap = FileCMap()
        data = b"""
        begincmap
        /NonExistentCMap usecmap
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()

    def test_parser_warn_once_bfrange_start_not_bytes(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        123 <0003> <0041>
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_bfrange_end_not_bytes(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        <0001> 123 <0041>
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_bfrange_length_mismatch(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        <0001> <000003> <0041>
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_cidrange_start_not_bytes(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        123 <0043> 100
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_cidrange_end_not_bytes(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        <0041> 123 100
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_cidrange_cid_not_int(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        <0041> <0043> <0064>
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_cidrange_length_mismatch(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        <0041> <000043> 100
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_warn_once_cidrange_prefix_mismatch(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 begincidrange
        <0000000041> <0100000043> 100
        endcidrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0


class TestCMapDBCachingBehavior:
    """Tests for CMapDB caching behavior"""

    def test_cmap_cache_class_variable(self):
        assert hasattr(CMapDB, "_cmap_cache")
        assert isinstance(CMapDB._cmap_cache, dict)

    def test_umap_cache_class_variable(self):
        assert hasattr(CMapDB, "_umap_cache")
        assert isinstance(CMapDB._umap_cache, dict)


class TestCMapDBLoadData:
    """Tests for CMapDB._load_data method"""

    def test_load_data_sanitizes_null_chars(self):
        with pytest.raises(CMapDB.CMapNotFound):
            CMapDB._load_data("test\x00name")


class TestCMapIntegration:
    """Integration tests combining multiple components"""

    def test_cmap_decode_with_file_cmap(self):
        cmap = FileCMap()
        cmap.add_code2cid("A", 1)
        cmap.add_code2cid("B", 2)
        cmap.add_code2cid("CD", 3)
        result = list(cmap.decode(b"ABCD"))
        assert result == [1, 2, 3]

    def test_unicode_map_from_parser(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        /CMapName /TestToUnicode def
        3 beginbfchar
        <0001> <0048>
        <0002> <0069>
        <0003> <0021>
        endbfchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        cmap_name = umap.attrs.get("CMapName")
        assert isinstance(cmap_name, PSLiteral)
        assert cmap_name.name == "TestToUnicode"
        assert umap.get_unichr(1) == "H"
        assert umap.get_unichr(2) == "i"
        assert umap.get_unichr(3) == "!"

    def test_cmap_with_usecmap(self):
        base_cmap = CMap()
        base_cmap.code2cid[0x41] = 100
        base_cmap.code2cid[0x42] = 101

        derived_cmap = CMap()
        derived_cmap.use_cmap(base_cmap)
        derived_cmap.code2cid[0x43] = 102

        result = list(derived_cmap.decode(b"ABC"))
        assert result == [100, 101, 102]

    def test_identity_cmap_with_chinese_characters(self):
        cmap = IdentityCMap()
        buffer = struct.pack(">H", 0x4E2D)
        result = cmap.decode(buffer)
        assert result == (0x4E2D,)

    def test_file_unicode_map_utf16be_surrogates(self):
        umap = FileUnicodeMap()
        umap.add_cid2unichr(100, b"\xD8\x3D\xDE\x00")
        assert umap.cid2unichr[100] == "\U0001F600"


class TestCMapParserEdgeCases:
    """Edge case tests for CMapParser"""

    def test_parser_empty_input(self):
        cmap = FileCMap()
        fp = BytesIO(b"")
        parser = CMapParser(cmap, fp)
        parser.run()
        assert cmap.code2cid == {}

    def test_parser_only_comments(self):
        cmap = FileCMap()
        data = b"""
        % This is a comment
        % Another comment
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        parser.run()

    def test_parser_malformed_def(self):
        cmap = FileCMap()
        data = b"""
        begincmap
        /CMapName def
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(cmap, fp)
        with pytest.raises(ValueError):
            parser.run()

    def test_parser_bfrange_array_length_mismatch(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        1 beginbfrange
        <0001> <0005> [<0041> <0042>]
        endbfrange
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert umap.cid2unichr.get(1) == "A"
        assert umap.cid2unichr.get(2) == "B"
        assert 3 not in umap.cid2unichr

    def test_parser_bfchar_wrong_types(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        2 beginbfchar
        123 <0042>
        <0043> 456
        endbfchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0

    def test_parser_cidchar_wrong_types(self):
        umap = FileUnicodeMap()
        data = b"""
        begincmap
        2 begincidchar
        <0064> <0041>
        100 123
        endcidchar
        endcmap
        """
        fp = BytesIO(data)
        parser = CMapParser(umap, fp)
        parser.run()
        assert len(umap.cid2unichr) == 0


class TestCMapVerticalWriting:
    """Tests for vertical writing mode support"""

    def test_cmap_base_wmode_variations(self):
        assert CMapBase(WMode=0).is_vertical() is False
        assert CMapBase(WMode=1).is_vertical() is True
        assert CMapBase(WMode=2).is_vertical() is True
        assert CMapBase(WMode=-1).is_vertical() is True

    def test_unicode_map_vertical_from_db(self):
        umap_h = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        umap_v = CMapDB.get_unicode_map("Adobe-Japan1", vertical=True)
        assert umap_h is not umap_v

    def test_identity_cmap_vertical(self):
        cmap_h = IdentityCMap(WMode=0)
        cmap_v = IdentityCMap(WMode=1)
        assert cmap_h.is_vertical() is False
        assert cmap_v.is_vertical() is True
        buffer = struct.pack(">H", 0x0041)
        assert cmap_h.decode(buffer) == cmap_v.decode(buffer)


class TestCMapDBCode2CIDKeyConversion:
    """Tests for CODE2CID key conversion from strings to integers"""

    def test_code2cid_has_integer_keys(self):
        cmap = CMapDB.get_cmap("H")
        for key in cmap.code2cid:
            assert isinstance(key, int), f"Expected int key, got {type(key)}: {key}"

    def test_code2cid_nested_integer_keys(self):
        cmap = CMapDB.get_cmap("H")

        def check_all_keys(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    assert isinstance(k, int), f"Expected int key, got {type(k)}: {k}"
                    check_all_keys(v)

        check_all_keys(cmap.code2cid)


class TestUnicodeMapCID2UnicharKeyConversion:
    """Tests for CID2UNICHR key conversion from strings to integers"""

    def test_cid2unichr_has_integer_keys(self):
        umap = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        for key in umap.cid2unichr:
            assert isinstance(key, int), f"Expected int key, got {type(key)}: {key}"

    def test_cid2unichr_known_mapping(self):
        umap = CMapDB.get_unicode_map("Adobe-Japan1", vertical=False)
        assert 1 in umap.cid2unichr
        assert umap.cid2unichr[1] == " "
