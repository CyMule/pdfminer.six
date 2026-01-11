"""Comprehensive tests for pdfminer/pdffont.py"""

from io import BytesIO
from typing import Any
from unittest.mock import patch

import pytest

from pdfminer.cmapdb import FileUnicodeMap, IdentityUnicodeMap
from pdfminer.pdffont import (
    IDENTITY_ENCODER,
    CFFFont,
    FontMetricsDB,
    PDFCIDFont,
    PDFFont,
    PDFFontError,
    PDFSimpleFont,
    PDFTrueTypeFont,
    PDFType1Font,
    PDFType3Font,
    PDFUnicodeNotDefined,
    TrueTypeFont,
    Type1FontHeaderParser,
    get_widths,
    get_widths2,
    getdict,
)
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdftypes import PDFStream
from pdfminer.psparser import LIT, PSLiteral


class MockPDFFont(PDFFont):
    """Mock implementation of PDFFont for testing base class functionality."""

    def __init__(
        self,
        descriptor: dict[str, Any] | None = None,
        widths: dict[str | int, float] | None = None,
        default_width: float | None = None,
    ):
        if descriptor is None:
            descriptor = {}
        if widths is None:
            widths = {}
        super().__init__(descriptor, widths, default_width)

    def to_unichr(self, cid: int) -> str:
        return chr(cid) if cid < 256 else str(cid)


class TestGetWidths:
    """Tests for the get_widths function."""

    def test_empty_sequence(self):
        result = get_widths([])
        assert result == {}

    def test_simple_list_format(self):
        result = get_widths([0, [100, 200, 300]])
        assert result == {0: 100, 1: 200, 2: 300}

    def test_range_format(self):
        result = get_widths([0, 3, 500])
        assert result == {0: 500, 1: 500, 2: 500, 3: 500}

    def test_multiple_ranges(self):
        result = get_widths([0, 2, 100, 5, 7, 200])
        assert result == {0: 100, 1: 100, 2: 100, 5: 200, 6: 200, 7: 200}

    def test_mixed_formats(self):
        result = get_widths([0, [100, 200], 5, 7, 300])
        assert result == {0: 100, 1: 200, 5: 300, 6: 300, 7: 300}

    def test_single_char_width(self):
        result = get_widths([42, [600]])
        assert result == {42: 600}

    def test_float_widths(self):
        result = get_widths([0, [100.5, 200.75]])
        assert result == {0: 100.5, 1: 200.75}


class TestGetWidths2:
    """Tests for the get_widths2 function (vertical writing)."""

    def test_empty_sequence(self):
        result = get_widths2([])
        assert result == {}

    def test_simple_list_format(self):
        result = get_widths2([0, [100, 10, 20, 200, 30, 40]])
        assert result == {0: (100, (10, 20)), 1: (200, (30, 40))}

    def test_range_format(self):
        result = get_widths2([0, 2, 500, 10, 20])
        assert result == {
            0: (500, (10, 20)),
            1: (500, (10, 20)),
            2: (500, (10, 20)),
        }


class TestGetDict:
    """Tests for the getdict function (CFF font dictionary parsing)."""

    def test_empty_data(self):
        result = getdict(b"")
        assert result == {}

    def test_single_byte_value(self):
        data = bytes([139 + 10, 1])
        result = getdict(data)
        assert result == {1: [10]}

    def test_negative_value(self):
        data = bytes([139 - 10, 2])
        result = getdict(data)
        assert result == {2: [-10]}


class TestFontMetricsDB:
    """Tests for the FontMetricsDB class."""

    def test_get_metrics_for_standard_font(self):
        descriptor, widths = FontMetricsDB.get_metrics("Helvetica")
        assert "FontName" in descriptor
        assert descriptor["FontName"] == "Helvetica"
        assert len(widths) > 0

    def test_get_metrics_for_nonexistent_font(self):
        with pytest.raises(KeyError):
            FontMetricsDB.get_metrics("NonexistentFont")


class TestPDFFont:
    """Tests for the PDFFont base class."""

    def test_init_with_empty_descriptor(self):
        font = MockPDFFont({}, {})
        assert font.fontname == "unknown"
        assert font.flags == 0
        assert font.ascent == 0
        assert font.descent == 0
        assert font.italic_angle == 0
        assert font.default_width == 0
        assert font.leading == 0
        assert font.bbox == (0.0, 0.0, 0.0, 0.0)
        assert font.hscale == 0.001
        assert font.vscale == 0.001

    def test_init_with_font_descriptor(self):
        descriptor = {
            "FontName": LIT("TestFont"),
            "Flags": 32,
            "Ascent": 800,
            "Descent": -200,
            "ItalicAngle": -12,
            "MissingWidth": 500,
            "Leading": 100,
            "FontBBox": [0, -200, 1000, 800],
        }
        font = MockPDFFont(descriptor, {})
        assert font.fontname == "TestFont"
        assert font.flags == 32
        assert font.ascent == 800
        assert font.descent == -200
        assert font.italic_angle == -12
        assert font.default_width == 500
        assert font.leading == 100
        assert font.bbox == (0.0, -200.0, 1000.0, 800.0)

    def test_init_with_string_fontname(self):
        descriptor = {"FontName": "TestFont"}
        font = MockPDFFont(descriptor, {})
        assert font.fontname == "TestFont"

    def test_init_with_positive_descent_corrected(self):
        descriptor = {"Descent": 200}
        font = MockPDFFont(descriptor, {})
        assert font.descent == -200

    def test_init_with_default_width_override(self):
        descriptor = {"MissingWidth": 500}
        font = MockPDFFont(descriptor, {}, default_width=600)
        assert font.default_width == 600

    def test_repr(self):
        font = MockPDFFont({}, {})
        assert repr(font) == "<PDFFont>"

    def test_is_vertical_default(self):
        font = MockPDFFont({}, {})
        assert font.is_vertical() is False

    def test_is_multibyte_default(self):
        font = MockPDFFont({}, {})
        assert font.is_multibyte() is False

    def test_decode(self):
        font = MockPDFFont({}, {})
        result = list(font.decode(b"ABC"))
        assert result == [65, 66, 67]

    def test_get_ascent(self):
        descriptor = {"Ascent": 800}
        font = MockPDFFont(descriptor, {})
        assert font.get_ascent() == 0.8

    def test_get_descent(self):
        descriptor = {"Descent": -200}
        font = MockPDFFont(descriptor, {})
        assert font.get_descent() == -0.2

    def test_get_width(self):
        descriptor = {"FontBBox": [0, -200, 1000, 800]}
        font = MockPDFFont(descriptor, {})
        assert font.get_width() == 1.0

    def test_get_width_with_zero_bbox(self):
        descriptor = {"FontBBox": [0, 0, 0, 0], "MissingWidth": 500}
        font = MockPDFFont(descriptor, {})
        assert font.get_width() == -0.5

    def test_get_height(self):
        descriptor = {"FontBBox": [0, -200, 1000, 800]}
        font = MockPDFFont(descriptor, {})
        assert font.get_height() == 1.0

    def test_get_height_with_zero_bbox(self):
        descriptor = {"FontBBox": [0, 0, 0, 0], "Ascent": 800, "Descent": -200}
        font = MockPDFFont(descriptor, {})
        assert font.get_height() == 1.0

    def test_char_width_with_int_cid(self):
        widths: dict[str | int, float] = {65: 600.0}
        font = MockPDFFont({}, widths)
        assert font.char_width(65) == 0.6

    def test_char_width_with_str_cid(self):
        widths: dict[str | int, float] = {"A": 600.0}
        font = MockPDFFont({}, widths)
        assert font.char_width(65) == 0.6

    def test_char_width_with_default(self):
        font = MockPDFFont({"MissingWidth": 500}, {})
        assert font.char_width(65) == 0.5

    def test_char_disp_default(self):
        font = MockPDFFont({}, {})
        assert font.char_disp(65) == 0

    def test_string_width(self):
        widths: dict[str | int, float] = {65: 600.0, 66: 700.0, 67: 800.0}
        font = MockPDFFont({}, widths)
        assert font.string_width(b"ABC") == pytest.approx(2.1)

    def test_to_unichr_not_implemented(self):
        font = PDFFont({}, {})
        with pytest.raises(NotImplementedError):
            font.to_unichr(65)

    def test_parse_bbox_valid(self):
        descriptor = {"FontBBox": [0, -200, 1000, 800]}
        bbox = PDFFont._parse_bbox(descriptor)
        assert bbox == (0.0, -200.0, 1000.0, 800.0)

    def test_parse_bbox_missing(self):
        bbox = PDFFont._parse_bbox({})
        assert bbox == (0.0, 0.0, 0.0, 0.0)

    def test_parse_bbox_invalid(self):
        descriptor = {"FontBBox": "invalid"}
        bbox = PDFFont._parse_bbox(descriptor)
        assert bbox == (0.0, 0.0, 0.0, 0.0)


class TestPDFSimpleFont:
    """Tests for the PDFSimpleFont class."""

    def test_to_unichr_with_unicode_map(self):
        with patch.object(PDFSimpleFont, "__init__", lambda self, d, w, s: None):
            font = PDFSimpleFont.__new__(PDFSimpleFont)
            font.unicode_map = FileUnicodeMap()
            font.unicode_map.add_cid2unichr(65, ord("X"))
            font.cid2unicode = {}

            result = font.to_unichr(65)
            assert result == "X"

    def test_to_unichr_fallback_to_cid2unicode(self):
        with patch.object(PDFSimpleFont, "__init__", lambda self, d, w, s: None):
            font = PDFSimpleFont.__new__(PDFSimpleFont)
            font.unicode_map = None
            font.cid2unicode = {65: "A"}

            result = font.to_unichr(65)
            assert result == "A"

    def test_to_unichr_raises_unicode_not_defined(self):
        with patch.object(PDFSimpleFont, "__init__", lambda self, d, w, s: None):
            font = PDFSimpleFont.__new__(PDFSimpleFont)
            font.unicode_map = None
            font.cid2unicode = {}

            with pytest.raises(PDFUnicodeNotDefined):
                font.to_unichr(999)


class TestPDFType1Font:
    """Tests for the PDFType1Font class."""

    def test_init_with_standard_font(self):
        rsrcmgr = PDFResourceManager()
        spec = {"BaseFont": LIT("Helvetica")}
        font = PDFType1Font(rsrcmgr, spec)
        assert font.basefont == "Helvetica"
        assert len(font.widths) > 0

    def test_init_with_widths(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("CustomFont"),
            "FirstChar": 32,
            "Widths": [250, 300, 350],
            "FontDescriptor": {},
        }
        font = PDFType1Font(rsrcmgr, spec)
        assert font.basefont == "CustomFont"
        assert font.widths[32] == 250
        assert font.widths[33] == 300
        assert font.widths[34] == 350

    def test_repr(self):
        rsrcmgr = PDFResourceManager()
        spec = {"BaseFont": LIT("Helvetica")}
        font = PDFType1Font(rsrcmgr, spec)
        assert "PDFType1Font" in repr(font)
        assert "Helvetica" in repr(font)

    def test_init_without_basefont_non_strict(self):
        rsrcmgr = PDFResourceManager()
        spec: dict[str, Any] = {}
        with patch("pdfminer.pdffont.settings") as mock_settings:
            mock_settings.STRICT = False
            font = PDFType1Font(rsrcmgr, spec)
            assert font.basefont == "unknown"


class TestPDFTrueTypeFont:
    """Tests for the PDFTrueTypeFont class."""

    def test_init_with_standard_font(self):
        rsrcmgr = PDFResourceManager()
        spec = {"BaseFont": LIT("Helvetica"), "Subtype": LIT("TrueType")}
        font = PDFTrueTypeFont(rsrcmgr, spec)
        assert font.basefont == "Helvetica"

    def test_repr(self):
        rsrcmgr = PDFResourceManager()
        spec = {"BaseFont": LIT("Helvetica")}
        font = PDFTrueTypeFont(rsrcmgr, spec)
        assert "PDFTrueTypeFont" in repr(font)
        assert "Helvetica" in repr(font)


class TestPDFType3Font:
    """Tests for the PDFType3Font class."""

    def test_init_basic(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "FirstChar": 0,
            "Widths": [500, 600, 700],
            "FontBBox": [0, -200, 1000, 800],
            "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        }
        font = PDFType3Font(rsrcmgr, spec)
        assert font.widths[0] == 500
        assert font.widths[1] == 600
        assert font.widths[2] == 700
        assert font.matrix == (0.001, 0, 0, 0.001, 0, 0)

    def test_init_with_font_descriptor(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "FirstChar": 0,
            "Widths": [500],
            "FontBBox": [0, -200, 1000, 800],
            "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
            "FontDescriptor": {"FontBBox": [0, -300, 1000, 900]},
        }
        font = PDFType3Font(rsrcmgr, spec)
        assert font.ascent == 900.0
        assert font.descent == -300.0

    def test_repr(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "FontBBox": [0, 0, 1000, 1000],
            "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        }
        font = PDFType3Font(rsrcmgr, spec)
        assert repr(font) == "<PDFType3Font>"


class TestPDFCIDFont:
    """Tests for the PDFCIDFont class."""

    def test_init_horizontal(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.basefont == "TestCIDFont"
        assert font.cidcoding == "Adobe-Identity"
        assert font.is_vertical() is False
        assert font.is_multibyte() is True
        assert font.default_disp == 0

    def test_init_vertical(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-V"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.is_vertical() is True
        assert font.default_disp == (None, 880)

    def test_init_with_widths(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
            "W": [0, [500, 600, 700]],
            "DW": 1000,
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.widths[0] == 500
        assert font.widths[1] == 600
        assert font.widths[2] == 700
        assert font.default_width == 1000

    def test_decode(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        result = list(font.decode(b"\x00A\x00B"))
        assert result == [65, 66]

    def test_char_disp_horizontal(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.char_disp(65) == 0

    def test_char_disp_vertical_with_custom_disp(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-V"),
            "FontDescriptor": {},
            "W2": [65, [500, 10, 20]],
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.char_disp(65) == (10, 20)
        assert font.char_disp(999) == (None, 880)

    def test_to_unichr_with_unicode_map(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        font.unicode_map = IdentityUnicodeMap()
        assert font.to_unichr(65) == "A"

    def test_to_unichr_without_unicode_map(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        font.unicode_map = None
        with pytest.raises(PDFUnicodeNotDefined):
            font.to_unichr(65)

    def test_repr(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert "PDFCIDFont" in repr(font)
        assert "TestCIDFont" in repr(font)
        assert "Adobe-Identity" in repr(font)

    def test_init_without_basefont_non_strict(self):
        rsrcmgr = PDFResourceManager()
        spec: dict[str, Any] = {
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.basefont == "unknown"


class TestPDFCIDFontGetCmapFromSpec:
    """Tests for PDFCIDFont.get_cmap_from_spec method."""

    def test_get_cmap_identity_h(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestFont"),
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        cmap = font.get_cmap_from_spec(spec, strict=False)
        assert cmap is not None

    def test_get_cmap_dlident_h_mapped(self):
        assert IDENTITY_ENCODER["DLIdent-H"] == "Identity-H"
        assert IDENTITY_ENCODER["DLIdent-V"] == "Identity-V"


class TestPDFCIDFontGetCmapName:
    """Tests for PDFCIDFont._get_cmap_name static method."""

    def test_get_cmap_name_from_encoding(self):
        spec = {"Encoding": LIT("Identity-H")}
        name = PDFCIDFont._get_cmap_name(spec, strict=False)
        assert name == "Identity-H"

    def test_get_cmap_name_from_cmap_name(self):
        spec = {"Encoding": {"CMapName": LIT("Identity-V")}}
        name = PDFCIDFont._get_cmap_name(spec, strict=False)
        assert name == "Identity-V"

    def test_get_cmap_name_missing_non_strict(self):
        spec: dict[str, Any] = {}
        name = PDFCIDFont._get_cmap_name(spec, strict=False)
        assert name == "unknown"

    def test_get_cmap_name_missing_strict(self):
        spec: dict[str, Any] = {}
        with pytest.raises(PDFFontError) as exc_info:
            PDFCIDFont._get_cmap_name(spec, strict=True)
        assert "Encoding is unspecified" in str(exc_info.value)


class TestType1FontHeaderParser:
    """Tests for the Type1FontHeaderParser class."""

    def test_get_encoding_basic(self):
        data = (
            b"/Encoding 256 array\n"
            b"0 1 255 {1 index exch /.notdef put} for\n"
            b"dup 65 /A put\n"
            b"dup 66 /B put\n"
            b"readonly def\n"
        )
        parser = Type1FontHeaderParser(BytesIO(data))
        encoding = parser.get_encoding()
        assert 65 in encoding
        assert 66 in encoding
        assert encoding[65] == "A"
        assert encoding[66] == "B"

    def test_get_encoding_empty(self):
        data = b""
        parser = Type1FontHeaderParser(BytesIO(data))
        encoding = parser.get_encoding()
        assert encoding == {}


class TestTrueTypeFont:
    """Tests for the TrueTypeFont class."""

    def test_init_empty_data(self):
        data = b"\x00\x01\x00\x00"
        font = TrueTypeFont("test", BytesIO(data))
        assert font.name == "test"
        assert font.tables == {}

    def test_cmap_not_found_exception(self):
        data = b"\x00\x01\x00\x00"
        font = TrueTypeFont("test", BytesIO(data))
        with pytest.raises(TrueTypeFont.CMapNotFound):
            font.create_unicode_map()


class TestCFFFont:
    """Tests for the CFFFont class."""

    def test_standard_strings(self):
        assert CFFFont.STANDARD_STRINGS[0] == ".notdef"
        assert CFFFont.STANDARD_STRINGS[1] == "space"
        assert "A" in CFFFont.STANDARD_STRINGS
        assert "Z" in CFFFont.STANDARD_STRINGS
        assert "a" in CFFFont.STANDARD_STRINGS
        assert "z" in CFFFont.STANDARD_STRINGS

    def test_standard_strings_length(self):
        assert len(CFFFont.STANDARD_STRINGS) == 391


class TestPDFFontError:
    """Tests for the PDFFontError exception."""

    def test_pdffont_error(self):
        with pytest.raises(PDFFontError):
            raise PDFFontError("Test error")


class TestPDFUnicodeNotDefined:
    """Tests for the PDFUnicodeNotDefined exception."""

    def test_unicode_not_defined(self):
        with pytest.raises(PDFUnicodeNotDefined):
            raise PDFUnicodeNotDefined(None, 999)


class TestIdentityEncoder:
    """Tests for the IDENTITY_ENCODER mapping."""

    def test_dlident_h_to_identity_h(self):
        assert IDENTITY_ENCODER.get("DLIdent-H") == "Identity-H"

    def test_dlident_v_to_identity_v(self):
        assert IDENTITY_ENCODER.get("DLIdent-V") == "Identity-V"

    def test_unknown_returns_none(self):
        assert IDENTITY_ENCODER.get("Unknown") is None


class TestPDFFontCharWidthEdgeCases:
    """Edge case tests for PDFFont.char_width method."""

    def test_char_width_with_none_value(self):
        widths: dict[str | int, float] = {65: None}  # type: ignore[dict-item]
        font = MockPDFFont({"MissingWidth": 500}, widths)
        assert font.char_width(65) == 0.5

    def test_char_width_with_invalid_str_value(self):
        widths: dict[str | int, float] = {"A": "invalid"}  # type: ignore[dict-item]
        font = MockPDFFont({"MissingWidth": 500}, widths)
        assert font.char_width(65) == 0.5


class TestPDFCIDFontToUnicodeStream:
    """Tests for PDFCIDFont with ToUnicode stream."""

    def test_tounicode_stream(self):
        rsrcmgr = PDFResourceManager()
        tounicode_data = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"/CMapType 2 def\n"
            b"/CMapName /test def\n"
            b"1 begincodespacerange\n"
            b"<0000> <FFFF>\n"
            b"endcodespacerange\n"
            b"1 beginbfchar\n"
            b"<0041> <0058>\n"
            b"endbfchar\n"
            b"endcmap\n"
            b"CMapName currentdict /CMap defineresource pop\n"
            b"end\n"
            b"end\n"
        )
        tounicode_stream = PDFStream(
            {"Length": len(tounicode_data)}, tounicode_data
        )

        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
            "ToUnicode": tounicode_stream,
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.unicode_map is not None


class TestPDFCIDFontIdentityUnicodeMap:
    """Tests for PDFCIDFont with Identity in encoding/cmap name."""

    def test_identity_in_ordering(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-H"),
            "FontDescriptor": {},
            "ToUnicode": LIT("Identity"),
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert isinstance(font.unicode_map, IdentityUnicodeMap)


class TestPDFSimpleFontEncoding:
    """Tests for PDFSimpleFont encoding handling."""

    def test_standard_encoding(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("Helvetica"),
            "Encoding": LIT("StandardEncoding"),
        }
        font = PDFType1Font(rsrcmgr, spec)
        assert len(font.cid2unicode) > 0

    def test_win_ansi_encoding(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("Helvetica"),
            "Encoding": LIT("WinAnsiEncoding"),
        }
        font = PDFType1Font(rsrcmgr, spec)
        assert len(font.cid2unicode) > 0

    def test_mac_roman_encoding(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("Helvetica"),
            "Encoding": LIT("MacRomanEncoding"),
        }
        font = PDFType1Font(rsrcmgr, spec)
        assert len(font.cid2unicode) > 0

    def test_encoding_with_differences(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("Helvetica"),
            "Encoding": {
                "BaseEncoding": LIT("StandardEncoding"),
                "Differences": [65, PSLiteral("X")],
            },
        }
        font = PDFType1Font(rsrcmgr, spec)
        assert font.cid2unicode[65] == "X"


class TestPDFType3FontAscDescent:
    """Tests for PDFType3Font ascent and descent from FontBBox."""

    def test_asc_descent_from_bbox(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "FontBBox": [0, -300, 1000, 900],
            "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        }
        font = PDFType3Font(rsrcmgr, spec)
        assert font.ascent == 900.0
        assert font.descent == -300.0


class TestPDFCIDFontW2Parameter:
    """Tests for PDFCIDFont W2 (vertical width) parameter."""

    def test_w2_with_range_format(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-V"),
            "FontDescriptor": {},
            "W2": [0, 5, 500, 10, 20],
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.disps[0] == (10, 20)
        assert font.disps[5] == (10, 20)

    def test_dw2_default_values(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-V"),
            "FontDescriptor": {},
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.default_disp == (None, 880)

    def test_dw2_custom_values(self):
        rsrcmgr = PDFResourceManager()
        spec = {
            "BaseFont": LIT("TestCIDFont"),
            "CIDSystemInfo": {"Registry": b"Adobe", "Ordering": b"Identity"},
            "Encoding": LIT("Identity-V"),
            "FontDescriptor": {},
            "DW2": [500, -800],
        }
        font = PDFCIDFont(rsrcmgr, spec, strict=False)
        assert font.default_disp == (None, 500)
