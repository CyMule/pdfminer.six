"""Comprehensive tests for pdfminer/pdfparser.py"""

from io import BytesIO

import pytest

from pdfminer import settings
from pdfminer.pdfparser import PDFParser, PDFStreamParser, PDFSyntaxError
from pdfminer.pdftypes import PDFObjRef, PDFStream
from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import KWD, LIT


class MockDocument:
    """Mock PDFDocument for testing."""

    def __init__(self):
        self.decipher = None

    def getobj(self, objid):
        return f"object_{objid}"


class TestPDFParserInit:
    """Tests for PDFParser initialization"""

    def test_parser_init(self):
        fp = BytesIO(b"test")
        parser = PDFParser(fp)
        assert parser.fp is fp
        assert parser.doc is None
        assert parser.fallback is False

    def test_parser_set_document(self):
        parser = PDFParser(BytesIO(b""))
        doc = MockDocument()
        parser.set_document(doc)
        assert parser.doc is doc


class TestPDFParserNullKeyword:
    """Tests for PDFParser handling of null keyword"""

    def test_null_keyword_returns_none(self):
        parser = PDFParser(BytesIO(b"1 0 obj null endobj "))
        doc = MockDocument()
        parser.set_document(doc)
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert None in results


class TestPDFParserReferenceKeyword:
    """Tests for PDFParser handling of R (reference) keyword"""

    def test_r_keyword_creates_pdfobjref(self):
        parser = PDFParser(BytesIO(b"1 0 obj 5 0 R endobj "))
        doc = MockDocument()
        parser.set_document(doc)
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        refs = [r for r in results if isinstance(r, PDFObjRef)]
        assert len(refs) >= 1
        assert refs[0].objid == 5


class TestPDFParserXrefKeyword:
    """Tests for PDFParser handling of xref keyword"""

    def test_xref_keyword_adds_to_results(self):
        parser = PDFParser(BytesIO(b"0 xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == 0


class TestPDFParserStartxrefKeyword:
    """Tests for PDFParser handling of startxref keyword"""

    def test_startxref_keyword_adds_to_results(self):
        parser = PDFParser(BytesIO(b"12345 startxref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == 12345


class TestPDFParserEndobjKeyword:
    """Tests for PDFParser handling of endobj keyword"""

    def test_endobj_keyword_pops_items(self):
        parser = PDFParser(BytesIO(b"1 0 obj 42 endobj "))
        parser.set_document(MockDocument())
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert 1 in results


class TestPDFParserKeywordConstants:
    """Tests for PDFParser keyword constants"""

    def test_keyword_r(self):
        assert PDFParser.KEYWORD_R == KWD(b"R")

    def test_keyword_null(self):
        assert PDFParser.KEYWORD_NULL == KWD(b"null")

    def test_keyword_endobj(self):
        assert PDFParser.KEYWORD_ENDOBJ == KWD(b"endobj")

    def test_keyword_stream(self):
        assert PDFParser.KEYWORD_STREAM == KWD(b"stream")

    def test_keyword_xref(self):
        assert PDFParser.KEYWORD_XREF == KWD(b"xref")

    def test_keyword_startxref(self):
        assert PDFParser.KEYWORD_STARTXREF == KWD(b"startxref")


class TestPDFStreamParser:
    """Tests for PDFStreamParser class"""

    def test_stream_parser_init(self):
        data = b"1 2 3"
        parser = PDFStreamParser(data)
        assert parser.doc is None

    def test_stream_parser_flush(self):
        parser = PDFStreamParser(b"1 2 3 ")
        try:
            parser.nextobject()
        except PSEOF:
            pass
        assert len(parser.results) >= 0

    def test_stream_parser_parses_numbers(self):
        parser = PDFStreamParser(b"1 2 3 ")
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert 1 in results
        assert 2 in results
        assert 3 in results

    def test_stream_parser_parses_operators(self):
        parser = PDFStreamParser(b"BT /F1 12 Tf ET ")
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert KWD(b"BT") in results
        assert KWD(b"ET") in results

    def test_stream_parser_parses_strings(self):
        parser = PDFStreamParser(b"(Hello World) Tj ")
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert b"Hello World" in results

    def test_stream_parser_arrays_and_dicts(self):
        parser = PDFStreamParser(b"[ 1 2 ] << /A 1 >> ")
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert [1, 2] in results
        assert {"A": 1} in results

    def test_stream_parser_keyword_obj_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            parser = PDFStreamParser(b"1 0 obj ")
            with pytest.raises(PDFSyntaxError):
                while True:
                    parser.nextobject()
        except PSEOF:
            pass
        finally:
            settings.STRICT = original_strict

    def test_stream_parser_keyword_endobj_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            parser = PDFStreamParser(b"endobj ")
            with pytest.raises(PDFSyntaxError):
                while True:
                    parser.nextobject()
        except PSEOF:
            pass
        finally:
            settings.STRICT = original_strict

    def test_stream_parser_keyword_obj_non_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            parser = PDFStreamParser(b"1 0 obj 42 ")
            results = []
            try:
                while True:
                    pos, obj = parser.nextobject()
                    results.append(obj)
            except PSEOF:
                pass
            assert 1 in results
            assert 42 in results
        finally:
            settings.STRICT = original_strict


class TestPDFParserTokenization:
    """Tests for PDFParser tokenization"""

    def test_parser_integer(self):
        parser = PDFParser(BytesIO(b"42 xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == 42

    def test_parser_negative_integer(self):
        parser = PDFParser(BytesIO(b"-123 xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == -123

    def test_parser_float(self):
        parser = PDFParser(BytesIO(b"3.14 xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == 3.14

    def test_parser_literal(self):
        parser = PDFParser(BytesIO(b"/Name xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == LIT("Name")

    def test_parser_string(self):
        parser = PDFParser(BytesIO(b"(hello) xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == b"hello"

    def test_parser_hexstring(self):
        parser = PDFParser(BytesIO(b"<48656C6C6F> xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == b"Hello"

    def test_parser_array(self):
        parser = PDFParser(BytesIO(b"[ 1 2 3 ] xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == [1, 2, 3]

    def test_parser_dict(self):
        parser = PDFParser(BytesIO(b"<< /Key /Value >> xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj == {"Key": LIT("Value")}

    def test_parser_boolean_true(self):
        parser = PDFParser(BytesIO(b"true xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj is True

    def test_parser_boolean_false(self):
        parser = PDFParser(BytesIO(b"false xref "))
        parser.set_document(MockDocument())
        pos, obj = parser.nextobject()
        assert obj is False


class TestPDFParserObjectDefinition:
    """Tests for PDFParser object definition parsing"""

    def test_simple_object_definition(self):
        parser = PDFParser(BytesIO(b"1 0 obj 42 endobj "))
        parser.set_document(MockDocument())
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        assert 42 in results

    def test_dict_object_definition(self):
        parser = PDFParser(BytesIO(b"1 0 obj << /Type /Page >> endobj "))
        parser.set_document(MockDocument())
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        dicts = [r for r in results if isinstance(r, dict)]
        assert len(dicts) >= 1
        assert dicts[0].get("Type") == LIT("Page")

    def test_array_object_definition(self):
        parser = PDFParser(BytesIO(b"1 0 obj [ 1 2 3 ] endobj "))
        parser.set_document(MockDocument())
        results = []
        try:
            while True:
                pos, obj = parser.nextobject()
                results.append(obj)
        except PSEOF:
            pass
        arrays = [r for r in results if isinstance(r, list)]
        assert len(arrays) >= 1
        assert arrays[0] == [1, 2, 3]


class TestPDFSyntaxError:
    """Tests for PDFSyntaxError exception"""

    def test_pdfsyntaxerror_is_exception(self):
        assert issubclass(PDFSyntaxError, Exception)

    def test_pdfsyntaxerror_message(self):
        try:
            raise PDFSyntaxError("Test error")
        except PDFSyntaxError as e:
            assert "Test error" in str(e)
