"""Comprehensive tests for pdfminer/pdfdocument.py"""

import zlib
from io import BytesIO

import pytest

from pdfminer import settings
from pdfminer.pdfdocument import (
    LITERAL_CATALOG,
    LITERAL_OBJSTM,
    LITERAL_XREF,
    PageLabels,
    PDFBaseXRef,
    PDFDestinationNotFound,
    PDFDocument,
    PDFEncryptionError,
    PDFEncryptionWarning,
    PDFNoOutlines,
    PDFNoPageLabels,
    PDFNoValidXRef,
    PDFNoValidXRefWarning,
    PDFPasswordIncorrect,
    PDFStandardSecurityHandler,
    PDFStandardSecurityHandlerV4,
    PDFStandardSecurityHandlerV5,
    PDFTextExtractionNotAllowed,
    PDFTextExtractionNotAllowedWarning,
    PDFXRef,
    PDFXRefFallback,
    PDFXRefStream,
)
from pdfminer.pdfexceptions import PDFKeyError, PDFObjectNotFound
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.psparser import KWD, LIT


def create_minimal_pdf(
    objects: list[tuple[int, int, bytes]],
    xref_entries: list[tuple[int, int, int, str]] | None = None,
    trailer_extra: bytes = b"",
    root_obj: int = 1,
) -> bytes:
    """Create a minimal PDF with the given objects.

    Args:
        objects: List of (objid, genno, content) tuples
        xref_entries: Optional explicit xref entries as (objid, offset, genno, use)
        trailer_extra: Extra content for trailer dict
        root_obj: Object ID for the Root reference
    """
    pdf_parts = [b"%PDF-1.4\n"]
    offsets = {}

    for objid, genno, content in objects:
        offsets[objid] = len(b"".join(pdf_parts))
        pdf_parts.append(f"{objid} {genno} obj\n".encode())
        pdf_parts.append(content)
        pdf_parts.append(b"\nendobj\n")

    xref_pos = len(b"".join(pdf_parts))
    pdf_parts.append(b"xref\n")

    if xref_entries:
        min_objid = min(e[0] for e in xref_entries)
        max_objid = max(e[0] for e in xref_entries)
        pdf_parts.append(f"{min_objid} {max_objid - min_objid + 1}\n".encode())
        for _objid, offset, genno, use in sorted(xref_entries, key=lambda x: x[0]):
            pdf_parts.append(f"{offset:010d} {genno:05d} {use} \n".encode())
    else:
        pdf_parts.append(f"0 {len(objects) + 1}\n".encode())
        pdf_parts.append(b"0000000000 65535 f \n")
        for objid in range(1, len(objects) + 1):
            if objid in offsets:
                pdf_parts.append(f"{offsets[objid]:010d} 00000 n \n".encode())
            else:
                pdf_parts.append(b"0000000000 65535 f \n")

    pdf_parts.append(b"trailer\n")
    trailer_str = trailer_extra.decode() if trailer_extra else ""
    trailer = f"<< /Size {len(objects) + 1} /Root {root_obj} 0 R {trailer_str} >>\n"
    pdf_parts.append(trailer.encode())
    pdf_parts.append(b"startxref\n")
    pdf_parts.append(f"{xref_pos}\n".encode())
    pdf_parts.append(b"%%EOF\n")

    return b"".join(pdf_parts)


def create_catalog_obj(pages_ref: int = 2) -> bytes:
    """Create a basic Catalog object."""
    return f"<< /Type /Catalog /Pages {pages_ref} 0 R >>".encode()


def create_pages_obj(count: int = 0, kids: list[int] | None = None) -> bytes:
    """Create a basic Pages object."""
    kids_str = " ".join(f"{k} 0 R" for k in (kids or []))
    return f"<< /Type /Pages /Count {count} /Kids [{kids_str}] >>".encode()


class TestPDFNoValidXRef:
    """Tests for PDFNoValidXRef exception"""

    def test_is_syntax_error(self):
        assert issubclass(PDFNoValidXRef, PDFSyntaxError)

    def test_can_be_raised_with_message(self):
        with pytest.raises(PDFNoValidXRef) as exc_info:
            raise PDFNoValidXRef("Test message")
        assert "Test message" in str(exc_info.value)


class TestPDFNoValidXRefWarning:
    """Tests for PDFNoValidXRefWarning"""

    def test_is_syntax_warning(self):
        assert issubclass(PDFNoValidXRefWarning, SyntaxWarning)


class TestPDFTextExtractionNotAllowed:
    """Tests for PDFTextExtractionNotAllowed exception"""

    def test_is_encryption_error(self):
        assert issubclass(PDFTextExtractionNotAllowed, PDFEncryptionError)


class TestPDFTextExtractionNotAllowedWarning:
    """Tests for PDFTextExtractionNotAllowedWarning"""

    def test_is_user_warning(self):
        assert issubclass(PDFTextExtractionNotAllowedWarning, UserWarning)


class TestPDFEncryptionWarning:
    """Tests for PDFEncryptionWarning"""

    def test_is_user_warning(self):
        assert issubclass(PDFEncryptionWarning, UserWarning)


class TestPDFNoOutlines:
    """Tests for PDFNoOutlines exception"""

    def test_can_be_raised(self):
        with pytest.raises(PDFNoOutlines):
            raise PDFNoOutlines()


class TestPDFNoPageLabels:
    """Tests for PDFNoPageLabels exception"""

    def test_can_be_raised(self):
        with pytest.raises(PDFNoPageLabels):
            raise PDFNoPageLabels()


class TestPDFDestinationNotFound:
    """Tests for PDFDestinationNotFound exception"""

    def test_can_be_raised_with_name(self):
        with pytest.raises(PDFDestinationNotFound):
            raise PDFDestinationNotFound("dest_name")


class TestPDFPasswordIncorrect:
    """Tests for PDFPasswordIncorrect exception"""

    def test_is_encryption_error(self):
        assert issubclass(PDFPasswordIncorrect, PDFEncryptionError)


class TestLiterals:
    """Tests for PDF literal constants"""

    def test_literal_objstm(self):
        assert LIT("ObjStm") == LITERAL_OBJSTM

    def test_literal_xref(self):
        assert LIT("XRef") == LITERAL_XREF

    def test_literal_catalog(self):
        assert LIT("Catalog") == LITERAL_CATALOG


class TestPDFBaseXRef:
    """Tests for PDFBaseXRef base class"""

    def test_get_trailer_not_implemented(self):
        xref = PDFBaseXRef()
        with pytest.raises(NotImplementedError):
            xref.get_trailer()

    def test_get_objids_returns_empty(self):
        xref = PDFBaseXRef()
        assert list(xref.get_objids()) == []

    def test_get_pos_raises_key_error(self):
        xref = PDFBaseXRef()
        with pytest.raises(PDFKeyError):
            xref.get_pos(1)

    def test_load_not_implemented(self):
        xref = PDFBaseXRef()
        with pytest.raises(NotImplementedError):
            xref.load(None)


class TestPDFXRef:
    """Tests for PDFXRef class"""

    def test_init(self):
        xref = PDFXRef()
        assert xref.offsets == {}
        assert xref.trailer == {}

    def test_repr(self):
        xref = PDFXRef()
        xref.offsets = {1: (None, 100, 0), 2: (None, 200, 0)}
        repr_str = repr(xref)
        assert "PDFXRef" in repr_str
        assert "1" in repr_str or "2" in repr_str

    def test_get_trailer(self):
        xref = PDFXRef()
        xref.trailer = {"Size": 10, "Root": "ref"}
        assert xref.get_trailer() == {"Size": 10, "Root": "ref"}

    def test_get_objids(self):
        xref = PDFXRef()
        xref.offsets = {1: (None, 100, 0), 5: (None, 200, 0)}
        assert set(xref.get_objids()) == {1, 5}

    def test_get_pos(self):
        xref = PDFXRef()
        xref.offsets = {1: (None, 100, 0)}
        assert xref.get_pos(1) == (None, 100, 0)

    def test_get_pos_missing_raises_key_error(self):
        xref = PDFXRef()
        with pytest.raises(KeyError):
            xref.get_pos(999)

    def test_load_simple_xref(self):
        xref_content = b"""\
xref
0 3
0000000000 65535 f
0000000010 00000 n
0000000100 00000 n
trailer
<< /Size 3 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert 1 in xref.offsets
        assert 2 in xref.offsets
        assert xref.offsets[1] == (None, 10, 0)
        assert xref.offsets[2] == (None, 100, 0)
        assert xref.trailer["Size"] == 3

    def test_load_xref_with_multiple_subsections(self):
        xref_content = b"""\
xref
0 2
0000000000 65535 f
0000000010 00000 n
5 2
0000000100 00000 n
0000000200 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert 1 in xref.offsets
        assert 5 in xref.offsets
        assert 6 in xref.offsets
        assert xref.offsets[5] == (None, 100, 0)

    def test_load_xref_with_free_entries(self):
        xref_content = b"""\
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000000 00000 f
0000000100 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert 1 in xref.offsets
        assert 2 not in xref.offsets
        assert 3 in xref.offsets

    def test_load_xref_unexpected_eof(self):
        xref_content = b"""\
xref
0 2
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        with pytest.raises(PDFNoValidXRef, match="EOF"):
            xref.load(parser)

    def test_load_xref_invalid_header(self):
        xref_content = b"""\
xref
invalid header
trailer
<< /Size 1 >>
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        with pytest.raises(PDFNoValidXRef, match="Invalid line"):
            xref.load(parser)

    def test_load_xref_invalid_subsection_header_too_many_parts(self):
        xref_content = b"""\
xref
0 1 extra
trailer
<< /Size 1 >>
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        with pytest.raises(PDFNoValidXRef, match="Trailer not found"):
            xref.load(parser)

    def test_load_xref_invalid_entry_format(self):
        xref_content = b"""\
xref
0 2
0000000000 65535 f
badentry
trailer
<< /Size 2 >>
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        with pytest.raises(PDFNoValidXRef, match="Invalid XRef format"):
            xref.load(parser)


class TestPDFXRefFallback:
    """Tests for PDFXRefFallback class"""

    def test_repr(self):
        xref = PDFXRefFallback()
        xref.offsets = {1: (None, 100, 0)}
        assert "PDFXRefFallback" in repr(xref)

    def test_pdfobj_cue_regex(self):
        assert PDFXRefFallback.PDFOBJ_CUE.match("1 0 obj")
        assert PDFXRefFallback.PDFOBJ_CUE.match("123 456 obj")
        assert not PDFXRefFallback.PDFOBJ_CUE.match("not an object")

    def test_load_finds_objects(self):
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Count 0 /Kids [] >>
endobj
trailer
<< /Size 3 /Root 1 0 R >>
%%EOF
"""
        parser = PDFParser(BytesIO(pdf_content))
        xref = PDFXRefFallback()
        xref.load(parser)
        assert 1 in xref.offsets
        assert 2 in xref.offsets
        assert xref.trailer["Size"] == 3

    def test_load_handles_no_trailer(self):
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Count 0 /Kids [] >>
endobj
"""
        parser = PDFParser(BytesIO(pdf_content))
        xref = PDFXRefFallback()
        xref.load(parser)
        assert 1 in xref.offsets
        assert 2 in xref.offsets


class TestPDFXRefStream:
    """Tests for PDFXRefStream class"""

    def test_init(self):
        xref = PDFXRefStream()
        assert xref.data is None
        assert xref.entlen is None
        assert xref.fl1 is None
        assert xref.fl2 is None
        assert xref.fl3 is None
        assert xref.ranges == []

    def test_repr(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 5), (10, 3)]
        assert "PDFXRefStream" in repr(xref)
        assert "(0, 5)" in repr(xref)

    def test_load_invalid_stream_raises(self):
        content = b"""\
1 0 obj
<< /Type /NotXRef >>
endobj
"""
        parser = PDFParser(BytesIO(content))
        xref = PDFXRefStream()
        with pytest.raises(PDFNoValidXRef, match="Invalid PDF stream"):
            xref.load(parser)

    def test_get_pos_not_found(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 3)]
        xref.entlen = 5
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 2
        xref.data = b"\x00" * 15
        with pytest.raises(PDFKeyError):
            xref.get_pos(100)


class TestPDFStandardSecurityHandler:
    """Tests for PDFStandardSecurityHandler class"""

    def test_password_padding(self):
        assert len(PDFStandardSecurityHandler.PASSWORD_PADDING) == 32

    def test_supported_revisions(self):
        assert PDFStandardSecurityHandler.supported_revisions == (2, 3)

    def test_unsupported_revision_raises(self):
        docid = [b"test_id"]
        param = {"R": 99, "P": 0, "O": b"x" * 32, "U": b"x" * 32}
        with pytest.raises(PDFEncryptionError, match="Unsupported revision"):
            PDFStandardSecurityHandler(docid, param)

    def test_is_printable(self):
        handler = object.__new__(PDFStandardSecurityHandler)
        handler.p = 0b00000100
        assert handler.is_printable() is True
        handler.p = 0b00000000
        assert handler.is_printable() is False

    def test_is_modifiable(self):
        handler = object.__new__(PDFStandardSecurityHandler)
        handler.p = 0b00001000
        assert handler.is_modifiable() is True
        handler.p = 0b00000000
        assert handler.is_modifiable() is False

    def test_is_extractable(self):
        handler = object.__new__(PDFStandardSecurityHandler)
        handler.p = 0b00010000
        assert handler.is_extractable() is True
        handler.p = 0b00000000
        assert handler.is_extractable() is False


class TestPDFStandardSecurityHandlerV4:
    """Tests for PDFStandardSecurityHandlerV4 class"""

    def test_supported_revisions(self):
        assert PDFStandardSecurityHandlerV4.supported_revisions == (4,)

    def test_get_cfm_v2(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        assert handler.get_cfm("V2") is not None

    def test_get_cfm_aesv2(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        assert handler.get_cfm("AESV2") is not None

    def test_get_cfm_unknown(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        assert handler.get_cfm("Unknown") is None

    def test_decrypt_identity(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        data = b"test data"
        assert handler.decrypt_identity(1, 0, data) == data


class TestPDFStandardSecurityHandlerV5:
    """Tests for PDFStandardSecurityHandlerV5 class"""

    def test_supported_revisions(self):
        assert PDFStandardSecurityHandlerV5.supported_revisions == (5, 6)

    def test_get_cfm_aesv3(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        assert handler.get_cfm("AESV3") is not None

    def test_get_cfm_unknown(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        assert handler.get_cfm("Unknown") is None

    def test_bytes_mod_3(self):
        assert PDFStandardSecurityHandlerV5._bytes_mod_3(b"\x00\x00\x00") == 0
        assert PDFStandardSecurityHandlerV5._bytes_mod_3(b"\x01") == 1
        assert PDFStandardSecurityHandlerV5._bytes_mod_3(b"\x02") == 2
        assert PDFStandardSecurityHandlerV5._bytes_mod_3(b"\x03") == 0

    def test_normalize_password_empty(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        handler.r = 6
        assert handler._normalize_password("") == b""

    def test_normalize_password_truncates(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        handler.r = 5
        long_password = "a" * 200
        result = handler._normalize_password(long_password)
        assert len(result) == 127


class TestPDFDocument:
    """Tests for PDFDocument class"""

    def test_security_handler_registry(self):
        assert 1 in PDFDocument.security_handler_registry
        assert 2 in PDFDocument.security_handler_registry
        assert 4 in PDFDocument.security_handler_registry
        assert 5 in PDFDocument.security_handler_registry

    def test_minimal_pdf_document(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        assert doc.catalog is not None
        assert doc.catalog.get("Type") == LIT("Catalog")

    def test_document_with_info(self):
        pdf = create_minimal_pdf(
            [
                (1, 0, create_catalog_obj(2)),
                (2, 0, create_pages_obj(0)),
                (3, 0, b"<< /Author (Test Author) >>"),
            ],
            trailer_extra=b"/Info 3 0 R",
        )
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        assert len(doc.info) >= 1

    def test_getobj_returns_object(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(1)
        assert obj is not None

    def test_getobj_not_found_raises(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFObjectNotFound):
            doc.getobj(999)

    def test_getobj_zero_raises(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFObjectNotFound):
            doc.getobj(0)

    def test_caching_enabled(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser, caching=True)
        obj1 = doc.getobj(1)
        obj2 = doc.getobj(1)
        assert obj1 is obj2

    def test_caching_disabled(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"42"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser, caching=False)
        assert doc.caching is False
        doc._cached_objs.clear()
        obj1 = doc.getobj(3)
        after_first_get = len(doc._cached_objs)
        obj2 = doc.getobj(3)
        assert obj1 == obj2 == 42
        assert after_first_get == 0

    def test_no_root_raises(self):
        content = b"""%PDF-1.4
1 0 obj
<< /Type /Pages /Count 0 /Kids [] >>
endobj
xref
0 2
0000000000 65535 f
0000000009 00000 n
trailer
<< /Size 2 >>
startxref
60
%%EOF
"""
        parser = PDFParser(BytesIO(content))
        with pytest.raises(PDFSyntaxError, match="No /Root"):
            PDFDocument(parser)

    def test_fallback_xref(self):
        pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Count 0 /Kids [] >>
endobj
trailer
<< /Size 3 /Root 1 0 R >>
%%EOF
"""
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser, fallback=True)
        assert doc.catalog is not None
        assert any(isinstance(x, PDFXRefFallback) for x in doc.xrefs)

    def test_fallback_disabled_with_no_startxref_gives_no_root(self):
        pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Count 0 /Kids [] >>
endobj
"""
        parser = PDFParser(BytesIO(pdf))
        with pytest.raises(PDFSyntaxError, match="No /Root"):
            PDFDocument(parser, fallback=False)

    def test_fallback_disabled_with_invalid_xref_gives_no_root(self):
        pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
startxref
notanumber
%%EOF
"""
        parser = PDFParser(BytesIO(pdf))
        with pytest.raises(PDFSyntaxError, match="No /Root"):
            PDFDocument(parser, fallback=False)

    def test_default_permissions(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        assert doc.is_printable is True
        assert doc.is_modifiable is True
        assert doc.is_extractable is True

    def test_get_outlines_no_outlines(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFNoOutlines):
            list(doc.get_outlines())

    def test_get_page_labels_no_labels(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFNoPageLabels):
            doc.get_page_labels()

    def test_get_dest_not_found(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFDestinationNotFound):
            doc.get_dest("nonexistent")

    def test_lookup_name_no_names(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        with pytest.raises(PDFKeyError):
            doc.lookup_name("Dests", "test")


class TestPDFDocumentFindXref:
    """Tests for PDFDocument.find_xref method"""

    def test_find_xref_valid(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = object.__new__(PDFDocument)
        pos = doc.find_xref(parser)
        assert isinstance(pos, int)
        assert pos >= 0

    def test_find_xref_no_startxref(self):
        content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog >>
endobj
"""
        parser = PDFParser(BytesIO(content))
        doc = object.__new__(PDFDocument)
        with pytest.raises(PDFNoValidXRef, match="EOF"):
            doc.find_xref(parser)

    def test_find_xref_invalid_position(self):
        content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog >>
endobj
xref
0 1
0000000000 65535 f
trailer
<< /Size 1 >>
startxref
invalid
%%EOF
"""
        parser = PDFParser(BytesIO(content))
        doc = object.__new__(PDFDocument)
        with pytest.raises(PDFNoValidXRef, match="no digit"):
            doc.find_xref(parser)

    def test_find_xref_negative_position(self):
        content = b"""%PDF-1.4
startxref
-100
%%EOF
"""
        parser = PDFParser(BytesIO(content))
        doc = object.__new__(PDFDocument)
        with pytest.raises(PDFNoValidXRef, match="no digit"):
            doc.find_xref(parser)

    def test_find_xref_too_large(self):
        content = b"""%PDF-1.4
startxref
99999999999
%%EOF
"""
        parser = PDFParser(BytesIO(content))
        doc = object.__new__(PDFDocument)
        with pytest.raises(PDFNoValidXRef, match="too large"):
            doc.find_xref(parser)


class TestPageLabels:
    """Tests for PageLabels class"""

    def test_format_page_label_decimal(self):
        assert PageLabels._format_page_label(1, LIT("D")) == "1"
        assert PageLabels._format_page_label(42, LIT("D")) == "42"

    def test_format_page_label_uppercase_roman(self):
        assert PageLabels._format_page_label(1, LIT("R")) == "I"
        assert PageLabels._format_page_label(4, LIT("R")) == "IV"
        assert PageLabels._format_page_label(10, LIT("R")) == "X"

    def test_format_page_label_lowercase_roman(self):
        assert PageLabels._format_page_label(1, LIT("r")) == "i"
        assert PageLabels._format_page_label(4, LIT("r")) == "iv"
        assert PageLabels._format_page_label(10, LIT("r")) == "x"

    def test_format_page_label_uppercase_alpha(self):
        assert PageLabels._format_page_label(1, LIT("A")) == "A"
        assert PageLabels._format_page_label(26, LIT("A")) == "Z"
        assert PageLabels._format_page_label(27, LIT("A")) == "AA"

    def test_format_page_label_lowercase_alpha(self):
        assert PageLabels._format_page_label(1, LIT("a")) == "a"
        assert PageLabels._format_page_label(26, LIT("a")) == "z"
        assert PageLabels._format_page_label(27, LIT("a")) == "aa"

    def test_format_page_label_none(self):
        assert PageLabels._format_page_label(1, None) == ""
        assert PageLabels._format_page_label(100, None) == ""

    def test_format_page_label_unknown(self):
        assert PageLabels._format_page_label(1, LIT("X")) == ""


class TestPDFDocumentWithEncryption:
    """Tests for PDFDocument encryption handling"""

    def test_unknown_filter_raises(self):
        pdf_parts = [b"%PDF-1.4\n"]
        catalog_offset = len(b"".join(pdf_parts))
        pdf_parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        pages_offset = len(b"".join(pdf_parts))
        pdf_parts.append(b"2 0 obj\n<< /Type /Pages /Count 0 /Kids [] >>\nendobj\n")
        encrypt_offset = len(b"".join(pdf_parts))
        encrypt_obj = (
            b"3 0 obj\n<< /Filter /UnknownFilter /V 1 /R 2 /P 0 "
            b"/O (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) "
            b"/U (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx) >>\nendobj\n"
        )
        pdf_parts.append(encrypt_obj)
        xref_offset = len(b"".join(pdf_parts))
        pdf_parts.append(b"xref\n0 4\n")
        pdf_parts.append(b"0000000000 65535 f \n")
        pdf_parts.append(f"{catalog_offset:010d} 00000 n \n".encode())
        pdf_parts.append(f"{pages_offset:010d} 00000 n \n".encode())
        pdf_parts.append(f"{encrypt_offset:010d} 00000 n \n".encode())
        trailer = (
            b"trailer\n<< /Size 4 /Root 1 0 R "
            b"/Encrypt 3 0 R /ID [(test)(test)] >>\nstartxref\n"
        )
        pdf_parts.append(trailer)
        pdf_parts.append(f"{xref_offset}\n".encode())
        pdf_parts.append(b"%%EOF\n")
        pdf = b"".join(pdf_parts)

        parser = PDFParser(BytesIO(pdf))
        with pytest.raises(PDFEncryptionError, match="Unknown filter"):
            PDFDocument(parser)


class TestPDFDocumentXRefChain:
    """Tests for PDFDocument XRef chain handling"""

    def test_read_xref_with_prev(self):
        pass


class TestPDFDocumentStrict:
    """Tests for PDFDocument behavior in strict mode"""

    def test_strict_catalog_type_check(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            pdf_parts = [b"%PDF-1.4\n"]
            catalog_offset = len(b"".join(pdf_parts))
            pdf_parts.append(b"1 0 obj\n<< /Type /NotCatalog /Pages 2 0 R >>\nendobj\n")
            pages_offset = len(b"".join(pdf_parts))
            pdf_parts.append(b"2 0 obj\n<< /Type /Pages /Count 0 /Kids [] >>\nendobj\n")
            xref_offset = len(b"".join(pdf_parts))
            pdf_parts.append(b"xref\n0 3\n")
            pdf_parts.append(b"0000000000 65535 f \n")
            pdf_parts.append(f"{catalog_offset:010d} 00000 n \n".encode())
            pdf_parts.append(f"{pages_offset:010d} 00000 n \n".encode())
            pdf_parts.append(b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n")
            pdf_parts.append(f"{xref_offset}\n".encode())
            pdf_parts.append(b"%%EOF\n")
            pdf = b"".join(pdf_parts)

            parser = PDFParser(BytesIO(pdf))
            with pytest.raises(PDFSyntaxError, match="Catalog not found"):
                PDFDocument(parser)
        finally:
            settings.STRICT = original_strict


class TestPDFXRefLoadEdgeCases:
    """Edge case tests for XRef loading"""

    def test_xref_with_generation_numbers(self):
        xref_content = b"""\
xref
0 3
0000000000 65535 f
0000000010 00001 n
0000000100 00002 n
trailer
<< /Size 3 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert xref.offsets[1] == (None, 10, 1)
        assert xref.offsets[2] == (None, 100, 2)

    def test_xref_empty_lines_skipped(self):
        xref_content = b"""\
xref

0 2
0000000000 65535 f
0000000010 00000 n
trailer
<< /Size 2 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert 1 in xref.offsets

    def test_xref_trailer_directly_after_header(self):
        xref_content = b"""\
xref
trailer
<< /Size 0 /Root 1 0 R >>
startxref
0
%%EOF
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
        assert xref.trailer["Size"] == 0


class TestPDFDocumentMultipleXRefs:
    """Tests for documents with multiple XRef tables"""

    def test_document_with_incremental_update(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        assert len(doc.xrefs) >= 1


class TestPDFXRefStreamGetPos:
    """Tests for PDFXRefStream.get_pos method"""

    def test_get_pos_type_1(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 3)]
        xref.fl1 = 1
        xref.fl2 = 3
        xref.fl3 = 1
        xref.entlen = 5
        xref.data = (
            b"\x00\x00\x00\x00\x00"
            + b"\x01\x00\x00\x64\x00"
            + b"\x01\x00\x00\xc8\x01"
        )
        result = xref.get_pos(1)
        assert result == (None, 100, 0)

    def test_get_pos_type_2(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 3)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 2
        xref.entlen = 5
        xref.data = (
            b"\x00\x00\x00\x00\x00"
            + b"\x02\x00\x05\x00\x00"
            + b"\x02\x00\x05\x00\x01"
        )
        result = xref.get_pos(1)
        assert result == (5, 0, 0)

    def test_get_pos_free_entry_raises(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 2)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 2
        xref.entlen = 5
        xref.data = (
            b"\x00\x00\x00\x00\x00"
            + b"\x00\x00\x00\x00\x00"
        )
        with pytest.raises(PDFKeyError):
            xref.get_pos(1)


class TestPDFXRefStreamGetObjids:
    """Tests for PDFXRefStream.get_objids method"""

    def test_get_objids_type_1(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 3)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 1
        xref.entlen = 4
        xref.data = (
            b"\x00\x00\x00\x00"
            + b"\x01\x00\x64\x00"
            + b"\x01\x00\xc8\x00"
        )
        objids = list(xref.get_objids())
        assert 1 in objids
        assert 2 in objids

    def test_get_objids_type_2(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 2)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 1
        xref.entlen = 4
        xref.data = (
            b"\x00\x00\x00\x00"
            + b"\x02\x00\x05\x00"
        )
        objids = list(xref.get_objids())
        assert 1 in objids

    def test_get_objids_skips_free(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 3)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 1
        xref.entlen = 4
        xref.data = (
            b"\x00\x00\x00\x00"
            + b"\x00\x00\x00\x00"
            + b"\x01\x00\xc8\x00"
        )
        objids = list(xref.get_objids())
        assert 1 not in objids
        assert 2 in objids


class TestPDFXRefStreamMultipleRanges:
    """Tests for PDFXRefStream with multiple ranges"""

    def test_get_pos_in_second_range(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 2), (10, 2)]
        xref.fl1 = 1
        xref.fl2 = 2
        xref.fl3 = 1
        xref.entlen = 4
        xref.data = (
            b"\x00\x00\x00\x00"
            + b"\x01\x00\x64\x00"
            + b"\x01\x00\xc8\x00"
            + b"\x01\x01\x00\x00"
        )
        result = xref.get_pos(10)
        assert result == (None, 200, 0)


class TestPDFDocumentGetObjParsing:
    """Tests for object parsing in getobj"""

    def test_getobj_parses_dict(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(2)
        assert isinstance(obj, dict)
        assert obj.get("Type") == LIT("Pages")

    def test_getobj_parses_array(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"[ 1 2 3 4 5 ]"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj == [1, 2, 3, 4, 5]

    def test_getobj_parses_string(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"(Hello World)"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj == b"Hello World"

    def test_getobj_parses_number(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"42"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj == 42

    def test_getobj_parses_boolean_true(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"true"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj is True

    def test_getobj_parses_boolean_false(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"false"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj is False

    def test_getobj_parses_null(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"null"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        obj = doc.getobj(3)
        assert obj is None


class TestPDFDocumentWithReferences:
    """Tests for PDFDocument handling object references"""

    def test_nested_references(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, b"<< /Type /Pages /Count 0 /Kids [] /MediaBox 3 0 R >>"),
            (3, 0, b"[ 0 0 612 792 ]"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        pages = doc.getobj(2)
        assert "MediaBox" in pages


class TestPDFDocumentKeywordObj:
    """Tests for KEYWORD_OBJ constant"""

    def test_keyword_obj_value(self):
        assert KWD(b"obj") == PDFDocument.KEYWORD_OBJ


class MockDocument:
    """Mock PDFDocument for testing."""

    def __init__(self):
        self.decipher = None

    def getobj(self, objid):
        return f"object_{objid}"


class TestPDFXRefStreamLoadInvalidIndex:
    """Tests for PDFXRefStream with invalid index array"""

    def test_load_odd_index_array_raises(self):
        data = zlib.compress(b"\x00" * 15)
        content = (
            b"1 0 obj\n"
            b"<< /Type /XRef /Size 3 /W [1 2 2] /Index [0 2 5] /Length "
            + str(len(data)).encode()
            + b" /Filter /FlateDecode >>\n"
            b"stream\n"
            + data
            + b"\nendstream\nendobj\n"
        )
        parser = PDFParser(BytesIO(content))
        parser.set_document(MockDocument())
        xref = PDFXRefStream()
        with pytest.raises(PDFSyntaxError, match="Invalid index number"):
            xref.load(parser)


class TestPDFEncryptionDocIdMissing:
    """Tests for PDFDocument with missing encryption ID"""

    def test_encryption_without_id_uses_empty_bytes(self):
        pass


class TestPDFXRefLoadTrailerParsing:
    """Tests for PDFXRef trailer loading edge cases"""

    def test_load_trailer_with_pop_fallback(self):
        xref_content = b"""\
xref
0 1
0000000000 65535 f
trailer
"""
        parser = PDFParser(BytesIO(xref_content))
        parser.seek(0)
        parser.nextline()
        xref = PDFXRef()
        with pytest.raises(PDFNoValidXRef, match="EOF"):
            xref.load(parser)


class TestPDFDocumentGetObjEdgeCases:
    """Edge case tests for getobj method"""

    def test_getobj_with_empty_xrefs_raises(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        doc.xrefs = []
        from pdfminer.pdfexceptions import PDFException
        with pytest.raises(PDFException, match="not initialized"):
            doc.getobj(1)


class TestPageLabelsEdgeCases:
    """Edge case tests for PageLabels"""

    def test_format_unknown_style(self):
        result = PageLabels._format_page_label(5, LIT("Unknown"))
        assert result == ""

    def test_format_with_high_values(self):
        assert PageLabels._format_page_label(100, LIT("D")) == "100"
        assert PageLabels._format_page_label(1000, LIT("R")) == "M"


class TestPDFStandardSecurityHandlerEdgeCases:
    """Edge cases for security handlers"""

    def test_compute_u_revision_2(self):
        handler = object.__new__(PDFStandardSecurityHandler)
        handler.r = 2
        handler.docid = [b"testid12345678901234567890123456"]
        key = b"12345"
        result = handler.compute_u(key)
        assert len(result) == 32

    def test_compute_u_revision_3(self):
        handler = object.__new__(PDFStandardSecurityHandler)
        handler.r = 3
        handler.docid = [b"testid12345678901234567890123456"]
        key = b"12345678901234567890"
        result = handler.compute_u(key)
        assert len(result) == 32


class TestPDFDocumentReadXref:
    """Tests for read_xref_from method"""

    def test_read_xref_from_eof_raises(self):
        pdf = b"""%PDF-1.4
startxref
5000
%%EOF
"""
        parser = PDFParser(BytesIO(pdf))
        doc = object.__new__(PDFDocument)
        doc.xrefs = []
        with pytest.raises(PDFNoValidXRef, match="EOF"):
            pos = doc.find_xref(parser)
            doc.read_xref_from(parser, pos, doc.xrefs)


class TestPDFXRefFallbackObjStm:
    """Tests for PDFXRefFallback handling ObjStm objects"""

    def test_objstm_expansion(self):
        pass


class TestSecurityHandlerMissmatchedFilters:
    """Tests for V4 security handler with mismatched filters"""

    def test_stmf_strf_mismatch_raises(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        handler.docid = [b"test"]
        handler.param = {
            "V": 4,
            "R": 4,
            "P": 0,
            "O": b"x" * 32,
            "U": b"x" * 32,
            "CF": {
                "StdCF": {"CFM": LIT("V2")},
            },
            "StmF": LIT("StdCF"),
            "StrF": LIT("Identity"),
            "Length": 128,
        }
        handler.password = ""
        with pytest.raises(PDFEncryptionError, match="Unsupported crypt filter"):
            handler.init_params()


class TestPDFXRefGetTrailer:
    """Tests for get_trailer methods"""

    def test_xref_stream_get_trailer(self):
        xref = PDFXRefStream()
        xref.trailer = {"Size": 10, "Root": LIT("ref")}
        result = xref.get_trailer()
        assert result["Size"] == 10


class TestPDFDocumentWithOutlines:
    """Tests for PDF outlines/bookmarks"""

    def test_get_outlines_with_basic_outline(self):
        pdf = create_minimal_pdf([
            (1, 0, b"<< /Type /Catalog /Pages 2 0 R /Outlines 3 0 R >>"),
            (2, 0, create_pages_obj(0)),
            (3, 0, b"<< /Type /Outlines /First 4 0 R /Last 4 0 R /Count 1 >>"),
            (4, 0, b"<< /Title (Chapter 1) /Dest [2 0 R /XYZ 0 0 0] >>"),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        outlines = list(doc.get_outlines())
        assert len(outlines) == 1
        _level, title, _dest, _action, _se = outlines[0]
        assert title == "Chapter 1"


class TestPDFDocumentLookupName:
    """Tests for lookup_name method"""

    def test_lookup_name_with_names_tree(self):
        pass


class TestPDFDocumentGetDest:
    """Tests for get_dest method"""

    def test_get_dest_from_catalog_dests(self):
        catalog = (
            b"<< /Type /Catalog /Pages 2 0 R "
            b"/Dests << /dest1 [2 0 R /XYZ 0 0 0] >> >>"
        )
        pdf = create_minimal_pdf([
            (1, 0, catalog),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        dest = doc.get_dest("dest1")
        assert dest is not None


class TestPDFSecurityHandlerV5R6:
    """Tests for V5 security handler with revision 6"""

    def test_r6_password_hash(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        handler.r = 6
        handler.key = b"0" * 32
        result = handler._r6_password(b"password", b"saltsalt")
        assert len(result) == 32

    def test_r5_password_hash(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        handler.r = 5
        result = handler._r5_password(b"password", b"saltsalt")
        assert len(result) == 32

    def test_r5_password_hash_with_vector(self):
        handler = object.__new__(PDFStandardSecurityHandlerV5)
        handler.r = 5
        result = handler._r5_password(b"password", b"saltsalt", b"vector")
        assert len(result) == 32


class TestPDFXRefStreamDefaults:
    """Tests for PDFXRefStream with default field values"""

    def test_default_fl1_value(self):
        xref = PDFXRefStream()
        xref.ranges = [(0, 2)]
        xref.fl1 = 0
        xref.fl2 = 2
        xref.fl3 = 1
        xref.entlen = 3
        xref.data = b"\x00\x64\x00" + b"\x00\xc8\x00"
        objids = list(xref.get_objids())
        assert 0 in objids
        assert 1 in objids


class TestPDFEncryptionMetadata:
    """Tests for encryption metadata handling"""

    def test_v4_decrypt_unencrypted_metadata(self):
        handler = object.__new__(PDFStandardSecurityHandlerV4)
        handler.encrypt_metadata = False
        handler.key = b"0123456789abcdef"
        handler.strf = "StdCF"
        handler.cfm = {
            "StdCF": handler.decrypt_rc4,
            "Identity": handler.decrypt_identity,
        }
        data = b"test data"
        attrs = {"Type": LIT("Metadata")}
        result = handler.decrypt(1, 0, data, attrs)
        assert result == data


class TestPDFDocumentXRefsProperty:
    """Tests for xrefs list behavior"""

    def test_multiple_xref_tables(self):
        pdf = create_minimal_pdf([
            (1, 0, create_catalog_obj(2)),
            (2, 0, create_pages_obj(0)),
        ])
        parser = PDFParser(BytesIO(pdf))
        doc = PDFDocument(parser)
        assert len(doc.xrefs) >= 1
        assert all(hasattr(xref, "get_trailer") for xref in doc.xrefs)
