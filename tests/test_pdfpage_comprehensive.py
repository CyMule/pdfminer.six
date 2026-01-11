"""Comprehensive tests for pdfminer/pdfpage.py.

Tests cover:
- PDFPage class initialization and attributes
- PDFPage.create_pages() method
- PDFPage.get_pages() static method
- Page iteration, filtering, and maxpages
- Edge cases (missing MediaBox, invalid CropBox, rotation normalization, etc.)
"""

from io import BytesIO

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import LITERAL_PAGE, LITERAL_PAGES, PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.psparser import LIT


def make_minimal_pdf(
    num_pages: int = 1,
    mediabox: str = "0 0 612 792",
    cropbox: str | None = None,
    rotate: int | None = None,
    resources: str = "<< >>",
    page_contents: list[str] | None = None,
) -> bytes:
    """Create a minimal valid PDF with the specified number of pages.

    Args:
        num_pages: Number of pages to create
        mediabox: MediaBox specification (e.g., "0 0 612 792")
        cropbox: Optional CropBox specification
        rotate: Optional rotation value in degrees
        resources: Resources dictionary string
        page_contents: Optional list of content stream strings per page

    Returns:
        PDF as bytes
    """
    objects: list[str] = []
    obj_num = 1

    catalog_obj = obj_num
    objects.append(f"""{obj_num} 0 obj
<<
  /Type /Catalog
  /Pages {obj_num + 1} 0 R
>>
endobj""")
    obj_num += 1

    pages_obj = obj_num
    kids_refs = " ".join([f"{obj_num + 1 + i} 0 R" for i in range(num_pages)])
    objects.append(f"""{obj_num} 0 obj
<<
  /Type /Pages
  /Kids [ {kids_refs} ]
  /Count {num_pages}
>>
endobj""")
    obj_num += 1

    for i in range(num_pages):
        page_attrs = [
            "/Type /Page",
            f"/Parent {pages_obj} 0 R",
            f"/MediaBox [ {mediabox} ]",
            f"/Resources {resources}",
        ]
        if cropbox:
            page_attrs.append(f"/CropBox [ {cropbox} ]")
        if rotate is not None:
            page_attrs.append(f"/Rotate {rotate}")

        if page_contents and i < len(page_contents):
            content_obj = obj_num + num_pages + i
            page_attrs.append(f"/Contents {content_obj} 0 R")

        objects.append(f"""{obj_num} 0 obj
<<
  {chr(10).join('  ' + attr for attr in page_attrs)}
>>
endobj""")
        obj_num += 1

    if page_contents:
        for content in page_contents:
            content_bytes = content.encode("latin-1")
            objects.append(f"""{obj_num} 0 obj
<< /Length {len(content_bytes)} >>
stream
{content}
endstream
endobj""")
            obj_num += 1

    trailer = f"""trailer
<<
  /Size {obj_num}
  /Root {catalog_obj} 0 R
>>
%%EOF"""

    pdf_content = "%PDF-1.4\n" + "\n".join(objects) + "\n" + trailer
    return pdf_content.encode("latin-1")


def make_pdf_with_inherited_attrs(
    mediabox: str = "0 0 612 792",
    cropbox: str | None = None,
    rotate: int | None = None,
    resources: str | None = None,
) -> bytes:
    """Create a PDF where pages inherit attributes from the Pages node."""
    pages_attrs = ["/Type /Pages", "/Kids [ 3 0 R ]", "/Count 1"]
    if mediabox:
        pages_attrs.append(f"/MediaBox [ {mediabox} ]")
    if cropbox:
        pages_attrs.append(f"/CropBox [ {cropbox} ]")
    if rotate is not None:
        pages_attrs.append(f"/Rotate {rotate}")
    if resources:
        pages_attrs.append(f"/Resources {resources}")

    pdf = f"""%PDF-1.4
1 0 obj
<<
  /Type /Catalog
  /Pages 2 0 R
>>
endobj
2 0 obj
<<
  {chr(10).join('  ' + attr for attr in pages_attrs)}
>>
endobj
3 0 obj
<<
  /Type /Page
  /Parent 2 0 R
>>
endobj
trailer
<<
  /Size 4
  /Root 1 0 R
>>
%%EOF"""
    return pdf.encode("latin-1")


def make_pdf_no_pages_catalog() -> bytes:
    """Create a PDF without /Pages in catalog (fallback case)."""
    pdf = b"""%PDF-1.4
1 0 obj
<<
  /Type /Catalog
>>
endobj
2 0 obj
<<
  /Type /Page
  /MediaBox [ 0 0 612 792 ]
  /Resources << >>
>>
endobj
trailer
<<
  /Size 3
  /Root 1 0 R
>>
%%EOF"""
    return pdf


class TestPDFPageInit:
    """Tests for PDFPage.__init__ and attribute parsing."""

    def test_basic_page_attributes(self) -> None:
        """Test that basic page attributes are correctly parsed."""
        pdf_bytes = make_minimal_pdf(num_pages=1)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        page = pages[0]
        assert page.mediabox == (0.0, 0.0, 612.0, 792.0)
        assert page.cropbox == (0.0, 0.0, 612.0, 792.0)
        assert page.rotate == 0

    def test_custom_mediabox(self) -> None:
        """Test custom MediaBox dimensions."""
        pdf_bytes = make_minimal_pdf(mediabox="0 0 1000 1500")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].mediabox == (0.0, 0.0, 1000.0, 1500.0)

    def test_custom_cropbox(self) -> None:
        """Test that CropBox is correctly parsed when specified."""
        pdf_bytes = make_minimal_pdf(
            mediabox="0 0 612 792", cropbox="50 50 562 742"
        )
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        page = pages[0]
        assert page.mediabox == (0.0, 0.0, 612.0, 792.0)
        assert page.cropbox == (50.0, 50.0, 562.0, 742.0)

    def test_cropbox_defaults_to_mediabox(self) -> None:
        """Test that CropBox defaults to MediaBox when not specified."""
        pdf_bytes = make_minimal_pdf(mediabox="0 0 800 600")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        page = pages[0]
        assert page.cropbox == page.mediabox
        assert page.cropbox == (0.0, 0.0, 800.0, 600.0)

    def test_rotation_values(self) -> None:
        """Test various rotation values are normalized correctly."""
        for rotate_val, expected in [
            (0, 0),
            (90, 90),
            (180, 180),
            (270, 270),
            (360, 0),
            (-90, 270),
            (-180, 180),
            (-270, 90),
            (450, 90),
        ]:
            pdf_bytes = make_minimal_pdf(rotate=rotate_val)
            fp = BytesIO(pdf_bytes)
            pages = list(PDFPage.get_pages(fp))
            assert pages[0].rotate == expected, f"rotate={rotate_val}"

    def test_page_repr(self) -> None:
        """Test PDFPage.__repr__ method."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        repr_str = repr(pages[0])
        assert "<PDFPage:" in repr_str
        assert "MediaBox=" in repr_str

    def test_page_has_doc_reference(self) -> None:
        """Test that PDFPage has reference to PDFDocument."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].doc is not None
        assert isinstance(pages[0].doc, PDFDocument)

    def test_page_has_pageid(self) -> None:
        """Test that PDFPage has a pageid attribute."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].pageid is not None


class TestPDFPageInheritance:
    """Tests for attribute inheritance from parent Pages node."""

    def test_inherit_mediabox(self) -> None:
        """Test that MediaBox is inherited from Pages node."""
        pdf_bytes = make_pdf_with_inherited_attrs(mediabox="0 0 500 700")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].mediabox == (0.0, 0.0, 500.0, 700.0)

    def test_inherit_cropbox(self) -> None:
        """Test that CropBox is inherited from Pages node."""
        pdf_bytes = make_pdf_with_inherited_attrs(
            mediabox="0 0 612 792", cropbox="10 10 602 782"
        )
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].cropbox == (10.0, 10.0, 602.0, 782.0)

    def test_inherit_rotate(self) -> None:
        """Test that Rotate is inherited from Pages node."""
        pdf_bytes = make_pdf_with_inherited_attrs(
            mediabox="0 0 612 792", rotate=90
        )
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].rotate == 90

    def test_inherit_resources(self) -> None:
        """Test that Resources is inherited from Pages node."""
        pdf_bytes = make_pdf_with_inherited_attrs(
            mediabox="0 0 612 792",
            resources="<< /Font << /F1 4 0 R >> >>",
        )
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].resources is not None


class TestPDFPageCreatePages:
    """Tests for PDFPage.create_pages() class method."""

    def test_create_pages_single(self) -> None:
        """Test create_pages with a single page document."""
        pdf_bytes = make_minimal_pdf(num_pages=1)
        fp = BytesIO(pdf_bytes)
        parser = PDFParser(fp)
        doc = PDFDocument(parser)

        pages = list(PDFPage.create_pages(doc))
        assert len(pages) == 1

    def test_create_pages_multiple(self) -> None:
        """Test create_pages with multiple pages."""
        pdf_bytes = make_minimal_pdf(num_pages=5)
        fp = BytesIO(pdf_bytes)
        parser = PDFParser(fp)
        doc = PDFDocument(parser)

        pages = list(PDFPage.create_pages(doc))
        assert len(pages) == 5

    def test_create_pages_returns_iterator(self) -> None:
        """Test that create_pages returns an iterator."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)
        parser = PDFParser(fp)
        doc = PDFDocument(parser)

        result = PDFPage.create_pages(doc)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_create_pages_unique_pageids(self) -> None:
        """Test that each page has a unique pageid."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)
        parser = PDFParser(fp)
        doc = PDFDocument(parser)

        pages = list(PDFPage.create_pages(doc))
        pageids = [p.pageid for p in pages]
        assert len(set(pageids)) == len(pageids)


class TestPDFPageGetPages:
    """Tests for PDFPage.get_pages() static method."""

    def test_get_pages_basic(self) -> None:
        """Test basic get_pages functionality."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))
        assert len(pages) == 3

    def test_get_pages_with_pagenos_filter(self) -> None:
        """Test filtering pages by page numbers."""
        pdf_bytes = make_minimal_pdf(num_pages=5)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, pagenos=[0, 2, 4]))
        assert len(pages) == 3

    def test_get_pages_with_set_pagenos(self) -> None:
        """Test filtering with set of page numbers."""
        pdf_bytes = make_minimal_pdf(num_pages=5)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, pagenos={1, 3}))
        assert len(pages) == 2

    def test_get_pages_pagenos_out_of_range(self) -> None:
        """Test that out-of-range page numbers are ignored."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, pagenos=[0, 10, 20]))
        assert len(pages) == 1

    def test_get_pages_empty_pagenos(self) -> None:
        """Test with empty pagenos list yields all pages."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, pagenos=[]))
        assert len(pages) == 3

    def test_get_pages_with_maxpages(self) -> None:
        """Test limiting number of pages with maxpages."""
        pdf_bytes = make_minimal_pdf(num_pages=10)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, maxpages=3))
        assert len(pages) == 3

    def test_get_pages_maxpages_zero_means_unlimited(self) -> None:
        """Test that maxpages=0 means no limit."""
        pdf_bytes = make_minimal_pdf(num_pages=5)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, maxpages=0))
        assert len(pages) == 5

    def test_get_pages_maxpages_with_pagenos(self) -> None:
        """Test maxpages with pagenos filter."""
        pdf_bytes = make_minimal_pdf(num_pages=10)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, pagenos=[0, 2, 4, 6, 8], maxpages=2))
        assert len(pages) == 2

    def test_get_pages_password_empty_string(self) -> None:
        """Test get_pages with empty password string."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, password=""))
        assert len(pages) == 1

    def test_get_pages_caching_enabled(self) -> None:
        """Test get_pages with caching enabled (default)."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, caching=True))
        assert len(pages) == 1

    def test_get_pages_caching_disabled(self) -> None:
        """Test get_pages with caching disabled."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, caching=False))
        assert len(pages) == 1


class TestPDFPageEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_mediabox_defaults_to_us_letter(self) -> None:
        """Test that missing MediaBox defaults to US Letter size."""
        pdf_bytes = b"""%PDF-1.4
1 0 obj
<<
  /Type /Catalog
  /Pages 2 0 R
>>
endobj
2 0 obj
<<
  /Type /Pages
  /Kids [ 3 0 R ]
  /Count 1
>>
endobj
3 0 obj
<<
  /Type /Page
  /Parent 2 0 R
  /Resources << >>
>>
endobj
trailer
<<
  /Size 4
  /Root 1 0 R
>>
%%EOF"""
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].mediabox == (0.0, 0.0, 612.0, 792.0)

    def test_page_with_contents_stream(self) -> None:
        """Test page with content stream."""
        content = "BT /F1 12 Tf 100 700 Td (Hello) Tj ET"
        pdf_bytes = make_minimal_pdf(num_pages=1, page_contents=[content])
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].contents is not None

    def test_page_without_contents(self) -> None:
        """Test page without content stream."""
        pdf_bytes = make_minimal_pdf(num_pages=1)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1
        assert pages[0].contents == []

    def test_fallback_when_pages_missing_from_catalog(self) -> None:
        """Test fallback path when /Pages is missing from catalog."""
        pdf_bytes = make_pdf_no_pages_catalog()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert len(pages) == 1

    def test_negative_rotation_normalized(self) -> None:
        """Test that negative rotation values are normalized."""
        pdf_bytes = make_minimal_pdf(rotate=-90)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].rotate == 270

    def test_large_rotation_normalized(self) -> None:
        """Test that rotation values > 360 are normalized."""
        pdf_bytes = make_minimal_pdf(rotate=720)
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].rotate == 0

    def test_negative_mediabox_coordinates(self) -> None:
        """Test MediaBox with negative coordinates."""
        pdf_bytes = make_minimal_pdf(mediabox="-100 -100 500 700")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].mediabox == (-100.0, -100.0, 500.0, 700.0)

    def test_float_mediabox_values(self) -> None:
        """Test MediaBox with floating point values."""
        pdf_bytes = make_minimal_pdf(mediabox="0.5 0.5 612.5 792.5")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].mediabox == (0.5, 0.5, 612.5, 792.5)


class TestPDFPageLiterals:
    """Tests for PDF literal constants."""

    def test_literal_page(self) -> None:
        """Test LITERAL_PAGE constant."""
        assert LIT("Page") == LITERAL_PAGE

    def test_literal_pages(self) -> None:
        """Test LITERAL_PAGES constant."""
        assert LIT("Pages") == LITERAL_PAGES


class TestPDFPageLabel:
    """Tests for page label functionality."""

    def test_page_label_is_none_by_default(self) -> None:
        """Test that page label is None when not specified in PDF."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].label is None


class TestPDFPageExtractable:
    """Tests for text extraction permission checking."""

    def test_check_extractable_false_by_default(self) -> None:
        """Test that check_extractable is False by default."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, check_extractable=False))
        assert len(pages) == 1

    def test_check_extractable_with_allowed_pdf(self) -> None:
        """Test check_extractable=True with a PDF that allows extraction."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp, check_extractable=True))
        assert len(pages) == 1


class TestPDFPageMultiplePages:
    """Tests for multi-page PDF handling."""

    def test_iterate_all_pages(self) -> None:
        """Test iterating through all pages of a multi-page PDF."""
        pdf_bytes = make_minimal_pdf(num_pages=10)
        fp = BytesIO(pdf_bytes)

        pages = list(PDFPage.get_pages(fp))
        assert len(pages) == 10

    def test_each_page_has_correct_structure(self) -> None:
        """Test that each page in a multi-page PDF has correct structure."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)

        for page in PDFPage.get_pages(fp):
            assert page.doc is not None
            assert page.pageid is not None
            assert page.mediabox is not None
            assert page.cropbox is not None
            assert page.rotate is not None

    def test_pages_can_be_consumed_only_once(self) -> None:
        """Test that page iterator can only be consumed once."""
        pdf_bytes = make_minimal_pdf(num_pages=3)
        fp = BytesIO(pdf_bytes)

        pages_iter = PDFPage.get_pages(fp)
        first_pass = list(pages_iter)
        second_pass = list(pages_iter)

        assert len(first_pass) == 3
        assert len(second_pass) == 0


class TestPDFPageInheritableAttrs:
    """Tests for INHERITABLE_ATTRS class variable."""

    def test_inheritable_attrs_contains_resources(self) -> None:
        """Test that Resources is in INHERITABLE_ATTRS."""
        assert "Resources" in PDFPage.INHERITABLE_ATTRS

    def test_inheritable_attrs_contains_mediabox(self) -> None:
        """Test that MediaBox is in INHERITABLE_ATTRS."""
        assert "MediaBox" in PDFPage.INHERITABLE_ATTRS

    def test_inheritable_attrs_contains_cropbox(self) -> None:
        """Test that CropBox is in INHERITABLE_ATTRS."""
        assert "CropBox" in PDFPage.INHERITABLE_ATTRS

    def test_inheritable_attrs_contains_rotate(self) -> None:
        """Test that Rotate is in INHERITABLE_ATTRS."""
        assert "Rotate" in PDFPage.INHERITABLE_ATTRS

    def test_inheritable_attrs_is_set(self) -> None:
        """Test that INHERITABLE_ATTRS is a set."""
        assert isinstance(PDFPage.INHERITABLE_ATTRS, set)


class TestPDFPageParseMediabox:
    """Tests for _parse_mediabox method."""

    def test_parse_mediabox_returns_rect(self) -> None:
        """Test that _parse_mediabox returns a Rect tuple."""
        pdf_bytes = make_minimal_pdf(mediabox="0 0 612 792")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        mediabox = pages[0].mediabox
        assert isinstance(mediabox, tuple)
        assert len(mediabox) == 4
        assert all(isinstance(v, float) for v in mediabox)


class TestPDFPageParseCropbox:
    """Tests for _parse_cropbox method."""

    def test_parse_cropbox_returns_rect(self) -> None:
        """Test that _parse_cropbox returns a Rect tuple."""
        pdf_bytes = make_minimal_pdf(cropbox="10 10 602 782")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        cropbox = pages[0].cropbox
        assert isinstance(cropbox, tuple)
        assert len(cropbox) == 4
        assert all(isinstance(v, float) for v in cropbox)


class TestPDFPageParseContents:
    """Tests for _parse_contents method."""

    def test_parse_contents_empty_when_none(self) -> None:
        """Test that _parse_contents returns empty list when no contents."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].contents == []

    def test_parse_contents_returns_list(self) -> None:
        """Test that _parse_contents returns a list."""
        content = "BT /F1 12 Tf 100 700 Td (Test) Tj ET"
        pdf_bytes = make_minimal_pdf(page_contents=[content])
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert isinstance(pages[0].contents, list)


class TestPDFPageAttrs:
    """Tests for PDFPage.attrs dictionary."""

    def test_attrs_is_dict(self) -> None:
        """Test that attrs is a dictionary."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert isinstance(pages[0].attrs, dict)

    def test_attrs_contains_type(self) -> None:
        """Test that attrs contains Type key."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert LIT("Type") in pages[0].attrs or "Type" in pages[0].attrs


class TestPDFPageResources:
    """Tests for PDFPage.resources attribute."""

    def test_resources_is_dict(self) -> None:
        """Test that resources is a dictionary."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert isinstance(pages[0].resources, dict)

    def test_empty_resources(self) -> None:
        """Test page with empty resources dictionary."""
        pdf_bytes = make_minimal_pdf(resources="<< >>")
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].resources == {}


class TestPDFPageAnnotsAndBeads:
    """Tests for annotations and beads attributes."""

    def test_annots_is_none_when_not_present(self) -> None:
        """Test that annots is None when not specified in PDF."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].annots is None

    def test_beads_is_none_when_not_present(self) -> None:
        """Test that beads is None when not specified in PDF."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].beads is None


class TestPDFPageLastMod:
    """Tests for lastmod attribute."""

    def test_lastmod_is_none_when_not_present(self) -> None:
        """Test that lastmod is None when not specified in PDF."""
        pdf_bytes = make_minimal_pdf()
        fp = BytesIO(pdf_bytes)
        pages = list(PDFPage.get_pages(fp))

        assert pages[0].lastmod is None
