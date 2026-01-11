"""Comprehensive tests for pdfminer.high_level module.

This module tests the three main high-level functions:
- extract_text()
- extract_pages()
- extract_text_to_fp()
"""

import io
import tempfile
from pathlib import Path

import pytest

from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
from pdfminer.layout import LAParams, LTPage, LTTextBox, LTTextContainer
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfparser import PDFSyntaxError
from tests.helpers import absolute_sample_path

# Minimal valid PDF bytes for testing
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 0 >>
stream
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000222 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
275
%%EOF
"""


# PDF with text content for testing
PDF_WITH_TEXT = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000261 00000 n
0000000360 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
432
%%EOF
"""


# PDF with two pages for pagination testing
PDF_TWO_PAGES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
4 0 obj
<< /Length 48 >>
stream
BT
/F1 12 Tf
100 700 Td
(Page One) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 7 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
7 0 obj
<< /Length 48 >>
stream
BT
/F1 12 Tf
100 700 Td
(Page Two) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000120 00000 n
0000000266 00000 n
0000000369 00000 n
0000000437 00000 n
0000000583 00000 n
trailer
<< /Size 8 /Root 1 0 R >>
startxref
686
%%EOF
"""


class TestExtractText:
    """Tests for the extract_text() function."""

    def test_extract_text_from_bytesio(self):
        """Test extract_text with a BytesIO object."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        result = extract_text(pdf_file)
        assert isinstance(result, str)

    def test_extract_text_from_file_path(self):
        """Test extract_text with a file path string."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_from_pathlib(self):
        """Test extract_text with a pathlib.Path object."""
        path = Path(absolute_sample_path("simple1.pdf"))
        result = extract_text(path)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_from_file_handle(self):
        """Test extract_text with an open file handle."""
        path = absolute_sample_path("simple1.pdf")
        with open(path, "rb") as f:
            result = extract_text(f)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_empty_pdf(self):
        """Test extract_text with a minimal PDF containing no text."""
        pdf_file = io.BytesIO(MINIMAL_PDF)
        result = extract_text(pdf_file)
        assert isinstance(result, str)
        assert result.strip() == "" or result == "\f"

    def test_extract_text_returns_string(self):
        """Verify extract_text returns a string type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        result = extract_text(pdf_file)
        assert isinstance(result, str)

    def test_extract_text_with_laparams_none(self):
        """Test that passing laparams=None uses default LAParams."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, laparams=None)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_with_custom_laparams(self):
        """Test extract_text with custom LAParams."""
        path = absolute_sample_path("simple1.pdf")
        laparams = LAParams(line_margin=0.5, word_margin=0.1)
        result = extract_text(path, laparams=laparams)
        assert isinstance(result, str)

    def test_extract_text_with_boxes_flow_none(self):
        """Test extract_text with boxes_flow=None (disables layout analysis)."""
        path = absolute_sample_path("simple4.pdf")
        laparams = LAParams(boxes_flow=None)
        result = extract_text(path, laparams=laparams)
        assert isinstance(result, str)

    def test_extract_text_maxpages_zero(self):
        """Test that maxpages=0 means no limit (all pages extracted)."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, maxpages=0)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_maxpages_one(self):
        """Test extracting only the first page."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        result = extract_text(pdf_file, maxpages=1)
        assert isinstance(result, str)

    def test_extract_text_page_numbers_single(self):
        """Test extracting a specific page by page number (zero-indexed)."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        result = extract_text(pdf_file, page_numbers=[0])
        assert isinstance(result, str)

    def test_extract_text_page_numbers_list(self):
        """Test extracting multiple specific pages."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, page_numbers=[0])
        assert isinstance(result, str)
        assert "Hello" in result

    def test_extract_text_page_numbers_empty_list(self):
        """Test with empty page_numbers list returns all pages.

        Note: An empty list is falsy in Python, so PDFPage.get_pages treats
        it the same as None, meaning all pages are extracted.
        """
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, page_numbers=[])
        assert "Hello" in result

    def test_extract_text_page_numbers_set(self):
        """Test page_numbers can be a set (Container[int])."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, page_numbers={0})
        assert isinstance(result, str)

    def test_extract_text_page_numbers_out_of_range(self):
        """Test with page number beyond document pages."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, page_numbers=[999])
        assert result == ""

    def test_extract_text_caching_enabled(self):
        """Test extract_text with caching enabled (default)."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, caching=True)
        assert isinstance(result, str)

    def test_extract_text_caching_disabled(self):
        """Test extract_text with caching disabled."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, caching=False)
        assert isinstance(result, str)

    def test_extract_text_codec_utf8(self):
        """Test extract_text with utf-8 codec (default)."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, codec="utf-8")
        assert isinstance(result, str)

    def test_extract_text_codec_latin1(self):
        """Test extract_text with latin-1 codec."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, codec="latin-1")
        assert isinstance(result, str)

    def test_extract_text_empty_password(self):
        """Test extract_text with empty password (default for non-encrypted)."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, password="")
        assert isinstance(result, str)


class TestExtractPages:
    """Tests for the extract_pages() function."""

    def test_extract_pages_returns_iterator(self):
        """Test that extract_pages returns an iterator."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_pages(path)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_extract_pages_yields_ltpage(self):
        """Test that extract_pages yields LTPage objects."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        assert len(pages) >= 1
        for page in pages:
            assert isinstance(page, LTPage)

    def test_extract_pages_from_bytesio(self):
        """Test extract_pages with a BytesIO object."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        pages = list(extract_pages(pdf_file))
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)

    def test_extract_pages_from_file_path(self):
        """Test extract_pages with a file path."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        assert len(pages) >= 1

    def test_extract_pages_from_pathlib(self):
        """Test extract_pages with a pathlib.Path object."""
        path = Path(absolute_sample_path("simple1.pdf"))
        pages = list(extract_pages(path))
        assert len(pages) >= 1

    def test_extract_pages_from_file_handle(self):
        """Test extract_pages with an open file handle."""
        path = absolute_sample_path("simple1.pdf")
        with open(path, "rb") as f:
            pages = list(extract_pages(f))
        assert len(pages) >= 1

    def test_extract_pages_ltpage_has_pageid(self):
        """Test that LTPage objects have a pageid attribute."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        assert hasattr(pages[0], "pageid")
        assert pages[0].pageid == 1

    def test_extract_pages_ltpage_has_bbox(self):
        """Test that LTPage objects have bounding box attributes."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        page = pages[0]
        assert hasattr(page, "bbox")
        assert hasattr(page, "x0")
        assert hasattr(page, "y0")
        assert hasattr(page, "x1")
        assert hasattr(page, "y1")
        assert hasattr(page, "width")
        assert hasattr(page, "height")

    def test_extract_pages_ltpage_iterable(self):
        """Test that LTPage objects can be iterated to get children."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        page = pages[0]
        elements = list(page)
        assert isinstance(elements, list)

    def test_extract_pages_contains_text_elements(self):
        """Test that page contains text elements."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        page = pages[0]
        text_elements = [e for e in page if isinstance(e, LTTextContainer)]
        assert len(text_elements) > 0

    def test_extract_pages_with_laparams_none(self):
        """Test that passing laparams=None uses default LAParams."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path, laparams=None))
        assert len(pages) >= 1

    def test_extract_pages_with_custom_laparams(self):
        """Test extract_pages with custom LAParams."""
        path = absolute_sample_path("simple4.pdf")
        laparams = LAParams(line_margin=0.21)
        pages = list(extract_pages(path, laparams=laparams))
        assert len(pages) == 1
        text_boxes = [e for e in pages[0] if isinstance(e, LTTextBox)]
        assert len(text_boxes) == 1

    def test_extract_pages_line_margin_affects_grouping(self):
        """Test that line_margin parameter affects text grouping."""
        path = absolute_sample_path("simple4.pdf")

        laparams_small = LAParams(line_margin=0.19)
        pages_small = list(extract_pages(path, laparams=laparams_small))
        text_boxes_small = [e for e in pages_small[0] if isinstance(e, LTTextBox)]

        laparams_large = LAParams(line_margin=0.21)
        pages_large = list(extract_pages(path, laparams=laparams_large))
        text_boxes_large = [e for e in pages_large[0] if isinstance(e, LTTextBox)]

        assert len(text_boxes_small) > len(text_boxes_large)

    def test_extract_pages_boxes_flow_none(self):
        """Test extract_pages with boxes_flow=None."""
        path = absolute_sample_path("simple4.pdf")
        laparams = LAParams(boxes_flow=None)
        pages = list(extract_pages(path, laparams=laparams))
        assert len(pages) == 1

    def test_extract_pages_maxpages_zero(self):
        """Test that maxpages=0 means no limit."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        pages = list(extract_pages(pdf_file, maxpages=0))
        assert len(pages) == 2

    def test_extract_pages_maxpages_one(self):
        """Test extracting only the first page with maxpages=1."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        pages = list(extract_pages(pdf_file, maxpages=1))
        assert len(pages) == 1

    def test_extract_pages_page_numbers_single(self):
        """Test extracting a specific page by number."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        pages = list(extract_pages(pdf_file, page_numbers=[1]))
        assert len(pages) == 1

    def test_extract_pages_page_numbers_multiple(self):
        """Test extracting multiple specific pages."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        pages = list(extract_pages(pdf_file, page_numbers=[0, 1]))
        assert len(pages) == 2

    def test_extract_pages_page_numbers_empty(self):
        """Test with empty page_numbers list returns all pages.

        Note: An empty list is falsy in Python, so PDFPage.get_pages treats
        it the same as None, meaning all pages are extracted.
        """
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path, page_numbers=[]))
        assert len(pages) >= 1

    def test_extract_pages_page_numbers_out_of_range(self):
        """Test with page number beyond document pages."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path, page_numbers=[999]))
        assert len(pages) == 0

    def test_extract_pages_caching_enabled(self):
        """Test extract_pages with caching enabled."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path, caching=True))
        assert len(pages) >= 1

    def test_extract_pages_caching_disabled(self):
        """Test extract_pages with caching disabled."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path, caching=False))
        assert len(pages) >= 1


class TestExtractTextToFp:
    """Tests for the extract_text_to_fp() function."""

    def test_extract_text_to_fp_text_output(self):
        """Test extract_text_to_fp with text output type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, output_type="text")
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_xml_output(self):
        """Test extract_text_to_fp with xml output type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="xml")
        result = output.getvalue()
        assert b"<?xml" in result or b"<pages" in result

    def test_extract_text_to_fp_html_output(self):
        """Test extract_text_to_fp with html output type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="html")
        result = output.getvalue()
        assert b"<html" in result.lower() or b"<!doctype" in result.lower()

    def test_extract_text_to_fp_hocr_output(self):
        """Test extract_text_to_fp with hocr output type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="hocr")
        result = output.getvalue()
        assert b"ocr" in result.lower() or len(result) > 0

    def test_extract_text_to_fp_tag_output(self):
        """Test extract_text_to_fp with tag output type."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="tag")
        result = output.getvalue()
        assert isinstance(result, bytes)

    def test_extract_text_to_fp_invalid_output_type(self):
        """Test that invalid output_type raises PDFValueError."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        with pytest.raises(PDFValueError):
            extract_text_to_fp(pdf_file, output, output_type="invalid_type")

    def test_extract_text_to_fp_with_laparams(self):
        """Test extract_text_to_fp with custom LAParams."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        laparams = LAParams(line_margin=0.5)
        extract_text_to_fp(pdf_file, output, laparams=laparams)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_laparams_none(self):
        """Test extract_text_to_fp with laparams=None."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, laparams=None)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_maxpages(self):
        """Test extract_text_to_fp with maxpages parameter."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, maxpages=1)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_page_numbers(self):
        """Test extract_text_to_fp with page_numbers parameter."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, page_numbers=[0])
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_codec(self):
        """Test extract_text_to_fp with codec parameter."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, codec="utf-8")
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_scale(self):
        """Test extract_text_to_fp with scale parameter."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="html", scale=2.0)
        result = output.getvalue()
        assert len(result) > 0

    def test_extract_text_to_fp_rotation(self):
        """Test extract_text_to_fp with rotation parameter."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, rotation=90)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_rotation_values(self):
        """Test extract_text_to_fp with various rotation values."""
        for rotation in [0, 90, 180, 270]:
            pdf_file = io.BytesIO(PDF_WITH_TEXT)
            output = io.StringIO()
            extract_text_to_fp(pdf_file, output, rotation=rotation)
            assert isinstance(output.getvalue(), str)

    def test_extract_text_to_fp_strip_control(self):
        """Test extract_text_to_fp with strip_control parameter."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="xml", strip_control=True)
        result = output.getvalue()
        assert isinstance(result, bytes)

    def test_extract_text_to_fp_disable_caching(self):
        """Test extract_text_to_fp with disable_caching parameter."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, disable_caching=True)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_caching_enabled(self):
        """Test extract_text_to_fp with caching enabled (disable_caching=False)."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, disable_caching=False)
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_empty_password(self):
        """Test extract_text_to_fp with empty password."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        extract_text_to_fp(pdf_file, output, password="")
        result = output.getvalue()
        assert isinstance(result, str)

    def test_extract_text_to_fp_layoutmode_normal(self):
        """Test extract_text_to_fp with normal layoutmode."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="html", layoutmode="normal")
        result = output.getvalue()
        assert len(result) > 0

    def test_extract_text_to_fp_layoutmode_exact(self):
        """Test extract_text_to_fp with exact layoutmode."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="html", layoutmode="exact")
        result = output.getvalue()
        assert len(result) > 0

    def test_extract_text_to_fp_layoutmode_loose(self):
        """Test extract_text_to_fp with loose layoutmode."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.BytesIO()
        extract_text_to_fp(pdf_file, output, output_type="html", layoutmode="loose")
        result = output.getvalue()
        assert len(result) > 0

    def test_extract_text_to_fp_with_tempdir_output_dir(self):
        """Test extract_text_to_fp with output_dir for image extraction."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_text_to_fp(pdf_file, output, output_dir=tmpdir)
        result = output.getvalue()
        assert isinstance(result, str)


class TestLAParams:
    """Tests for LAParams parameter validation."""

    def test_laparams_default_values(self):
        """Test LAParams with default values."""
        laparams = LAParams()
        assert laparams.line_overlap == 0.5
        assert laparams.char_margin == 2.0
        assert laparams.line_margin == 0.5
        assert laparams.word_margin == 0.1
        assert laparams.boxes_flow == 0.5
        assert laparams.detect_vertical is False
        assert laparams.all_texts is False

    def test_laparams_custom_values(self):
        """Test LAParams with custom values."""
        laparams = LAParams(
            line_overlap=0.3,
            char_margin=1.5,
            line_margin=0.3,
            word_margin=0.2,
            boxes_flow=0.8,
            detect_vertical=True,
            all_texts=True,
        )
        assert laparams.line_overlap == 0.3
        assert laparams.char_margin == 1.5
        assert laparams.line_margin == 0.3
        assert laparams.word_margin == 0.2
        assert laparams.boxes_flow == 0.8
        assert laparams.detect_vertical is True
        assert laparams.all_texts is True

    def test_laparams_boxes_flow_none(self):
        """Test LAParams with boxes_flow=None."""
        laparams = LAParams(boxes_flow=None)
        assert laparams.boxes_flow is None

    def test_laparams_boxes_flow_min_value(self):
        """Test LAParams with boxes_flow=-1."""
        laparams = LAParams(boxes_flow=-1)
        assert laparams.boxes_flow == -1

    def test_laparams_boxes_flow_max_value(self):
        """Test LAParams with boxes_flow=1."""
        laparams = LAParams(boxes_flow=1)
        assert laparams.boxes_flow == 1

    def test_laparams_boxes_flow_invalid_too_low(self):
        """Test LAParams with boxes_flow < -1 raises error."""
        with pytest.raises(PDFValueError):
            LAParams(boxes_flow=-1.5)

    def test_laparams_boxes_flow_invalid_too_high(self):
        """Test LAParams with boxes_flow > 1 raises error."""
        with pytest.raises(PDFValueError):
            LAParams(boxes_flow=1.5)

    def test_laparams_repr(self):
        """Test LAParams __repr__ method."""
        laparams = LAParams()
        repr_str = repr(laparams)
        assert "LAParams" in repr_str
        assert "char_margin" in repr_str
        assert "line_margin" in repr_str
        assert "word_margin" in repr_str

    def test_laparams_detect_vertical(self):
        """Test LAParams with detect_vertical=True."""
        laparams = LAParams(detect_vertical=True)
        assert laparams.detect_vertical is True

    def test_laparams_all_texts(self):
        """Test LAParams with all_texts=True."""
        laparams = LAParams(all_texts=True)
        assert laparams.all_texts is True


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_extract_text_nonexistent_file(self):
        """Test extract_text with a nonexistent file path."""
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/path/to/file.pdf")

    def test_extract_pages_nonexistent_file(self):
        """Test extract_pages with a nonexistent file path."""
        with pytest.raises(FileNotFoundError):
            list(extract_pages("/nonexistent/path/to/file.pdf"))

    def test_extract_text_empty_bytesio(self):
        """Test extract_text with an empty BytesIO raises PDFSyntaxError."""
        pdf_file = io.BytesIO(b"")
        with pytest.raises(PDFSyntaxError):
            extract_text(pdf_file)

    def test_extract_text_invalid_pdf_bytes(self):
        """Test extract_text with invalid PDF bytes raises PDFSyntaxError."""
        pdf_file = io.BytesIO(b"This is not a PDF file")
        with pytest.raises(PDFSyntaxError):
            extract_text(pdf_file)

    def test_extract_pages_empty_bytesio(self):
        """Test extract_pages with an empty BytesIO raises PDFSyntaxError."""
        pdf_file = io.BytesIO(b"")
        with pytest.raises(PDFSyntaxError):
            list(extract_pages(pdf_file))

    def test_extract_text_to_fp_empty_bytesio(self):
        """Test extract_text_to_fp with an empty BytesIO raises PDFSyntaxError."""
        pdf_file = io.BytesIO(b"")
        output = io.StringIO()
        with pytest.raises(PDFSyntaxError):
            extract_text_to_fp(pdf_file, output)

    def test_multiple_calls_same_bytesio(self):
        """Test calling extract_text multiple times on the same BytesIO."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        result1 = extract_text(pdf_file)

        pdf_file.seek(0)
        result2 = extract_text(pdf_file)
        assert result1 == result2

    def test_page_numbers_negative(self):
        """Test page_numbers with negative index (should not match any page)."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path, page_numbers=[-1])
        assert result == ""

    def test_maxpages_and_page_numbers_together(self):
        """Test using both maxpages and page_numbers parameters."""
        pdf_file = io.BytesIO(PDF_TWO_PAGES)
        pages = list(extract_pages(pdf_file, page_numbers=[0, 1], maxpages=1))
        assert len(pages) == 1


class TestSamplePDFs:
    """Tests using sample PDF files."""

    def test_simple1_extract_text(self):
        """Test extract_text on simple1.pdf."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path)
        assert "Hello" in result
        assert "World" in result

    def test_simple2_extract_text(self):
        """Test extract_text on simple2.pdf (empty page)."""
        path = absolute_sample_path("simple2.pdf")
        result = extract_text(path)
        assert isinstance(result, str)

    def test_simple3_extract_text(self):
        """Test extract_text on simple3.pdf (Japanese text)."""
        path = absolute_sample_path("simple3.pdf")
        result = extract_text(path)
        assert "Hello" in result

    def test_simple4_extract_text(self):
        """Test extract_text on simple4.pdf."""
        path = absolute_sample_path("simple4.pdf")
        result = extract_text(path)
        assert "Text1" in result
        assert "Text2" in result
        assert "Text3" in result

    def test_simple5_extract_text(self):
        """Test extract_text on simple5.pdf."""
        path = absolute_sample_path("simple5.pdf")
        result = extract_text(path)
        assert "Heading" in result

    def test_simple1_extract_pages(self):
        """Test extract_pages on simple1.pdf."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)

    def test_simple4_extract_pages_content(self):
        """Test extract_pages content on simple4.pdf."""
        path = absolute_sample_path("simple4.pdf")
        pages = list(extract_pages(path))
        page = pages[0]
        text_boxes = [e for e in page if isinstance(e, LTTextContainer)]
        full_text = "".join(box.get_text() for box in text_boxes)
        assert "Text1" in full_text


class TestReturnTypes:
    """Tests to verify correct return types."""

    def test_extract_text_return_type(self):
        """Verify extract_text returns str."""
        path = absolute_sample_path("simple1.pdf")
        result = extract_text(path)
        assert isinstance(result, str)

    def test_extract_pages_yield_type(self):
        """Verify extract_pages yields LTPage objects."""
        path = absolute_sample_path("simple1.pdf")
        for page in extract_pages(path):
            assert isinstance(page, LTPage)

    def test_extract_text_to_fp_returns_none(self):
        """Verify extract_text_to_fp returns None."""
        pdf_file = io.BytesIO(PDF_WITH_TEXT)
        output = io.StringIO()
        result = extract_text_to_fp(pdf_file, output)
        assert result is None

    def test_ltpage_contains_expected_types(self):
        """Test that LTPage contains expected element types."""
        path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(path))
        page = pages[0]
        has_text_container = any(isinstance(e, LTTextContainer) for e in page)
        assert has_text_container
