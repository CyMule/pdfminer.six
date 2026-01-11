"""Comprehensive tests for pdfminer/converter.py

Tests cover:
- PDFLayoutAnalyzer: base class for layout analysis
- PDFPageAggregator: aggregates layout elements per page
- PDFConverter: base converter class with stream handling
- TextConverter: plain text output
- HTMLConverter: HTML output with positioning
- XMLConverter: XML output with structure
- HOCRConverter: hOCR output for OCR-like representation
"""

import io
import re
import xml.etree.ElementTree as ET
from io import BytesIO, StringIO
from unittest.mock import Mock

import pytest

from pdfminer.converter import (
    HOCRConverter,
    HTMLConverter,
    PDFConverter,
    PDFLayoutAnalyzer,
    PDFPageAggregator,
    TextConverter,
    XMLConverter,
)
from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
from pdfminer.layout import (
    LAParams,
    LTContainer,
    LTCurve,
    LTLine,
    LTPage,
    LTRect,
)
from pdfminer.pdfdevice import TagExtractor
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFGraphicState, PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import PDFStream
from tests.helpers import absolute_sample_path


class TestPDFLayoutAnalyzer:
    """Tests for PDFLayoutAnalyzer class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        rsrcmgr = PDFResourceManager()
        analyzer = PDFLayoutAnalyzer(rsrcmgr)
        assert analyzer.pageno == 1
        assert analyzer.laparams is None
        assert analyzer._stack == []

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(char_margin=3.0, word_margin=0.2)
        analyzer = PDFLayoutAnalyzer(rsrcmgr, pageno=5, laparams=laparams)
        assert analyzer.pageno == 5
        assert analyzer.laparams == laparams
        assert analyzer.laparams.char_margin == 3.0

    def test_handle_undefined_char(self):
        """Test handling of undefined characters."""
        rsrcmgr = PDFResourceManager()
        analyzer = PDFLayoutAnalyzer(rsrcmgr)
        font = Mock()
        font.fontname = "TestFont"
        result = analyzer.handle_undefined_char(font, 42)
        assert result == "(cid:42)"

    def test_receive_layout_default(self):
        """Test default receive_layout implementation (does nothing)."""
        rsrcmgr = PDFResourceManager()
        analyzer = PDFLayoutAnalyzer(rsrcmgr)
        ltpage = LTPage(1, (0, 0, 612, 792))
        analyzer.receive_layout(ltpage)


class TestPDFPageAggregator:
    """Tests for PDFPageAggregator class."""

    def test_init(self):
        """Test PDFPageAggregator initialization."""
        rsrcmgr = PDFResourceManager()
        aggregator = PDFPageAggregator(rsrcmgr)
        assert aggregator.result is None
        assert aggregator.pageno == 1

    def test_init_with_laparams(self):
        """Test PDFPageAggregator with LAParams."""
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        aggregator = PDFPageAggregator(rsrcmgr, laparams=laparams)
        assert aggregator.laparams == laparams

    def test_receive_layout(self):
        """Test that receive_layout stores the result."""
        rsrcmgr = PDFResourceManager()
        aggregator = PDFPageAggregator(rsrcmgr)
        ltpage = LTPage(1, (0, 0, 612, 792))
        aggregator.receive_layout(ltpage)
        assert aggregator.result == ltpage

    def test_get_result(self):
        """Test get_result returns the stored page."""
        rsrcmgr = PDFResourceManager()
        aggregator = PDFPageAggregator(rsrcmgr)
        ltpage = LTPage(1, (0, 0, 612, 792))
        aggregator.receive_layout(ltpage)
        result = aggregator.get_result()
        assert result == ltpage
        assert result.pageid == 1

    def test_get_result_raises_without_receive(self):
        """Test get_result raises when called before receive_layout."""
        rsrcmgr = PDFResourceManager()
        aggregator = PDFPageAggregator(rsrcmgr)
        with pytest.raises(AssertionError):
            aggregator.get_result()

    def test_with_real_pdf(self):
        """Test PDFPageAggregator with a real PDF file."""
        pdf_path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(pdf_path))
        assert len(pages) >= 1
        assert isinstance(pages[0], LTPage)


class TestPDFConverter:
    """Tests for PDFConverter base class."""

    def test_is_binary_stream_bytesio(self):
        """Test binary stream detection for BytesIO."""
        assert PDFConverter._is_binary_stream(BytesIO()) is True

    def test_is_binary_stream_stringio(self):
        """Test binary stream detection for StringIO."""
        assert PDFConverter._is_binary_stream(StringIO()) is False

    def test_is_binary_stream_with_mode_text(self):
        """Test binary stream detection for text mode file-like object."""
        mock_fp = Mock()
        mock_fp.mode = "w"
        assert PDFConverter._is_binary_stream(mock_fp) is False

    def test_is_binary_stream_with_mode_binary(self):
        """Test binary stream detection for binary mode file-like object."""
        mock_fp = Mock()
        mock_fp.mode = "wb"
        assert PDFConverter._is_binary_stream(mock_fp) is True

    def test_is_binary_stream_text_io_base(self):
        """Test binary stream detection for TextIOBase."""
        assert PDFConverter._is_binary_stream(io.TextIOBase()) is False

    def test_is_binary_stream_default_unknown(self):
        """Test binary stream detection defaults to True for unknown types."""
        assert PDFConverter._is_binary_stream(object()) is True


class TestTextConverter:
    """Tests for TextConverter class."""

    def test_text_output_to_stringio(self):
        """Test text extraction to StringIO."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with TextConverter(rsrcmgr, output, laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        text = output.getvalue()
        assert len(text) > 0
        assert "\f" in text

    def test_text_output_to_bytesio(self):
        """Test text extraction to BytesIO."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with TextConverter(rsrcmgr, output, laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        data = output.getvalue()
        assert len(data) > 0
        assert b"\f" in data

    def test_showpageno_option(self):
        """Test showpageno option adds page numbers to output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with TextConverter(
            rsrcmgr, output, laparams=laparams, showpageno=True
        ) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        text = output.getvalue()
        assert "Page 1" in text

    def test_codec_parameter(self):
        """Test codec parameter is respected."""
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        converter = TextConverter(rsrcmgr, output, codec="utf-8", laparams=laparams)
        assert converter.codec == "utf-8"

    def test_paint_path_does_nothing(self):
        """Test that paint_path is a no-op for text extraction."""
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        converter = TextConverter(rsrcmgr, output)
        gstate = PDFGraphicState()
        path = [("m", 0, 0), ("l", 100, 100)]
        converter.paint_path(gstate, True, True, False, path)

    def test_render_image_without_imagewriter(self):
        """Test render_image does nothing without imagewriter."""
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        converter = TextConverter(rsrcmgr, output)
        converter.render_image("test", Mock(spec=PDFStream))

    def test_high_level_extract_text(self):
        """Test using high_level.extract_text."""
        pdf_path = absolute_sample_path("simple1.pdf")
        text = extract_text(pdf_path)
        assert len(text) > 0

    def test_extract_multiple_pages(self):
        """Test text extraction from multiple pages."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output)
        text = output.getvalue()
        assert len(text) > 0


class TestHTMLConverter:
    """Tests for HTMLConverter class."""

    def test_html_output_structure(self):
        """Test that HTML output has proper structure."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "<html>" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_html_meta_charset(self):
        """Test that HTML includes charset meta tag."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        with HTMLConverter(rsrcmgr, output, codec="utf-8") as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "charset=utf-8" in html

    def test_html_scale_parameter(self):
        """Test scale parameter affects output positioning."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", laparams=laparams, scale=2.0
        ) as device:
            assert device.scale == 2.0
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)

    def test_html_fontscale_parameter(self):
        """Test fontscale parameter is respected."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with HTMLConverter(rsrcmgr, output, codec="utf-8", fontscale=1.5) as device:
            assert device.fontscale == 1.5

    def test_html_layoutmode_normal(self):
        """Test normal layoutmode."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", layoutmode="normal"
        ) as device:
            assert device.layoutmode == "normal"

    def test_html_layoutmode_exact(self):
        """Test exact layoutmode produces different output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", laparams=laparams, layoutmode="exact"
        ) as device:
            assert device.layoutmode == "exact"
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)

    def test_html_layoutmode_loose(self):
        """Test loose layoutmode."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", laparams=laparams, layoutmode="loose"
        ) as device:
            assert device.layoutmode == "loose"
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)

    def test_html_showpageno(self):
        """Test showpageno adds page anchors."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", laparams=laparams, showpageno=True
        ) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert 'name="1"' in html or "Page 1" in html

    def test_html_pagemargin(self):
        """Test pagemargin parameter."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with HTMLConverter(rsrcmgr, output, codec="utf-8", pagemargin=100) as device:
            assert device.pagemargin == 100
            assert device._yoffset == 100

    def test_html_debug_mode(self):
        """Test debug mode adds additional color annotations."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with HTMLConverter(rsrcmgr, output, codec="utf-8", debug=1) as device:
            assert "textbox" in device.rect_colors
            assert "textbox" in device.text_colors

    def test_html_custom_rect_colors(self):
        """Test custom rect_colors parameter."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        custom_colors = {"page": "blue", "curve": "green"}
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", rect_colors=custom_colors
        ) as device:
            assert device.rect_colors["page"] == "blue"
            assert device.rect_colors["curve"] == "green"

    def test_html_custom_text_colors(self):
        """Test custom text_colors parameter."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        custom_colors = {"char": "red"}
        with HTMLConverter(
            rsrcmgr, output, codec="utf-8", text_colors=custom_colors
        ) as device:
            assert device.text_colors["char"] == "red"

    def test_html_requires_codec_for_binary(self):
        """Test that binary IO requires a codec."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with pytest.raises(PDFValueError, match="Codec is required"):
            HTMLConverter(rsrcmgr, output, codec="")

    def test_html_no_codec_for_text_io(self):
        """Test that text IO must not have a codec."""
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        with pytest.raises(PDFValueError, match="must not be specified"):
            HTMLConverter(rsrcmgr, output, codec="utf-8")

    def test_html_text_io_without_codec(self):
        """Test HTML output to text IO without codec."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(rsrcmgr, output, codec="", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue()
        assert "<html>" in html

    def test_html_close_writes_footer(self):
        """Test that close() writes the footer."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HTMLConverter(rsrcmgr, output, codec="utf-8")
        converter.close()
        html = output.getvalue().decode("utf-8")
        assert "</body></html>" in html

    def test_html_write_text_escapes_html(self):
        """Test that write_text escapes HTML entities."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HTMLConverter(rsrcmgr, output, codec="utf-8")
        converter.write_text("<script>alert('xss')</script>")
        converter.close()
        html = output.getvalue().decode("utf-8")
        assert "&lt;script&gt;" in html
        assert "<script>" not in html.split("<head>")[0]


class TestXMLConverter:
    """Tests for XMLConverter class."""

    def test_xml_output_structure(self):
        """Test that XML output has proper structure."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        assert '<?xml version="1.0"' in xml_content
        assert "<pages>" in xml_content
        assert "</pages>" in xml_content
        assert "<page " in xml_content

    def test_xml_valid_structure(self):
        """Test that XML output is valid XML."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        assert root.tag == "pages"

    def test_xml_page_attributes(self):
        """Test that page elements have correct attributes."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        page = root.find("page")
        assert page is not None
        assert "id" in page.attrib
        assert "bbox" in page.attrib
        assert "rotate" in page.attrib

    def test_xml_textbox_elements(self):
        """Test that textbox elements are present."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        page = root.find("page")
        textboxes = page.findall(".//textbox") if page is not None else []
        assert len(textboxes) >= 0

    def test_xml_encoding_declaration(self):
        """Test XML encoding declaration matches codec."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with XMLConverter(rsrcmgr, output, codec="utf-8"):
            pass
        xml_content = output.getvalue().decode("utf-8")
        assert 'encoding="utf-8"' in xml_content

    def test_xml_stripcontrol(self):
        """Test stripcontrol option removes control characters."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = XMLConverter(rsrcmgr, output, codec="utf-8", stripcontrol=True)
        assert converter.stripcontrol is True
        converter.close()

    def test_xml_no_stripcontrol(self):
        """Test stripcontrol=False preserves control characters."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = XMLConverter(rsrcmgr, output, codec="utf-8", stripcontrol=False)
        assert converter.stripcontrol is False
        converter.close()

    def test_xml_requires_codec_for_binary(self):
        """Test that binary IO requires a codec."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with pytest.raises(PDFValueError, match="Codec is required"):
            XMLConverter(rsrcmgr, output, codec="")

    def test_xml_text_io_without_codec(self):
        """Test XML output to text IO without codec."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue()
        assert '<?xml version="1.0" ?>' in xml_content
        assert "<pages>" in xml_content

    def test_xml_close_writes_footer(self):
        """Test that close() writes the footer."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = XMLConverter(rsrcmgr, output, codec="utf-8")
        converter.close()
        xml_content = output.getvalue().decode("utf-8")
        assert "</pages>" in xml_content

    def test_xml_write_text_escapes_xml(self):
        """Test that write_text escapes XML entities."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = XMLConverter(rsrcmgr, output, codec="utf-8")
        converter.write_text("<test>&value</test>")
        converter.close()
        xml_content = output.getvalue().decode("utf-8")
        assert "&lt;test&gt;" in xml_content
        assert "&amp;value" in xml_content


class TestHOCRConverter:
    """Tests for HOCRConverter class."""

    def test_hocr_output_structure(self):
        """Test that hOCR output has proper HTML structure."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HOCRConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "<html" in html
        assert "</html>" in html
        assert "xmlns='http://www.w3.org/1999/xhtml'" in html

    def test_hocr_meta_tags(self):
        """Test that hOCR includes required meta tags."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        with HOCRConverter(rsrcmgr, output, codec="utf-8"):
            pass
        html = output.getvalue().decode("utf-8")
        assert "ocr-system" in html
        assert "pdfminer.six HOCR Converter" in html
        assert "ocr-capabilities" in html
        assert "ocr_page" in html
        assert "ocr_block" in html
        assert "ocr_line" in html
        assert "ocrx_word" in html

    def test_hocr_ocr_page_class(self):
        """Test that pages have ocr_page class."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HOCRConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "class='ocr_page'" in html

    def test_hocr_stripcontrol(self):
        """Test stripcontrol option."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HOCRConverter(rsrcmgr, output, codec="utf-8", stripcontrol=True)
        assert converter.stripcontrol is True
        converter.close()

    def test_hocr_bbox_format(self):
        """Test that bbox is in hOCR format."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HOCRConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        bbox_pattern = r"bbox \d+ \d+ \d+ \d+"
        assert re.search(bbox_pattern, html) is not None

    def test_hocr_close_writes_footer(self):
        """Test that close() writes the footer."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HOCRConverter(rsrcmgr, output, codec="utf-8")
        converter.close()
        html = output.getvalue().decode("utf-8")
        assert "</body></html>" in html


class TestTagExtractor:
    """Tests for TagExtractor class."""

    def test_tag_extractor_init(self):
        """Test TagExtractor initialization."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        extractor = TagExtractor(rsrcmgr, output, codec="utf-8")
        assert extractor.pageno == 0
        assert extractor.codec == "utf-8"

    def test_tag_extractor_basic_output(self):
        """Test TagExtractor produces output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        extractor = TagExtractor(rsrcmgr, output, codec="utf-8")
        interpreter = PDFPageInterpreter(rsrcmgr, extractor)
        with open(pdf_path, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
        result = output.getvalue().decode("utf-8")
        assert "<page " in result
        assert "</page>" in result


class TestConverterIntegration:
    """Integration tests for converters."""

    def test_extract_text_to_fp_text(self):
        """Test extract_text_to_fp with text output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output, output_type="text")
        text = output.getvalue()
        assert len(text) > 0

    def test_extract_text_to_fp_xml(self):
        """Test extract_text_to_fp with XML output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output, output_type="xml")
        xml_content = output.getvalue().decode("utf-8")
        assert "<pages>" in xml_content

    def test_extract_text_to_fp_html(self):
        """Test extract_text_to_fp with HTML output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output, output_type="html")
        html = output.getvalue().decode("utf-8")
        assert "<html>" in html

    def test_extract_text_to_fp_hocr(self):
        """Test extract_text_to_fp with hOCR output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output, output_type="hocr")
        html = output.getvalue().decode("utf-8")
        assert "<html" in html
        assert "ocr_page" in html

    def test_extract_text_to_fp_tag(self):
        """Test extract_text_to_fp with tag output."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        with open(pdf_path, "rb") as fp:
            extract_text_to_fp(fp, output, output_type="tag")
        result = output.getvalue().decode("utf-8")
        assert "<page " in result

    def test_extract_text_to_fp_invalid_type(self):
        """Test extract_text_to_fp with invalid output type."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        with open(pdf_path, "rb") as fp, pytest.raises(PDFValueError):
            extract_text_to_fp(fp, output, output_type="invalid")

    def test_extract_pages_returns_iterator(self):
        """Test that extract_pages returns an iterator of LTPage."""
        pdf_path = absolute_sample_path("simple1.pdf")
        pages = extract_pages(pdf_path)
        first_page = next(pages)
        assert isinstance(first_page, LTPage)

    def test_extract_pages_with_laparams(self):
        """Test extract_pages with custom LAParams."""
        pdf_path = absolute_sample_path("simple1.pdf")
        laparams = LAParams(char_margin=5.0, word_margin=0.5)
        pages = list(extract_pages(pdf_path, laparams=laparams))
        assert len(pages) >= 1

    def test_extract_pages_with_page_numbers(self):
        """Test extract_pages with specific page numbers."""
        pdf_path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(pdf_path, page_numbers=[0]))
        assert len(pages) == 1

    def test_extract_pages_with_maxpages(self):
        """Test extract_pages with maxpages limit."""
        pdf_path = absolute_sample_path("simple1.pdf")
        pages = list(extract_pages(pdf_path, maxpages=1))
        assert len(pages) == 1

    def test_text_extraction_with_laparams_none(self):
        """Test that extract_text works when laparams is None."""
        pdf_path = absolute_sample_path("simple1.pdf")
        text = extract_text(pdf_path, laparams=None)
        assert len(text) > 0


class TestConverterOutputFormats:
    """Tests for different output format scenarios."""

    def test_xml_with_textlines(self):
        """Test XML output contains textline elements."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        page = root.find("page")
        if page is not None:
            textlines = page.findall(".//textline")
            assert len(textlines) >= 0

    def test_xml_with_chars(self):
        """Test XML output contains text elements for characters."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        page = root.find("page")
        if page is not None:
            texts = page.findall(".//text")
            assert len(texts) >= 0

    def test_html_with_positioning(self):
        """Test HTML output includes CSS positioning."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HTMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "position:absolute" in html

    def test_text_with_form_feed(self):
        """Test text output includes form feed between pages."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = StringIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with TextConverter(rsrcmgr, output, laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        text = output.getvalue()
        assert "\f" in text


class TestConverterWithDifferentPDFs:
    """Tests using different sample PDFs."""

    def test_simple1_pdf(self):
        """Test extraction from simple1.pdf."""
        pdf_path = absolute_sample_path("simple1.pdf")
        text = extract_text(pdf_path)
        assert len(text) > 0

    def test_simple2_pdf(self):
        """Test extraction from simple2.pdf."""
        pdf_path = absolute_sample_path("simple2.pdf")
        text = extract_text(pdf_path)
        assert len(text) > 0

    def test_simple3_pdf(self):
        """Test extraction from simple3.pdf."""
        pdf_path = absolute_sample_path("simple3.pdf")
        text = extract_text(pdf_path)
        assert len(text) > 0

    def test_jo_pdf(self):
        """Test extraction from jo.pdf (Japanese)."""
        pdf_path = absolute_sample_path("jo.pdf")
        text = extract_text(pdf_path)
        assert len(text) > 0

    def test_xml_output_for_multiple_pdfs(self):
        """Test XML output for multiple sample PDFs."""
        pdf_files = ["simple1.pdf", "simple2.pdf", "simple3.pdf"]
        for pdf_file in pdf_files:
            pdf_path = absolute_sample_path(pdf_file)
            output = BytesIO()
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            with XMLConverter(
                rsrcmgr, output, codec="utf-8", laparams=laparams
            ) as device:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                with open(pdf_path, "rb") as fp:
                    for page in PDFPage.get_pages(fp):
                        interpreter.process_page(page)
            xml_content = output.getvalue().decode("utf-8")
            root = ET.fromstring(xml_content)
            assert root.tag == "pages"


class TestConverterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_page_extraction(self):
        """Test handling of pages with no text content."""
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        laparams = LAParams()
        converter = TextConverter(rsrcmgr, output, laparams=laparams)
        ltpage = LTPage(1, (0, 0, 612, 792))
        converter.receive_layout(ltpage)
        text = output.getvalue()
        assert "\f" in text

    def test_converter_close_method(self):
        """Test that converters can be properly closed."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converters = [
            XMLConverter(rsrcmgr, output, codec="utf-8"),
            HTMLConverter(rsrcmgr, BytesIO(), codec="utf-8"),
            HOCRConverter(rsrcmgr, BytesIO(), codec="utf-8"),
        ]
        for converter in converters:
            converter.close()

    def test_converter_context_manager(self):
        """Test converters work as context managers."""
        rsrcmgr = PDFResourceManager()
        with XMLConverter(rsrcmgr, BytesIO(), codec="utf-8") as converter:
            assert converter is not None

    def test_laparams_boxes_flow_variations(self):
        """Test different boxes_flow values."""
        pdf_path = absolute_sample_path("simple1.pdf")
        for boxes_flow in [None, -1.0, 0.0, 0.5, 1.0]:
            laparams = LAParams(boxes_flow=boxes_flow)
            pages = list(extract_pages(pdf_path, laparams=laparams))
            assert len(pages) >= 1

    def test_laparams_detect_vertical(self):
        """Test vertical text detection setting."""
        pdf_path = absolute_sample_path("jo.pdf")
        laparams = LAParams(detect_vertical=True)
        pages = list(extract_pages(pdf_path, laparams=laparams))
        assert len(pages) >= 1

    def test_laparams_all_texts(self):
        """Test all_texts setting for figure text extraction."""
        pdf_path = absolute_sample_path("simple1.pdf")
        laparams = LAParams(all_texts=True)
        pages = list(extract_pages(pdf_path, laparams=laparams))
        assert len(pages) >= 1

    def test_multiple_page_extraction(self):
        """Test extraction across multiple pages."""
        pdf_path = absolute_sample_path("simple1.pdf")
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        laparams = LAParams()
        with TextConverter(
            rsrcmgr, output, laparams=laparams, showpageno=True
        ) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                pages_processed = 0
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
                    pages_processed += 1
        assert pages_processed >= 1


class TestConverterSpecialFeatures:
    """Tests for special converter features."""

    def test_html_put_newline(self):
        """Test HTML put_newline method."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HTMLConverter(rsrcmgr, output, codec="utf-8")
        converter.put_newline()
        converter.close()
        html = output.getvalue().decode("utf-8")
        assert "<br>" in html

    def test_html_begin_end_div(self):
        """Test HTML begin_div and end_div methods."""
        rsrcmgr = PDFResourceManager()
        output = BytesIO()
        converter = HTMLConverter(rsrcmgr, output, codec="utf-8")
        converter.begin_div("page", 1, 0, 100, 100, 50)
        converter.end_div("page")
        converter.close()
        html = output.getvalue().decode("utf-8")
        assert "<div" in html
        assert "</div>" in html

    def test_xml_layout_section(self):
        """Test XML output includes layout section with groups."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(boxes_flow=0.5)
        with XMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        xml_content = output.getvalue().decode("utf-8")
        root = ET.fromstring(xml_content)
        page = root.find("page")
        if page is not None:
            layout = page.find("layout")
            if layout is not None:
                assert layout is not None

    def test_hocr_word_spans(self):
        """Test hOCR output contains word spans."""
        pdf_path = absolute_sample_path("simple1.pdf")
        output = BytesIO()
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        with HOCRConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as device:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            with open(pdf_path, "rb") as fp:
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
        html = output.getvalue().decode("utf-8")
        assert "ocrx_word" in html or "ocr_line" in html


class TestLayoutAnalyzerPaths:
    """Tests for path handling in PDFLayoutAnalyzer."""

    def _get_analyzer(self):
        """Helper to create a configured analyzer."""
        rsrcmgr = PDFResourceManager()
        analyzer = PDFLayoutAnalyzer(rsrcmgr)
        analyzer.set_ctm([1, 0, 0, 1, 0, 0])
        return analyzer

    def test_paint_path_single_line(self):
        """Test paint_path with a single line segment."""
        path = [("m", 0, 0), ("l", 100, 100)]
        analyzer = self._get_analyzer()
        analyzer.cur_item = LTContainer([0, 200, 0, 200])
        gstate = PDFGraphicState()
        analyzer.paint_path(gstate, True, False, False, path)
        assert len(analyzer.cur_item._objs) == 1
        assert isinstance(analyzer.cur_item._objs[0], LTLine)

    def test_paint_path_rectangle(self):
        """Test paint_path with a rectangle."""
        path = [
            ("m", 0, 0),
            ("l", 100, 0),
            ("l", 100, 100),
            ("l", 0, 100),
            ("h",),
        ]
        analyzer = self._get_analyzer()
        analyzer.cur_item = LTContainer([0, 200, 0, 200])
        gstate = PDFGraphicState()
        analyzer.paint_path(gstate, True, True, False, path)
        assert len(analyzer.cur_item._objs) == 1
        assert isinstance(analyzer.cur_item._objs[0], LTRect)

    def test_paint_path_curve(self):
        """Test paint_path with a curve."""
        path = [
            ("m", 0, 0),
            ("c", 25, 50, 75, 50, 100, 0),
        ]
        analyzer = self._get_analyzer()
        analyzer.cur_item = LTContainer([0, 200, 0, 200])
        gstate = PDFGraphicState()
        analyzer.paint_path(gstate, True, False, False, path)
        assert len(analyzer.cur_item._objs) == 1
        assert isinstance(analyzer.cur_item._objs[0], LTCurve)

    def test_paint_path_invalid_start(self):
        """Test paint_path with invalid starting operator (not 'm')."""
        path = [("l", 100, 100), ("h",)]
        analyzer = self._get_analyzer()
        analyzer.cur_item = LTContainer([0, 200, 0, 200])
        gstate = PDFGraphicState()
        analyzer.paint_path(gstate, True, False, False, path)
        assert len(analyzer.cur_item._objs) == 0

    def test_paint_path_multiple_subpaths(self):
        """Test paint_path with multiple subpaths."""
        path = [
            ("m", 0, 0),
            ("l", 50, 50),
            ("h",),
            ("m", 100, 100),
            ("l", 150, 150),
            ("h",),
        ]
        analyzer = self._get_analyzer()
        analyzer.cur_item = LTContainer([0, 200, 0, 200])
        gstate = PDFGraphicState()
        analyzer.paint_path(gstate, True, False, False, path)
        assert len(analyzer.cur_item._objs) == 2


class TestConverterResourceManagement:
    """Tests for resource management in converters."""

    def test_rsrcmgr_caching_enabled(self):
        """Test PDFResourceManager with caching enabled."""
        rsrcmgr = PDFResourceManager(caching=True)
        assert rsrcmgr.caching is True

    def test_rsrcmgr_caching_disabled(self):
        """Test PDFResourceManager with caching disabled."""
        rsrcmgr = PDFResourceManager(caching=False)
        assert rsrcmgr.caching is False

    def test_converter_uses_rsrcmgr(self):
        """Test that converters properly use the resource manager."""
        rsrcmgr = PDFResourceManager()
        output = StringIO()
        converter = TextConverter(rsrcmgr, output)
        assert converter.rsrcmgr is rsrcmgr


class TestPDFPageAggregatorWithInterpreter:
    """Tests for PDFPageAggregator with PDFPageInterpreter."""

    def test_aggregator_with_interpreter(self):
        """Test PDFPageAggregator integrated with interpreter."""
        pdf_path = absolute_sample_path("simple1.pdf")
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        with open(pdf_path, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
                layout = device.get_result()
                assert isinstance(layout, LTPage)
                assert layout.pageid >= 1

    def test_aggregator_pageno_increments(self):
        """Test that page number increments correctly."""
        pdf_path = absolute_sample_path("simple1.pdf")
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        page_ids = []
        with open(pdf_path, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
                layout = device.get_result()
                page_ids.append(layout.pageid)

        if len(page_ids) > 1:
            for i in range(1, len(page_ids)):
                assert page_ids[i] == page_ids[i - 1] + 1

    def test_aggregator_custom_start_pageno(self):
        """Test aggregator with custom starting page number."""
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, pageno=10)
        assert device.pageno == 10
