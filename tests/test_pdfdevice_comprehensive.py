"""Comprehensive tests for pdfminer/pdfdevice.py

Tests PDFDevice base class, PDFTextDevice, and TagExtractor.
"""

import io
from typing import Any
from unittest.mock import MagicMock

import pytest

from pdfminer.pdfcolor import PREDEFINED_COLORSPACE, PDFColorSpace
from pdfminer.pdfdevice import PDFDevice, PDFTextDevice, TagExtractor
from pdfminer.pdffont import PDFFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager, PDFTextState
from pdfminer.pdftypes import PDFStream
from pdfminer.psparser import LIT
from pdfminer.utils import Matrix, Rect


class MockPDFFont(PDFFont):
    """Mock PDFFont for testing purposes."""

    def __init__(
        self,
        vertical: bool = False,
        multibyte: bool = False,
        char_width: float = 1000.0,
        cid_to_unicode: dict[int, str] | None = None,
    ) -> None:
        self._vertical = vertical
        self._multibyte = multibyte
        self._char_width = char_width
        self._cid_to_unicode = cid_to_unicode or {}
        self.descriptor: dict[str, Any] = {}
        self.widths: dict[int | str, float] = {}
        self.default_width = 1000.0
        self.hscale = 0.001
        self.vscale = 0.001
        self.fontname = "MockFont"

    def is_vertical(self) -> bool:
        return self._vertical

    def is_multibyte(self) -> bool:
        return self._multibyte

    def decode(self, data: bytes) -> list[int]:
        return list(data)

    def char_width(self, cid: int) -> float:
        return self._char_width * self.hscale

    def to_unichr(self, cid: int) -> str:
        if cid in self._cid_to_unicode:
            return self._cid_to_unicode[cid]
        raise PDFUnicodeNotDefined(f"CID {cid} not defined")


class MockPDFPage:
    """Mock PDFPage for testing purposes."""

    def __init__(
        self,
        mediabox: Rect = (0.0, 0.0, 612.0, 792.0),
        rotate: int = 0,
    ) -> None:
        self.mediabox = mediabox
        self.rotate = rotate
        self.pageid = 1
        self.attrs: dict[str, Any] = {}


class TestPDFDevice:
    """Tests for the PDFDevice base class."""

    @pytest.fixture
    def rsrcmgr(self) -> PDFResourceManager:
        return PDFResourceManager()

    @pytest.fixture
    def device(self, rsrcmgr: PDFResourceManager) -> PDFDevice:
        return PDFDevice(rsrcmgr)

    def test_init(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFDevice(rsrcmgr)
        assert device.rsrcmgr is rsrcmgr
        assert device.ctm is None

    def test_repr(self, device: PDFDevice) -> None:
        assert repr(device) == "<PDFDevice>"

    def test_context_manager(self, rsrcmgr: PDFResourceManager) -> None:
        with PDFDevice(rsrcmgr) as device:
            assert isinstance(device, PDFDevice)

    def test_close(self, device: PDFDevice) -> None:
        device.close()

    def test_set_ctm(self, device: PDFDevice) -> None:
        ctm: Matrix = (1, 0, 0, 1, 100, 200)
        device.set_ctm(ctm)
        assert device.ctm == ctm

    def test_begin_tag(self, device: PDFDevice) -> None:
        tag = LIT("P")
        device.begin_tag(tag)

    def test_begin_tag_with_props(self, device: PDFDevice) -> None:
        tag = LIT("P")
        props: dict[str, Any] = {"id": "test"}
        device.begin_tag(tag, props)

    def test_end_tag(self, device: PDFDevice) -> None:
        device.end_tag()

    def test_do_tag(self, device: PDFDevice) -> None:
        tag = LIT("BR")
        device.do_tag(tag)

    def test_do_tag_with_props(self, device: PDFDevice) -> None:
        tag = LIT("BR")
        props: dict[str, Any] = {"class": "break"}
        device.do_tag(tag, props)

    def test_begin_page(self, device: PDFDevice) -> None:
        page = MockPDFPage()
        ctm: Matrix = (1, 0, 0, 1, 0, 0)
        device.begin_page(page, ctm)  # type: ignore[arg-type]

    def test_end_page(self, device: PDFDevice) -> None:
        page = MockPDFPage()
        device.end_page(page)  # type: ignore[arg-type]

    def test_begin_figure(self, device: PDFDevice) -> None:
        bbox: Rect = (0.0, 0.0, 100.0, 100.0)
        matrix: Matrix = (1, 0, 0, 1, 0, 0)
        device.begin_figure("Figure1", bbox, matrix)

    def test_end_figure(self, device: PDFDevice) -> None:
        device.end_figure("Figure1")

    def test_paint_path(self, device: PDFDevice) -> None:
        graphicstate = PDFGraphicState()
        path: list[tuple[str, ...] | tuple[str, float, float]] = [
            ("m", 0.0, 0.0),
            ("l", 100.0, 100.0),
            ("h",),
        ]
        device.paint_path(
            graphicstate,
            stroke=True,
            fill=False,
            evenodd=False,
            path=path,  # type: ignore[arg-type]
        )

    def test_paint_path_with_fill(self, device: PDFDevice) -> None:
        graphicstate = PDFGraphicState()
        path: list[tuple[str, ...] | tuple[str, float, float]] = [
            ("m", 0.0, 0.0),
            ("l", 100.0, 0.0),
            ("l", 100.0, 100.0),
            ("l", 0.0, 100.0),
            ("h",),
        ]
        device.paint_path(
            graphicstate,
            stroke=False,
            fill=True,
            evenodd=True,
            path=path,  # type: ignore[arg-type]
        )

    def test_render_image(self, device: PDFDevice) -> None:
        stream = MagicMock(spec=PDFStream)
        device.render_image("Image1", stream)

    def test_render_string(self, device: PDFDevice) -> None:
        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        seq = [b"Hello"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()
        device.render_string(textstate, seq, ncs, graphicstate)


class TestPDFTextDevice:
    """Tests for the PDFTextDevice class."""

    @pytest.fixture
    def rsrcmgr(self) -> PDFResourceManager:
        return PDFResourceManager()

    @pytest.fixture
    def device(self, rsrcmgr: PDFResourceManager) -> PDFTextDevice:
        return PDFTextDevice(rsrcmgr)

    def test_inheritance(self, device: PDFTextDevice) -> None:
        assert isinstance(device, PDFDevice)

    def test_render_char_default(self, device: PDFTextDevice) -> None:
        matrix: Matrix = (1, 0, 0, 1, 0, 0)
        font = MockPDFFont()
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        result = device.render_char(
            matrix=matrix,
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            cid=65,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        assert result == 0

    def test_render_string_horizontal(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_vertical(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=True)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_with_numeric_adjustments(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq: list[int | float | bytes] = [b"A", -100.0, b"B", -50, b"C"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_with_charspace(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 2.0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"Hello"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_with_wordspace(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 5.0
        textstate.rise = 0

        seq = [b"Hello World"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_multibyte_font_ignores_wordspace(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False, multibyte=True)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 5.0
        textstate.rise = 0

        seq = [b"Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_with_scaling(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 150
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"Scaled"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_with_rise(self, device: PDFTextDevice) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 5.0

        seq = [b"Raised"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_render_string_horizontal_invalid_object(
        self, device: PDFTextDevice, caplog: pytest.LogCaptureFixture
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq: list[Any] = [b"A", None, b"B"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)  # type: ignore[arg-type]
        assert "Cannot render horizontal string" in caplog.text

    def test_render_string_vertical_invalid_object(
        self, device: PDFTextDevice, caplog: pytest.LogCaptureFixture
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=True)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq: list[Any] = [b"A", {"invalid": "dict"}, b"B"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)  # type: ignore[arg-type]
        assert "Cannot render vertical string" in caplog.text

    def test_render_string_horizontal_returns_new_linematrix(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False, char_width=500.0)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0
        textstate.linematrix = (0.0, 0.0)

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        assert textstate.linematrix[0] != 0.0 or textstate.linematrix[1] == 0.0

    def test_render_string_vertical_returns_new_linematrix(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=True, char_width=500.0)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0
        textstate.linematrix = (0.0, 0.0)

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        assert textstate.linematrix[0] == 0.0 or textstate.linematrix[1] != 0.0

    def test_render_string_horizontal_position_adjustment(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0
        initial_x = 100.0
        textstate.linematrix = (initial_x, 0.0)

        seq = [-1000.0]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)
        assert textstate.linematrix[0] > initial_x

    def test_render_string_vertical_position_adjustment(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=True)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0
        initial_y = 100.0
        textstate.linematrix = (0.0, initial_y)

        seq = [-1000.0]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)
        assert textstate.linematrix[1] > initial_y

    def test_render_string_space_char_adds_wordspace(
        self, device: PDFTextDevice
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 10.0
        textstate.rise = 0
        textstate.linematrix = (0.0, 0.0)

        seq = [b" "]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)
        assert textstate.linematrix[0] >= 10.0


class TestPDFTextDeviceSubclass:
    """Tests for a subclass of PDFTextDevice that implements render_char."""

    class TrackingTextDevice(PDFTextDevice):
        """Subclass that tracks render_char calls."""

        def __init__(self, rsrcmgr: PDFResourceManager) -> None:
            super().__init__(rsrcmgr)
            self.rendered_chars: list[tuple[Matrix, int]] = []

        def render_char(
            self,
            matrix: Matrix,
            font: PDFFont,
            fontsize: float,
            scaling: float,
            rise: float,
            cid: int,
            ncs: PDFColorSpace,
            graphicstate: "PDFGraphicState",
        ) -> float:
            self.rendered_chars.append((matrix, cid))
            return font.char_width(cid)

    @pytest.fixture
    def rsrcmgr(self) -> PDFResourceManager:
        return PDFResourceManager()

    @pytest.fixture
    def device(
        self, rsrcmgr: PDFResourceManager
    ) -> "TestPDFTextDeviceSubclass.TrackingTextDevice":
        return self.TrackingTextDevice(rsrcmgr)

    def test_render_char_called_for_each_character(
        self, device: "TestPDFTextDeviceSubclass.TrackingTextDevice"
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False, char_width=500.0)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        assert len(device.rendered_chars) == 3
        assert device.rendered_chars[0][1] == ord("A")
        assert device.rendered_chars[1][1] == ord("B")
        assert device.rendered_chars[2][1] == ord("C")

    def test_render_char_receives_translated_matrix(
        self, device: "TestPDFTextDeviceSubclass.TrackingTextDevice"
    ) -> None:
        device.set_ctm((1, 0, 0, 1, 100, 200))

        textstate = PDFTextState()
        textstate.font = MockPDFFont(vertical=False, char_width=500.0)
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.charspace = 0
        textstate.wordspace = 0
        textstate.rise = 0

        seq = [b"X"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        assert len(device.rendered_chars) == 1
        matrix = device.rendered_chars[0][0]
        assert matrix[4] == 100
        assert matrix[5] == 200


class TestTagExtractor:
    """Tests for the TagExtractor class."""

    @pytest.fixture
    def rsrcmgr(self) -> PDFResourceManager:
        return PDFResourceManager()

    @pytest.fixture
    def outfp(self) -> io.BytesIO:
        return io.BytesIO()

    @pytest.fixture
    def device(
        self, rsrcmgr: PDFResourceManager, outfp: io.BytesIO
    ) -> TagExtractor:
        return TagExtractor(rsrcmgr, outfp)

    def test_init(
        self, rsrcmgr: PDFResourceManager, outfp: io.BytesIO
    ) -> None:
        device = TagExtractor(rsrcmgr, outfp)
        assert device.rsrcmgr is rsrcmgr
        assert device.outfp is outfp
        assert device.codec == "utf-8"
        assert device.pageno == 0
        assert device._stack == []

    def test_init_custom_codec(
        self, rsrcmgr: PDFResourceManager, outfp: io.BytesIO
    ) -> None:
        device = TagExtractor(rsrcmgr, outfp, codec="latin-1")
        assert device.codec == "latin-1"

    def test_inheritance(self, device: TagExtractor) -> None:
        assert isinstance(device, PDFDevice)

    def test_begin_page(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        page = MockPDFPage(mediabox=(0.0, 0.0, 612.0, 792.0), rotate=0)
        ctm: Matrix = (1, 0, 0, 1, 0, 0)

        device.begin_page(page, ctm)  # type: ignore[arg-type]

        output = outfp.getvalue().decode("utf-8")
        assert '<page id="0"' in output
        assert 'bbox="0.000,0.000,612.000,792.000"' in output
        assert 'rotate="0">' in output

    def test_end_page(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        device.end_page(MockPDFPage())  # type: ignore[arg-type]

        output = outfp.getvalue().decode("utf-8")
        assert output == "</page>\n"
        assert device.pageno == 1

    def test_multiple_pages(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        page = MockPDFPage()
        ctm: Matrix = (1, 0, 0, 1, 0, 0)

        device.begin_page(page, ctm)  # type: ignore[arg-type]
        device.end_page(page)  # type: ignore[arg-type]
        device.begin_page(page, ctm)  # type: ignore[arg-type]
        device.end_page(page)  # type: ignore[arg-type]

        assert device.pageno == 2

    def test_begin_tag_simple(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        tag = LIT("P")
        device.begin_tag(tag)

        output = outfp.getvalue().decode("utf-8")
        assert output == "<P>"
        assert len(device._stack) == 1
        assert device._stack[0] is tag

    def test_begin_tag_with_props(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        tag = LIT("Span")
        props = {"id": "test123", "class": "highlight"}
        device.begin_tag(tag, props)

        output = outfp.getvalue().decode("utf-8")
        assert "<Span " in output
        assert 'class="highlight"' in output
        assert 'id="test123"' in output
        assert ">" in output

    def test_end_tag(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        tag = LIT("P")
        device.begin_tag(tag)
        outfp.seek(0)
        outfp.truncate()

        device.end_tag()

        output = outfp.getvalue().decode("utf-8")
        assert output == "</P>"
        assert len(device._stack) == 0

    def test_end_tag_empty_stack_raises(self, device: TagExtractor) -> None:
        with pytest.raises(AssertionError):
            device.end_tag()

    def test_nested_tags(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        tag1 = LIT("P")
        tag2 = LIT("Span")

        device.begin_tag(tag1)
        device.begin_tag(tag2)
        device.end_tag()
        device.end_tag()

        output = outfp.getvalue().decode("utf-8")
        assert output == "<P><Span></Span></P>"

    def test_do_tag(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        tag = LIT("BR")
        device.do_tag(tag)

        output = outfp.getvalue().decode("utf-8")
        assert output == "<BR>"
        assert len(device._stack) == 0

    def test_do_tag_with_props(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        tag = LIT("Link")
        props = {"href": "http://example.com"}
        device.do_tag(tag, props)

        output = outfp.getvalue().decode("utf-8")
        assert "<Link " in output
        assert 'href="http://example.com"' in output

    def test_render_string_basic(self, device: TagExtractor, outfp: io.BytesIO) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "Hello"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq = [b"Hello"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("utf-8")
        assert output == "Hello"

    def test_render_string_multiple_sequences(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "Hello World"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq = [b"Hello", b" ", b"World"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("utf-8")
        assert output == "Hello World"

    def test_render_string_ignores_numeric_adjustments(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "AB"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq: list[int | float | bytes] = [b"A", -100.0, b"B"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("utf-8")
        assert output == "AB"

    def test_render_string_skips_undefined_unicode(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {ord("A"): "A", ord("C"): "C"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq = [b"ABC"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("utf-8")
        assert output == "AC"

    def test_render_string_with_str_in_seq(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "Test"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq: list[Any] = ["Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)  # type: ignore[arg-type]

        output = outfp.getvalue().decode("utf-8")
        assert output == "Test"

    def test_render_string_unicode_content(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        textstate = PDFTextState()
        cid_to_unicode = {
            0xE4: "\u00e4",
            0xF6: "\u00f6",
            0xFC: "\u00fc",
        }
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq = [bytes([0xE4, 0xF6, 0xFC])]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("utf-8")
        assert output == "\u00e4\u00f6\u00fc"

    def test_write_with_custom_codec(
        self, rsrcmgr: PDFResourceManager
    ) -> None:
        outfp = io.BytesIO()
        device = TagExtractor(rsrcmgr, outfp, codec="latin-1")

        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "Test"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)

        seq = [b"Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

        output = outfp.getvalue().decode("latin-1")
        assert output == "Test"

    def test_full_document_workflow(
        self, device: TagExtractor, outfp: io.BytesIO
    ) -> None:
        page = MockPDFPage()
        ctm: Matrix = (1, 0, 0, 1, 0, 0)

        device.begin_page(page, ctm)  # type: ignore[arg-type]

        tag_p = LIT("P")
        device.begin_tag(tag_p)

        textstate = PDFTextState()
        cid_to_unicode = {ord(c): c for c in "Hello World"}
        textstate.font = MockPDFFont(cid_to_unicode=cid_to_unicode)
        seq = [b"Hello World"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()
        device.render_string(textstate, seq, ncs, graphicstate)

        device.end_tag()

        device.end_page(page)  # type: ignore[arg-type]

        output = outfp.getvalue().decode("utf-8")
        assert '<page id="0"' in output
        assert "<P>" in output
        assert "Hello World" in output
        assert "</P>" in output
        assert "</page>" in output


class TestPDFGraphicStateIntegration:
    """Tests for PDFGraphicState usage within devices."""

    def test_graphicstate_default_values(self) -> None:
        gs = PDFGraphicState()
        assert gs.linewidth == 0
        assert gs.linecap is None
        assert gs.linejoin is None
        assert gs.miterlimit is None
        assert gs.dash is None
        assert gs.intent is None
        assert gs.flatness is None
        assert gs.scolor == 0
        assert gs.ncolor == 0

    def test_graphicstate_copy(self) -> None:
        gs = PDFGraphicState()
        gs.linewidth = 2.0
        gs.scolor = (1.0, 0.0, 0.0)

        gs_copy = gs.copy()
        assert gs_copy.linewidth == 2.0
        assert gs_copy.scolor == (1.0, 0.0, 0.0)

        gs_copy.linewidth = 5.0
        assert gs.linewidth == 2.0


class TestPDFTextStateIntegration:
    """Tests for PDFTextState usage within devices."""

    def test_textstate_default_values(self) -> None:
        ts = PDFTextState()
        assert ts.font is None
        assert ts.fontsize == 0
        assert ts.charspace == 0
        assert ts.wordspace == 0
        assert ts.scaling == 100
        assert ts.leading == 0
        assert ts.render == 0
        assert ts.rise == 0
        assert ts.matrix == (1, 0, 0, 1, 0, 0)
        assert ts.linematrix == (0, 0)

    def test_textstate_copy(self) -> None:
        ts = PDFTextState()
        ts.fontsize = 12.0
        ts.charspace = 2.0
        ts.linematrix = (100.0, 200.0)

        ts_copy = ts.copy()
        assert ts_copy.fontsize == 12.0
        assert ts_copy.charspace == 2.0
        assert ts_copy.linematrix == (100.0, 200.0)

        ts_copy.fontsize = 24.0
        assert ts.fontsize == 12.0

    def test_textstate_reset(self) -> None:
        ts = PDFTextState()
        ts.matrix = (2, 0, 0, 2, 50, 100)
        ts.linematrix = (500.0, 600.0)

        ts.reset()
        assert ts.matrix == (1, 0, 0, 1, 0, 0)
        assert ts.linematrix == (0, 0)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def rsrcmgr(self) -> PDFResourceManager:
        return PDFResourceManager()

    def test_device_with_identity_ctm(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = 100

        seq = [b"Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_device_with_scaled_ctm(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((2, 0, 0, 2, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = 100

        seq = [b"Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_device_with_rotated_ctm(self, rsrcmgr: PDFResourceManager) -> None:
        import math

        angle = math.pi / 4
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((cos_a, sin_a, -sin_a, cos_a, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = 100

        seq = [b"Test"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_empty_sequence(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.linematrix = (0.0, 0.0)

        seq: list[bytes] = []
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)
        assert textstate.linematrix == (0.0, 0.0)

    def test_empty_bytes_in_sequence(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = 100
        textstate.linematrix = (0.0, 0.0)

        seq = [b""]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)
        assert textstate.linematrix == (0.0, 0.0)

    def test_large_font_size(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 1000.0
        textstate.scaling = 100

        seq = [b"X"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_zero_font_size(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 0.0
        textstate.scaling = 100

        seq = [b"X"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_negative_scaling(self, rsrcmgr: PDFResourceManager) -> None:
        device = PDFTextDevice(rsrcmgr)
        device.set_ctm((1, 0, 0, 1, 0, 0))

        textstate = PDFTextState()
        textstate.font = MockPDFFont()
        textstate.fontsize = 12.0
        textstate.scaling = -100

        seq = [b"X"]
        ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        graphicstate = PDFGraphicState()

        device.render_string(textstate, seq, ncs, graphicstate)

    def test_tag_extractor_special_chars_in_props(
        self, rsrcmgr: PDFResourceManager
    ) -> None:
        outfp = io.BytesIO()
        device = TagExtractor(rsrcmgr, outfp)

        tag = LIT("Span")
        props = {"title": "Test <>&\"'"}
        device.begin_tag(tag, props)

        output = outfp.getvalue().decode("utf-8")
        assert "<Span " in output
        assert ">" in output

    def test_tag_extractor_empty_tag_name(
        self, rsrcmgr: PDFResourceManager
    ) -> None:
        outfp = io.BytesIO()
        device = TagExtractor(rsrcmgr, outfp)

        tag = LIT("")
        device.begin_tag(tag)

        output = outfp.getvalue().decode("utf-8")
        assert "<>" in output
