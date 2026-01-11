"""Comprehensive tests for pdfminer/layout.py

This module tests all layout classes and their functionality:
- LAParams class (all parameters and validation)
- LTItem, LTComponent base classes
- LTCurve, LTLine, LTRect classes
- LTFigure class
- LTChar class
- LTAnno class
- LTText, LTTextContainer classes
- LTTextBox, LTTextLine, LTTextBoxHorizontal, LTTextBoxVertical classes
- LTPage class
- LTLayoutContainer and layout analysis algorithms
- IndexAssigner class
- LTTextGroup classes

Type checking is relaxed for test files since we use mock objects extensively.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pdfminer.pdffont import PDFFont

from pdfminer.layout import (
    IndexAssigner,
    LAParams,
    LTAnno,
    LTChar,
    LTComponent,
    LTContainer,
    LTCurve,
    LTExpandableContainer,
    LTFigure,
    LTItem,
    LTLayoutContainer,
    LTLine,
    LTPage,
    LTRect,
    LTText,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextContainer,
    LTTextGroup,
    LTTextGroupLRTB,
    LTTextGroupTBRL,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfexceptions import PDFTypeError, PDFValueError
from pdfminer.pdfinterp import PDFGraphicState
from pdfminer.utils import INF, Plane


class MockPDFFont:
    """Mock PDFFont for testing LTChar without actual font resources.

    This mock implements the minimal interface needed by LTChar.
    Type checking is intentionally relaxed for test mocks.
    """

    def __init__(
        self,
        fontname: str = "TestFont",
        vertical: bool = False,
        descent: float = -0.2,
    ) -> None:
        self.fontname = fontname
        self._vertical = vertical
        self._descent = descent

    def is_vertical(self) -> bool:
        return self._vertical

    def get_descent(self) -> float:
        return self._descent


def _create_char(
    text: str,
    x: float = 0,
    y: float = 0,
    fontsize: float = 12.0,
    scaling: float = 1.0,
    textwidth: float = 0.6,
    vertical: bool = False,
) -> LTChar:
    """Helper to create LTChar objects for testing."""
    font = MockPDFFont(vertical=vertical)
    ncs = PDFColorSpace("DeviceGray", 1)
    graphicstate = PDFGraphicState()
    textdisp: float | tuple[float, float] = (500.0, 880.0) if vertical else 0.0

    if vertical:
        matrix = (0, 1, -1, 0, x, y)
    else:
        matrix = (1, 0, 0, 1, x, y)

    return LTChar(
        matrix=matrix,
        font=font,  # type: ignore[arg-type]
        fontsize=fontsize,
        scaling=scaling,
        rise=0.0,
        text=text,
        textwidth=textwidth,
        textdisp=textdisp,
        ncs=ncs,
        graphicstate=graphicstate,
    )


class TestLAParams:
    """Tests for LAParams class."""

    def test_default_values(self) -> None:
        laparams = LAParams()
        assert laparams.line_overlap == 0.5
        assert laparams.char_margin == 2.0
        assert laparams.line_margin == 0.5
        assert laparams.word_margin == 0.1
        assert laparams.boxes_flow == 0.5
        assert laparams.detect_vertical is False
        assert laparams.all_texts is False

    def test_custom_values(self) -> None:
        laparams = LAParams(
            line_overlap=0.3,
            char_margin=3.0,
            line_margin=0.7,
            word_margin=0.2,
            boxes_flow=-0.5,
            detect_vertical=True,
            all_texts=True,
        )
        assert laparams.line_overlap == 0.3
        assert laparams.char_margin == 3.0
        assert laparams.line_margin == 0.7
        assert laparams.word_margin == 0.2
        assert laparams.boxes_flow == -0.5
        assert laparams.detect_vertical is True
        assert laparams.all_texts is True

    def test_boxes_flow_none(self) -> None:
        laparams = LAParams(boxes_flow=None)
        assert laparams.boxes_flow is None

    def test_boxes_flow_boundary_values(self) -> None:
        laparams_min = LAParams(boxes_flow=-1.0)
        assert laparams_min.boxes_flow == -1.0

        laparams_max = LAParams(boxes_flow=1.0)
        assert laparams_max.boxes_flow == 1.0

        laparams_zero = LAParams(boxes_flow=0.0)
        assert laparams_zero.boxes_flow == 0.0

    def test_boxes_flow_integer(self) -> None:
        laparams = LAParams(boxes_flow=1)
        assert laparams.boxes_flow == 1

        laparams_neg = LAParams(boxes_flow=-1)
        assert laparams_neg.boxes_flow == -1

    def test_boxes_flow_out_of_range_raises(self) -> None:
        with pytest.raises(PDFValueError):
            LAParams(boxes_flow=1.1)

        with pytest.raises(PDFValueError):
            LAParams(boxes_flow=-1.1)

        with pytest.raises(PDFValueError):
            LAParams(boxes_flow=2.0)

    def test_boxes_flow_invalid_type_raises(self) -> None:
        with pytest.raises(PDFTypeError):
            LAParams(boxes_flow="invalid")  # type: ignore

        with pytest.raises(PDFTypeError):
            LAParams(boxes_flow=[0.5])  # type: ignore

    def test_repr(self) -> None:
        laparams = LAParams()
        repr_str = repr(laparams)
        assert "LAParams" in repr_str
        assert "char_margin=2.0" in repr_str
        assert "line_margin=0.5" in repr_str
        assert "word_margin=0.1" in repr_str
        assert "all_texts=False" in repr_str


class TestLTItem:
    """Tests for LTItem base class."""

    def test_analyze_does_nothing(self) -> None:
        item = LTItem()
        laparams = LAParams()
        item.analyze(laparams)


class TestLTText:
    """Tests for LTText interface."""

    def test_get_text_raises_not_implemented(self) -> None:
        class TestLTText(LTText):
            pass

        text_obj = TestLTText()
        with pytest.raises(NotImplementedError):
            text_obj.get_text()

    def test_repr_with_get_text(self) -> None:
        class ConcreteText(LTText):
            def get_text(self) -> str:
                return "Hello"

        text_obj = ConcreteText()
        assert repr(text_obj) == "<ConcreteText 'Hello'>"


class TestLTComponent:
    """Tests for LTComponent class."""

    def test_init_sets_bbox_properties(self) -> None:
        component = LTComponent((10.0, 20.0, 30.0, 50.0))
        assert component.x0 == 10.0
        assert component.y0 == 20.0
        assert component.x1 == 30.0
        assert component.y1 == 50.0
        assert component.width == 20.0
        assert component.height == 30.0
        assert component.bbox == (10.0, 20.0, 30.0, 50.0)

    def test_set_bbox(self) -> None:
        component = LTComponent((0, 0, 1, 1))
        component.set_bbox((5.0, 10.0, 15.0, 25.0))
        assert component.x0 == 5.0
        assert component.y0 == 10.0
        assert component.x1 == 15.0
        assert component.y1 == 25.0
        assert component.width == 10.0
        assert component.height == 15.0

    def test_repr(self) -> None:
        component = LTComponent((10.0, 20.0, 30.0, 40.0))
        repr_str = repr(component)
        assert "LTComponent" in repr_str
        assert "10" in repr_str and "20" in repr_str

    def test_is_empty_true(self) -> None:
        component = LTComponent((10, 10, 10, 20))
        assert component.is_empty() is True

        component2 = LTComponent((10, 10, 20, 10))
        assert component2.is_empty() is True

        component3 = LTComponent((10, 10, 5, 20))
        assert component3.is_empty() is True

    def test_is_empty_false(self) -> None:
        component = LTComponent((0, 0, 10, 10))
        assert component.is_empty() is False

    def test_comparison_raises_error(self) -> None:
        component1 = LTComponent((0, 0, 10, 10))
        component2 = LTComponent((5, 5, 15, 15))

        with pytest.raises(PDFValueError):
            _ = component1 < component2

        with pytest.raises(PDFValueError):
            _ = component1 <= component2

        with pytest.raises(PDFValueError):
            _ = component1 > component2

        with pytest.raises(PDFValueError):
            _ = component1 >= component2

    def test_is_hoverlap_true(self) -> None:
        comp1 = LTComponent((0, 0, 20, 10))
        comp2 = LTComponent((10, 5, 30, 15))
        assert comp1.is_hoverlap(comp2) is True
        assert comp2.is_hoverlap(comp1) is True

    def test_is_hoverlap_false(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((20, 0, 30, 10))
        assert comp1.is_hoverlap(comp2) is False

    def test_is_hoverlap_touching(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((10, 0, 20, 10))
        assert comp1.is_hoverlap(comp2) is True

    def test_hdistance_with_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 20, 10))
        comp2 = LTComponent((10, 0, 30, 10))
        assert comp1.hdistance(comp2) == 0

    def test_hdistance_without_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((20, 0, 30, 10))
        assert comp1.hdistance(comp2) == 10

    def test_hdistance_symmetric(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((25, 0, 35, 10))
        assert comp1.hdistance(comp2) == comp2.hdistance(comp1) == 15

    def test_hoverlap_with_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 20, 10))
        comp2 = LTComponent((10, 0, 30, 10))
        assert comp1.hoverlap(comp2) == 10

    def test_hoverlap_without_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((20, 0, 30, 10))
        assert comp1.hoverlap(comp2) == 0

    def test_is_voverlap_true(self) -> None:
        comp1 = LTComponent((0, 0, 10, 20))
        comp2 = LTComponent((0, 10, 10, 30))
        assert comp1.is_voverlap(comp2) is True
        assert comp2.is_voverlap(comp1) is True

    def test_is_voverlap_false(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((0, 20, 10, 30))
        assert comp1.is_voverlap(comp2) is False

    def test_vdistance_with_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 20))
        comp2 = LTComponent((0, 10, 10, 30))
        assert comp1.vdistance(comp2) == 0

    def test_vdistance_without_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((0, 20, 10, 30))
        assert comp1.vdistance(comp2) == 10

    def test_voverlap_with_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 20))
        comp2 = LTComponent((0, 10, 10, 30))
        assert comp1.voverlap(comp2) == 10

    def test_voverlap_without_overlap(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((0, 25, 10, 35))
        assert comp1.voverlap(comp2) == 0


class TestLTCurve:
    """Tests for LTCurve class."""

    def test_init_basic(self) -> None:
        pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        curve = LTCurve(linewidth=1.0, pts=pts)
        assert curve.linewidth == 1.0
        assert curve.pts == pts
        assert curve.stroke is False
        assert curve.fill is False
        assert curve.evenodd is False
        assert curve.stroking_color is None
        assert curve.non_stroking_color is None

    def test_init_with_all_parameters(self) -> None:
        pts = [(0.0, 0.0), (20.0, 20.0)]
        original_path = [("m", 0, 0), ("l", 20, 20)]
        dashing_style = ([3, 2], 0)

        curve = LTCurve(
            linewidth=2.5,
            pts=pts,
            stroke=True,
            fill=True,
            evenodd=True,
            stroking_color=(0.5, 0.5, 0.5),
            non_stroking_color=(0.2, 0.2, 0.2),
            original_path=original_path,
            dashing_style=dashing_style,
        )
        assert curve.linewidth == 2.5
        assert curve.stroke is True
        assert curve.fill is True
        assert curve.evenodd is True
        assert curve.stroking_color == (0.5, 0.5, 0.5)
        assert curve.non_stroking_color == (0.2, 0.2, 0.2)
        assert curve.original_path == original_path
        assert curve.dashing_style == dashing_style

    def test_bbox_calculation(self) -> None:
        pts = [(5.0, 10.0), (15.0, 10.0), (15.0, 30.0), (5.0, 30.0)]
        curve = LTCurve(linewidth=1.0, pts=pts)
        assert curve.x0 == 5.0
        assert curve.y0 == 10.0
        assert curve.x1 == 15.0
        assert curve.y1 == 30.0

    def test_get_pts(self) -> None:
        pts = [(0.0, 0.0), (10.5, 20.25)]
        curve = LTCurve(linewidth=1.0, pts=pts)
        pts_str = curve.get_pts()
        assert "0.000,0.000" in pts_str
        assert "10.500,20.250" in pts_str
        assert "," in pts_str


class TestLTLine:
    """Tests for LTLine class."""

    def test_init_basic(self) -> None:
        line = LTLine(linewidth=1.0, p0=(0.0, 0.0), p1=(10.0, 10.0))
        assert line.pts == [(0.0, 0.0), (10.0, 10.0)]
        assert line.linewidth == 1.0

    def test_init_with_colors(self) -> None:
        line = LTLine(
            linewidth=2.0,
            p0=(5.0, 5.0),
            p1=(25.0, 25.0),
            stroke=True,
            fill=False,
            stroking_color=(1.0, 0.0, 0.0),
        )
        assert line.stroke is True
        assert line.fill is False
        assert line.stroking_color == (1.0, 0.0, 0.0)

    def test_bbox(self) -> None:
        line = LTLine(linewidth=1.0, p0=(10.0, 20.0), p1=(30.0, 40.0))
        assert line.x0 == 10.0
        assert line.y0 == 20.0
        assert line.x1 == 30.0
        assert line.y1 == 40.0

    def test_horizontal_line(self) -> None:
        line = LTLine(linewidth=1.0, p0=(0.0, 10.0), p1=(100.0, 10.0))
        assert line.width == 100.0
        assert line.height == 0.0

    def test_vertical_line(self) -> None:
        line = LTLine(linewidth=1.0, p0=(10.0, 0.0), p1=(10.0, 100.0))
        assert line.width == 0.0
        assert line.height == 100.0


class TestLTRect:
    """Tests for LTRect class."""

    def test_init_basic(self) -> None:
        rect = LTRect(linewidth=1.0, bbox=(0.0, 0.0, 20.0, 10.0))
        expected_pts = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        assert rect.pts == expected_pts
        assert rect.linewidth == 1.0

    def test_init_with_colors(self) -> None:
        rect = LTRect(
            linewidth=0.5,
            bbox=(10.0, 10.0, 50.0, 50.0),
            stroke=True,
            fill=True,
            stroking_color=(0.0, 0.0, 0.0),
            non_stroking_color=(0.8, 0.8, 0.8),
        )
        assert rect.stroke is True
        assert rect.fill is True
        assert rect.stroking_color == (0.0, 0.0, 0.0)
        assert rect.non_stroking_color == (0.8, 0.8, 0.8)

    def test_bbox_dimensions(self) -> None:
        rect = LTRect(linewidth=1.0, bbox=(10.0, 20.0, 60.0, 80.0))
        assert rect.x0 == 10.0
        assert rect.y0 == 20.0
        assert rect.x1 == 60.0
        assert rect.y1 == 80.0
        assert rect.width == 50.0
        assert rect.height == 60.0


class TestLTAnno:
    """Tests for LTAnno class."""

    def test_init_and_get_text(self) -> None:
        anno = LTAnno("Hello")
        assert anno.get_text() == "Hello"

    def test_empty_string(self) -> None:
        anno = LTAnno("")
        assert anno.get_text() == ""

    def test_space(self) -> None:
        anno = LTAnno(" ")
        assert anno.get_text() == " "

    def test_newline(self) -> None:
        anno = LTAnno("\n")
        assert anno.get_text() == "\n"

    def test_repr(self) -> None:
        anno = LTAnno("Test")
        repr_str = repr(anno)
        assert "LTAnno" in repr_str
        assert "Test" in repr_str


class TestLTChar:
    """Tests for LTChar class."""

    def test_init_horizontal_char(self) -> None:
        font = MockPDFFont(fontname="TestFont", vertical=False, descent=-0.2)
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char = LTChar(
            matrix=(1, 0, 0, 1, 100, 200),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="A",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )

        assert char.get_text() == "A"
        assert char.fontname == "TestFont"
        assert char.matrix == (1, 0, 0, 1, 100, 200)

    def test_init_vertical_char(self) -> None:
        font = MockPDFFont(fontname="VerticalFont", vertical=True, descent=-0.2)
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char = LTChar(
            matrix=(1, 0, 0, 1, 100, 200),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="A",
            textwidth=0.6,
            textdisp=(500, 880),
            ncs=ncs,
            graphicstate=graphicstate,
        )

        assert char.get_text() == "A"
        assert char.fontname == "VerticalFont"

    def test_adv_calculation(self) -> None:
        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=0.01,
            rise=0.0,
            text="X",
            textwidth=500.0,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        assert char.adv == 500.0 * 12.0 * 0.01

    def test_upright_determination(self) -> None:
        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char_upright = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="A",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        assert char_upright.upright is True

        char_rotated = LTChar(
            matrix=(0, 1, -1, 0, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="A",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        assert char_rotated.upright is False

    def test_repr(self) -> None:
        font = MockPDFFont(fontname="TestFont")
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="B",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )

        repr_str = repr(char)
        assert "LTChar" in repr_str
        assert "TestFont" in repr_str
        assert "B" in repr_str


class TestLTContainer:
    """Tests for LTContainer class."""

    def test_init(self) -> None:
        container = LTContainer((0, 0, 100, 100))
        assert len(container) == 0

    def test_add_and_iterate(self) -> None:
        container = LTContainer((0, 0, 100, 100))
        item1 = LTItem()
        item2 = LTItem()
        container.add(item1)
        container.add(item2)

        assert len(container) == 2
        items = list(container)
        assert item1 in items
        assert item2 in items

    def test_extend(self) -> None:
        container = LTContainer((0, 0, 100, 100))
        items = [LTItem(), LTItem(), LTItem()]
        container.extend(items)
        assert len(container) == 3

    def test_analyze_calls_child_analyze(self) -> None:
        container = LTContainer((0, 0, 100, 100))
        mock_item = MagicMock(spec=LTItem)
        container.add(mock_item)

        laparams = LAParams()
        container.analyze(laparams)

        mock_item.analyze.assert_called_once_with(laparams)


class TestLTExpandableContainer:
    """Tests for LTExpandableContainer class."""

    def test_init_with_inf_bbox(self) -> None:
        container = LTExpandableContainer()
        assert container.x0 == INF
        assert container.y0 == INF
        assert container.x1 == -INF
        assert container.y1 == -INF

    def test_add_expands_bbox(self) -> None:
        container = LTExpandableContainer()

        comp1 = LTComponent((10, 20, 30, 40))
        container.add(comp1)
        assert container.x0 == 10
        assert container.y0 == 20
        assert container.x1 == 30
        assert container.y1 == 40

        comp2 = LTComponent((5, 15, 35, 45))
        container.add(comp2)
        assert container.x0 == 5
        assert container.y0 == 15
        assert container.x1 == 35
        assert container.y1 == 45


class TestLTTextContainer:
    """Tests for LTTextContainer class."""

    def test_get_text_concatenates_children(self) -> None:
        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        container = LTTextContainer()
        for i, char in enumerate("Hello"):
            c = LTChar(
                matrix=(1, 0, 0, 1, i * 10, 0),
                font=font,
                fontsize=12.0,
                scaling=1.0,
                rise=0.0,
                text=char,
                textwidth=0.6,
                textdisp=0.0,
                ncs=ncs,
                graphicstate=graphicstate,
            )
            container.add(c)

        assert container.get_text() == "Hello"

    def test_get_text_empty_container(self) -> None:
        container = LTTextContainer()
        assert container.get_text() == ""

    def test_get_text_with_ltanno_via_base_add(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        container = LTTextContainer()
        LTContainer.add(container, cast(LTItem, LTAnno("Hello")))
        LTContainer.add(container, cast(LTItem, LTAnno(" ")))
        LTContainer.add(container, cast(LTItem, LTAnno("World")))

        assert container.get_text() == "Hello World"


class TestLTTextLine:
    """Tests for LTTextLine and its subclasses."""

    def test_textline_horizontal_init(self) -> None:
        line = LTTextLineHorizontal(word_margin=0.1)
        assert line.word_margin == 0.1

    def test_textline_vertical_init(self) -> None:
        line = LTTextLineVertical(word_margin=0.2)
        assert line.word_margin == 0.2

    def test_repr(self) -> None:
        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        line = LTTextLineHorizontal(word_margin=0.1)
        char = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="T",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        line.add(char)
        repr_str = repr(line)
        assert "LTTextLineHorizontal" in repr_str
        assert "T" in repr_str

    def test_analyze_adds_newline(self) -> None:
        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        line = LTTextLineHorizontal(word_margin=0.1)
        char = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="X",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        line.add(char)
        laparams = LAParams()
        line.analyze(laparams)
        text = line.get_text()
        assert text.endswith("\n")

    def test_is_empty_whitespace_only(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        line = LTTextLineHorizontal(word_margin=0.1)
        line.set_bbox((0, 0, 10, 10))
        LTContainer.add(line, cast(LTItem, LTAnno("   ")))
        assert line.is_empty() is True

    def test_is_empty_with_content(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        line = LTTextLineHorizontal(word_margin=0.1)
        line.set_bbox((0, 0, 10, 10))
        LTContainer.add(line, cast(LTItem, LTAnno("Hello")))
        assert line.is_empty() is False

    def test_find_neighbors_raises_not_implemented(self) -> None:
        line = LTTextLine(word_margin=0.1)
        plane = Plane((0, 0, 100, 100))
        with pytest.raises(NotImplementedError):
            line.find_neighbors(plane, 0.5)


class TestLTTextLineHorizontal:
    """Tests for LTTextLineHorizontal class."""

    def test_add_inserts_space_when_gap_large(self) -> None:
        line = LTTextLineHorizontal(word_margin=0.1)

        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char1 = LTChar(
            matrix=(1, 0, 0, 1, 0, 0),
            font=font,
            fontsize=10.0,
            scaling=1.0,
            rise=0.0,
            text="A",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        line.add(char1)

        char2 = LTChar(
            matrix=(1, 0, 0, 1, 50, 0),
            font=font,
            fontsize=10.0,
            scaling=1.0,
            rise=0.0,
            text="B",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        line.add(char2)

        text = line.get_text()
        assert " " in text

    def test_alignment_methods(self) -> None:
        line1 = LTTextLineHorizontal(word_margin=0.1)
        line1.set_bbox((10, 0, 50, 10))

        other_left = LTComponent((10, 15, 40, 25))
        assert line1._is_left_aligned_with(other_left, tolerance=0) is True
        assert line1._is_left_aligned_with(other_left, tolerance=0.1) is True

        other_right = LTComponent((20, 15, 50, 25))
        assert line1._is_right_aligned_with(other_right, tolerance=0) is True

        other_center = LTComponent((15, 15, 45, 25))
        assert line1._is_centrally_aligned_with(other_center, tolerance=0) is True

        other_same_height = LTComponent((0, 15, 30, 25))
        assert line1._is_same_height_as(other_same_height, tolerance=0) is True

    def test_find_neighbors_horizontal(self) -> None:
        laparams = LAParams()
        plane = Plane((0, 0, 100, 100))

        line = LTTextLineHorizontal(laparams.word_margin)
        line.set_bbox((10, 10, 50, 20))
        plane.add(line)

        neighbor = LTTextLineHorizontal(laparams.word_margin)
        neighbor.set_bbox((10, 20, 50, 30))
        plane.add(neighbor)

        not_neighbor = LTTextLineHorizontal(laparams.word_margin)
        not_neighbor.set_bbox((70, 50, 90, 60))
        plane.add(not_neighbor)

        neighbors = line.find_neighbors(plane, laparams.line_margin)
        assert line in neighbors
        assert neighbor in neighbors
        assert not_neighbor not in neighbors


class TestLTTextLineVertical:
    """Tests for LTTextLineVertical class."""

    def test_alignment_methods(self) -> None:
        line = LTTextLineVertical(word_margin=0.1)
        line.set_bbox((0, 10, 10, 50))

        other_lower = LTComponent((15, 10, 25, 40))
        assert line._is_lower_aligned_with(other_lower, tolerance=0) is True

        other_upper = LTComponent((15, 20, 25, 50))
        assert line._is_upper_aligned_with(other_upper, tolerance=0) is True

        other_center = LTComponent((15, 15, 25, 45))
        assert line._is_centrally_aligned_with(other_center, tolerance=0) is True

        other_same_width = LTComponent((15, 0, 25, 30))
        assert line._is_same_width_as(other_same_width, tolerance=0) is True

    def test_find_neighbors_vertical(self) -> None:
        laparams = LAParams(detect_vertical=True)
        plane = Plane((0, 0, 100, 100))

        line = LTTextLineVertical(laparams.word_margin)
        line.set_bbox((10, 10, 20, 50))
        plane.add(line)

        neighbor = LTTextLineVertical(laparams.word_margin)
        neighbor.set_bbox((20, 10, 30, 50))
        plane.add(neighbor)

        not_neighbor = LTTextLineVertical(laparams.word_margin)
        not_neighbor.set_bbox((60, 70, 70, 100))
        plane.add(not_neighbor)

        neighbors = line.find_neighbors(plane, laparams.line_margin)
        assert line in neighbors
        assert neighbor in neighbors
        assert not_neighbor not in neighbors


class TestLTTextBox:
    """Tests for LTTextBox and its subclasses."""

    def test_textbox_init(self) -> None:
        box = LTTextBoxHorizontal()
        assert box.index == -1

    def test_textbox_repr(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        box = LTTextBoxHorizontal()
        box.index = 5
        line = LTTextLineHorizontal(word_margin=0.1)
        line.set_bbox((0, 0, 100, 12))
        LTContainer.add(line, cast(LTItem, LTAnno("Test")))
        box.add(line)

        repr_str = repr(box)
        assert "LTTextBoxHorizontal" in repr_str
        assert "(5)" in repr_str

    def test_textbox_horizontal_analyze_sorts_by_y1(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        box = LTTextBoxHorizontal()

        line1 = LTTextLineHorizontal(word_margin=0.1)
        line1.set_bbox((0, 0, 100, 10))
        LTContainer.add(line1, cast(LTItem, LTAnno("Line 1")))

        line2 = LTTextLineHorizontal(word_margin=0.1)
        line2.set_bbox((0, 20, 100, 30))
        LTContainer.add(line2, cast(LTItem, LTAnno("Line 2")))

        box.add(line1)
        box.add(line2)

        laparams = LAParams()
        box.analyze(laparams)

        lines = list(box)
        assert "Line 2" in lines[0].get_text()
        assert "Line 1" in lines[1].get_text()

    def test_textbox_vertical_analyze_sorts_by_x1(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        box = LTTextBoxVertical()

        line1 = LTTextLineVertical(word_margin=0.1)
        line1.set_bbox((0, 0, 10, 100))
        LTContainer.add(line1, cast(LTItem, LTAnno("Line 1")))

        line2 = LTTextLineVertical(word_margin=0.1)
        line2.set_bbox((20, 0, 30, 100))
        LTContainer.add(line2, cast(LTItem, LTAnno("Line 2")))

        box.add(line1)
        box.add(line2)

        laparams = LAParams(detect_vertical=True)
        box.analyze(laparams)

        lines = list(box)
        assert "Line 2" in lines[0].get_text()
        assert "Line 1" in lines[1].get_text()

    def test_textbox_horizontal_writing_mode(self) -> None:
        box = LTTextBoxHorizontal()
        assert box.get_writing_mode() == "lr-tb"

    def test_textbox_vertical_writing_mode(self) -> None:
        box = LTTextBoxVertical()
        assert box.get_writing_mode() == "tb-rl"

    def test_get_writing_mode_raises_not_implemented(self) -> None:
        box = LTTextBox()
        with pytest.raises(NotImplementedError):
            box.get_writing_mode()


class TestLTTextGroup:
    """Tests for LTTextGroup and its subclasses."""

    def test_textgroup_init_with_boxes(self) -> None:
        box1 = LTTextBoxHorizontal()
        box2 = LTTextBoxHorizontal()

        group = LTTextGroup([box1, box2])
        assert len(group) == 2

    def test_textgroup_lrtb_analyze_sorts(self) -> None:
        box1 = LTTextBoxHorizontal()
        box1.set_bbox((0, 0, 50, 50))

        box2 = LTTextBoxHorizontal()
        box2.set_bbox((100, 100, 150, 150))

        group = LTTextGroupLRTB([box1, box2])

        laparams = LAParams(boxes_flow=0.5)
        group.analyze(laparams)

        boxes = list(group)
        assert boxes[0].y0 >= boxes[1].y0

    def test_textgroup_tbrl_analyze_sorts(self) -> None:
        box1 = LTTextBoxVertical()
        box1.set_bbox((0, 0, 50, 50))

        box2 = LTTextBoxVertical()
        box2.set_bbox((100, 0, 150, 50))

        group = LTTextGroupTBRL([box1, box2])

        laparams = LAParams(boxes_flow=0.5)
        group.analyze(laparams)

        boxes = list(group)
        assert boxes[0].x0 >= boxes[1].x0


class TestLTFigure:
    """Tests for LTFigure class."""

    def test_init(self) -> None:
        figure = LTFigure(
            name="Figure1",
            bbox=(0, 0, 100, 100),
            matrix=(1, 0, 0, 1, 50, 50),
        )
        assert figure.name == "Figure1"
        assert figure.matrix == (1, 0, 0, 1, 50, 50)

    def test_repr(self) -> None:
        figure = LTFigure(
            name="TestFig",
            bbox=(0, 0, 50, 50),
            matrix=(1, 0, 0, 1, 0, 0),
        )
        repr_str = repr(figure)
        assert "LTFigure" in repr_str
        assert "TestFig" in repr_str
        assert "matrix" in repr_str

    def test_analyze_with_all_texts_false(self) -> None:
        figure = LTFigure(
            name="Fig",
            bbox=(0, 0, 100, 100),
            matrix=(1, 0, 0, 1, 0, 0),
        )

        laparams = LAParams(all_texts=False)
        figure.analyze(laparams)

    def test_analyze_with_all_texts_true(self) -> None:
        figure = LTFigure(
            name="Fig",
            bbox=(0, 0, 100, 100),
            matrix=(1, 0, 0, 1, 0, 0),
        )

        laparams = LAParams(all_texts=True)
        figure.analyze(laparams)


class TestLTPage:
    """Tests for LTPage class."""

    def test_init(self) -> None:
        page = LTPage(pageid=1, bbox=(0, 0, 612, 792), rotate=0)
        assert page.pageid == 1
        assert page.rotate == 0
        assert page.width == 612
        assert page.height == 792

    def test_init_with_rotation(self) -> None:
        page = LTPage(pageid=2, bbox=(0, 0, 612, 792), rotate=90)
        assert page.pageid == 2
        assert page.rotate == 90

    def test_repr(self) -> None:
        page = LTPage(pageid=1, bbox=(0, 0, 612, 792), rotate=90)
        repr_str = repr(page)
        assert "LTPage" in repr_str
        assert "1" in repr_str
        assert "rotate=90" in repr_str


class TestLTLayoutContainer:
    """Tests for LTLayoutContainer and layout analysis."""

    def test_init(self) -> None:
        layout = LTLayoutContainer((0, 0, 612, 792))
        assert layout.groups is None

    def test_group_objects_creates_textlines(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams()

        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        chars = []
        for i, char in enumerate("Hello"):
            c = LTChar(
                matrix=(1, 0, 0, 1, i * 10, 0),
                font=font,
                fontsize=12.0,
                scaling=1.0,
                rise=0.0,
                text=char,
                textwidth=0.6,
                textdisp=0.0,
                ncs=ncs,
                graphicstate=graphicstate,
            )
            chars.append(c)

        textlines = list(layout.group_objects(laparams, chars))
        assert len(textlines) >= 1
        assert any(isinstance(line, LTTextLineHorizontal) for line in textlines)

    def test_group_textlines_creates_textboxes(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams()

        line1 = LTTextLineHorizontal(laparams.word_margin)
        line1.set_bbox((0, 0, 50, 10))
        LTContainer.add(line1, cast(LTItem, LTAnno("Hello")))

        line2 = LTTextLineHorizontal(laparams.word_margin)
        line2.set_bbox((0, 10, 50, 20))
        LTContainer.add(line2, cast(LTItem, LTAnno("World")))

        textboxes = list(layout.group_textlines(laparams, [line1, line2]))
        assert len(textboxes) >= 1

    def test_group_textboxes_creates_groups(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        layout = LTLayoutContainer((0, 0, 200, 200))
        laparams = LAParams(boxes_flow=0.5)

        box1 = LTTextBoxHorizontal()
        box1.set_bbox((0, 0, 50, 20))
        line1 = LTTextLineHorizontal(laparams.word_margin)
        line1.set_bbox((0, 0, 50, 10))
        LTContainer.add(line1, cast(LTItem, LTAnno("Box1")))
        box1.add(line1)

        box2 = LTTextBoxHorizontal()
        box2.set_bbox((0, 30, 50, 50))
        line2 = LTTextLineHorizontal(laparams.word_margin)
        line2.set_bbox((0, 30, 50, 40))
        LTContainer.add(line2, cast(LTItem, LTAnno("Box2")))
        box2.add(line2)

        groups = layout.group_textboxes(laparams, [box1, box2])
        assert len(groups) >= 1

    def test_analyze_with_boxes_flow_none(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams(boxes_flow=None)

        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        for i, text in enumerate(["A", "B", "C"]):
            char = LTChar(
                matrix=(1, 0, 0, 1, i * 15, 50),
                font=font,
                fontsize=12.0,
                scaling=1.0,
                rise=0.0,
                text=text,
                textwidth=0.6,
                textdisp=0.0,
                ncs=ncs,
                graphicstate=graphicstate,
            )
            layout.add(char)

        layout.analyze(laparams)
        assert layout.groups is None

    def test_analyze_with_boxes_flow_set(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams(boxes_flow=0.5)

        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        for i, text in enumerate(["A", "B", "C"]):
            char = LTChar(
                matrix=(1, 0, 0, 1, i * 15, 50),
                font=font,
                fontsize=12.0,
                scaling=1.0,
                rise=0.0,
                text=text,
                textwidth=0.6,
                textdisp=0.0,
                ncs=ncs,
                graphicstate=graphicstate,
            )
            layout.add(char)

        layout.analyze(laparams)
        assert layout.groups is not None

    def test_analyze_with_empty_layout(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams()
        layout.analyze(laparams)
        assert len(list(layout)) == 0

    def test_analyze_with_non_text_objects(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams()

        rect = LTRect(linewidth=1.0, bbox=(10, 10, 50, 50))
        layout.add(rect)

        line = LTLine(linewidth=1.0, p0=(0, 0), p1=(100, 100))
        layout.add(line)

        layout.analyze(laparams)
        objects = list(layout)
        assert rect in objects
        assert line in objects

    def test_analyze_with_vertical_detection(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 200))
        laparams = LAParams(detect_vertical=True, boxes_flow=0.5)

        font = MockPDFFont(vertical=True)
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        for i, text in enumerate(["A", "B", "C"]):
            char = LTChar(
                matrix=(0, 1, -1, 0, 50, i * 15 + 10),
                font=font,
                fontsize=12.0,
                scaling=1.0,
                rise=0.0,
                text=text,
                textwidth=0.6,
                textdisp=(500, 880),
                ncs=ncs,
                graphicstate=graphicstate,
            )
            layout.add(char)

        layout.analyze(laparams)


class TestIndexAssigner:
    """Tests for IndexAssigner class."""

    def test_init(self) -> None:
        assigner = IndexAssigner()
        assert assigner.index == 0

    def test_init_with_start_index(self) -> None:
        assigner = IndexAssigner(index=5)
        assert assigner.index == 5

    def test_run_assigns_index_to_textbox(self) -> None:
        assigner = IndexAssigner()

        box = LTTextBoxHorizontal()
        assigner.run(box)

        assert box.index == 0
        assert assigner.index == 1

    def test_run_assigns_sequential_indices(self) -> None:
        assigner = IndexAssigner()

        box1 = LTTextBoxHorizontal()
        box2 = LTTextBoxHorizontal()
        box3 = LTTextBoxHorizontal()

        assigner.run(box1)
        assigner.run(box2)
        assigner.run(box3)

        assert box1.index == 0
        assert box2.index == 1
        assert box3.index == 2
        assert assigner.index == 3

    def test_run_with_textgroup(self) -> None:
        assigner = IndexAssigner()

        box1 = LTTextBoxHorizontal()
        box1.set_bbox((0, 0, 50, 20))
        box2 = LTTextBoxHorizontal()
        box2.set_bbox((0, 30, 50, 50))

        group = LTTextGroupLRTB([box1, box2])
        assigner.run(group)

        assert box1.index == 0
        assert box2.index == 1

    def test_run_with_nested_groups(self) -> None:
        assigner = IndexAssigner()

        box1 = LTTextBoxHorizontal()
        box1.set_bbox((0, 0, 50, 20))
        box2 = LTTextBoxHorizontal()
        box2.set_bbox((0, 30, 50, 50))
        box3 = LTTextBoxHorizontal()
        box3.set_bbox((60, 0, 100, 50))

        inner_group = LTTextGroupLRTB([box1, box2])
        outer_group = LTTextGroupLRTB([inner_group, box3])

        assigner.run(outer_group)

        assert box1.index == 0
        assert box2.index == 1
        assert box3.index == 2

    def test_run_ignores_non_textbox_items(self) -> None:
        assigner = IndexAssigner()

        rect = LTRect(linewidth=1.0, bbox=(0, 0, 50, 50))
        assigner.run(rect)

        assert assigner.index == 0


class TestBoundingBoxCalculations:
    """Additional tests for bounding box operations."""

    def test_nested_container_bbox_expansion(self) -> None:
        outer = LTExpandableContainer()

        inner1 = LTComponent((0, 0, 50, 10))
        outer.add(inner1)

        inner2 = LTComponent((100, 100, 150, 110))
        outer.add(inner2)

        assert outer.x0 == 0
        assert outer.y0 == 0
        assert outer.x1 == 150
        assert outer.y1 == 110

    def test_component_with_negative_coords(self) -> None:
        comp = LTComponent((-50, -30, 50, 30))
        assert comp.x0 == -50
        assert comp.y0 == -30
        assert comp.width == 100
        assert comp.height == 60

    def test_zero_size_component(self) -> None:
        comp = LTComponent((10, 10, 10, 10))
        assert comp.width == 0
        assert comp.height == 0
        assert comp.is_empty() is True


class TestTextExtraction:
    """Tests for text extraction from layout objects."""

    def test_textbox_get_text(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        box = LTTextBoxHorizontal()

        line1 = LTTextLineHorizontal(word_margin=0.1)
        line1.set_bbox((0, 10, 100, 22))
        LTContainer.add(line1, cast(LTItem, LTAnno("First line")))
        box.add(line1)

        line2 = LTTextLineHorizontal(word_margin=0.1)
        line2.set_bbox((0, 0, 100, 10))
        LTContainer.add(line2, cast(LTItem, LTAnno("Second line")))
        box.add(line2)

        text = box.get_text()
        assert "First line" in text
        assert "Second line" in text

    def test_page_text_extraction(self) -> None:
        from pdfminer.layout import LTContainer
        from typing import cast

        page = LTPage(pageid=1, bbox=(0, 0, 612, 792))

        box = LTTextBoxHorizontal()
        box.set_bbox((72, 720, 540, 732))
        line = LTTextLineHorizontal(word_margin=0.1)
        line.set_bbox((72, 720, 540, 732))
        LTContainer.add(line, cast(LTItem, LTAnno("Page content")))
        box.add(line)
        page.add(box)

        has_textbox = False
        for item in page:
            if isinstance(item, LTTextBox):
                has_textbox = True
                assert "Page content" in item.get_text()

        assert has_textbox


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_char_layout(self) -> None:
        layout = LTLayoutContainer((0, 0, 100, 100))
        laparams = LAParams()

        font = MockPDFFont()
        ncs = PDFColorSpace("DeviceGray", 1)
        graphicstate = PDFGraphicState()

        char = LTChar(
            matrix=(1, 0, 0, 1, 50, 50),
            font=font,
            fontsize=12.0,
            scaling=1.0,
            rise=0.0,
            text="X",
            textwidth=0.6,
            textdisp=0.0,
            ncs=ncs,
            graphicstate=graphicstate,
        )
        layout.add(char)

        layout.analyze(laparams)

    def test_overlapping_components(self) -> None:
        comp1 = LTComponent((0, 0, 50, 50))
        comp2 = LTComponent((25, 25, 75, 75))

        assert comp1.is_hoverlap(comp2) is True
        assert comp1.is_voverlap(comp2) is True
        assert comp1.hoverlap(comp2) == 25
        assert comp1.voverlap(comp2) == 25

    def test_touching_but_not_overlapping(self) -> None:
        comp1 = LTComponent((0, 0, 10, 10))
        comp2 = LTComponent((10, 10, 20, 20))

        assert comp1.is_hoverlap(comp2) is True
        assert comp1.is_voverlap(comp2) is True
        assert comp1.hoverlap(comp2) == 0
        assert comp1.voverlap(comp2) == 0

    def test_completely_contained(self) -> None:
        outer = LTComponent((0, 0, 100, 100))
        inner = LTComponent((25, 25, 75, 75))

        assert outer.is_hoverlap(inner) is True
        assert outer.is_voverlap(inner) is True

    def test_very_large_coordinates(self) -> None:
        comp = LTComponent((0, 0, 1e6, 1e6))
        assert comp.width == 1e6
        assert comp.height == 1e6

    def test_very_small_coordinates(self) -> None:
        comp = LTComponent((0, 0, 1e-6, 1e-6))
        assert comp.width == pytest.approx(1e-6)
        assert comp.height == pytest.approx(1e-6)
