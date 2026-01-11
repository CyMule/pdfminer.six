"""Comprehensive tests for pdfminer/pdfinterp.py"""

from unittest.mock import MagicMock, patch

import pytest

from pdfminer import settings
from pdfminer.cmapdb import CMapBase, CMapDB
from pdfminer.pdfcolor import PREDEFINED_COLORSPACE
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import (
    LITERAL_FONT,
    LITERAL_FORM,
    LITERAL_IMAGE,
    LITERAL_PDF,
    LITERAL_TEXT,
    PDFContentParser,
    PDFGraphicState,
    PDFInterpreterError,
    PDFPageInterpreter,
    PDFResourceError,
    PDFResourceManager,
    PDFTextState,
)
from pdfminer.pdftypes import PDFStream
from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import LIT, PSKeyword, PSLiteral
from pdfminer.utils import MATRIX_IDENTITY


# Fixture to reset PREDEFINED_COLORSPACE state that may be polluted by other tests
# (e.g., test_converter.py modifies shared PDFColorSpace objects via shallow copy)
@pytest.fixture(autouse=True)
def reset_predefined_colorspace():
    """Reset PREDEFINED_COLORSPACE to pristine state before each test."""
    original_values = [
        ("DeviceGray", 1),
        ("CalRGB", 3),
        ("CalGray", 1),
        ("Lab", 3),
        ("DeviceRGB", 3),
        ("DeviceCMYK", 4),
        ("Separation", 1),
        ("Indexed", 1),
        ("Pattern", 1),
    ]
    for name, ncomponents in original_values:
        if name in PREDEFINED_COLORSPACE:
            PREDEFINED_COLORSPACE[name].name = name
            PREDEFINED_COLORSPACE[name].ncomponents = ncomponents
    yield


class TestLiteralConstants:
    """Tests for literal constants defined in pdfinterp"""

    def test_literal_pdf(self):
        assert isinstance(LITERAL_PDF, PSLiteral)
        assert LITERAL_PDF.name == "PDF"

    def test_literal_text(self):
        assert isinstance(LITERAL_TEXT, PSLiteral)
        assert LITERAL_TEXT.name == "Text"

    def test_literal_font(self):
        assert isinstance(LITERAL_FONT, PSLiteral)
        assert LITERAL_FONT.name == "Font"

    def test_literal_form(self):
        assert isinstance(LITERAL_FORM, PSLiteral)
        assert LITERAL_FORM.name == "Form"

    def test_literal_image(self):
        assert isinstance(LITERAL_IMAGE, PSLiteral)
        assert LITERAL_IMAGE.name == "Image"


class TestPDFTextState:
    """Tests for PDFTextState class"""

    def test_init_defaults(self):
        ts = PDFTextState()
        assert ts.font is None
        assert ts.fontsize == 0
        assert ts.charspace == 0
        assert ts.wordspace == 0
        assert ts.scaling == 100
        assert ts.leading == 0
        assert ts.render == 0
        assert ts.rise == 0
        assert ts.matrix == MATRIX_IDENTITY
        assert ts.linematrix == (0, 0)

    def test_reset(self):
        ts = PDFTextState()
        ts.matrix = (2, 0, 0, 2, 100, 200)
        ts.linematrix = (50, 50)
        ts.reset()
        assert ts.matrix == MATRIX_IDENTITY
        assert ts.linematrix == (0, 0)

    def test_copy(self):
        ts = PDFTextState()
        ts.fontsize = 12
        ts.charspace = 2
        ts.wordspace = 3
        ts.scaling = 110
        ts.leading = 14
        ts.render = 1
        ts.rise = 5
        ts.matrix = (1, 0, 0, 1, 10, 20)
        ts.linematrix = (5, 5)

        ts_copy = ts.copy()

        assert ts_copy is not ts
        assert ts_copy.fontsize == ts.fontsize
        assert ts_copy.charspace == ts.charspace
        assert ts_copy.wordspace == ts.wordspace
        assert ts_copy.scaling == ts.scaling
        assert ts_copy.leading == ts.leading
        assert ts_copy.render == ts.render
        assert ts_copy.rise == ts.rise
        assert ts_copy.matrix == ts.matrix
        assert ts_copy.linematrix == ts.linematrix

    def test_repr(self):
        ts = PDFTextState()
        ts.fontsize = 12
        repr_str = repr(ts)
        assert "PDFTextState" in repr_str
        assert "fontsize" in repr_str
        assert "12" in repr_str


class TestPDFGraphicState:
    """Tests for PDFGraphicState class"""

    def test_init_defaults(self):
        gs = PDFGraphicState()
        assert gs.linewidth == 0
        assert gs.linecap is None
        assert gs.linejoin is None
        assert gs.miterlimit is None
        assert gs.dash is None
        assert gs.intent is None
        assert gs.flatness is None
        assert gs.scolor == 0
        assert gs.scs == PREDEFINED_COLORSPACE["DeviceGray"]
        assert gs.ncolor == 0
        assert gs.ncs == PREDEFINED_COLORSPACE["DeviceGray"]

    def test_copy(self):
        gs = PDFGraphicState()
        gs.linewidth = 2.0
        gs.linecap = 1
        gs.linejoin = 2
        gs.miterlimit = 10
        gs.dash = ([3, 5], 0)
        gs.intent = "RelativeColorimetric"
        gs.flatness = 0.5
        gs.scolor = 0.5
        gs.scs = PREDEFINED_COLORSPACE["DeviceRGB"]
        gs.ncolor = (1.0, 0.0, 0.0)
        gs.ncs = PREDEFINED_COLORSPACE["DeviceRGB"]

        gs_copy = gs.copy()

        assert gs_copy is not gs
        assert gs_copy.linewidth == gs.linewidth
        assert gs_copy.linecap == gs.linecap
        assert gs_copy.linejoin == gs.linejoin
        assert gs_copy.miterlimit == gs.miterlimit
        assert gs_copy.dash == gs.dash
        assert gs_copy.intent == gs.intent
        assert gs_copy.flatness == gs.flatness
        assert gs_copy.scolor == gs.scolor
        assert gs_copy.scs == gs.scs
        assert gs_copy.ncolor == gs.ncolor
        assert gs_copy.ncs == gs.ncs

    def test_repr(self):
        gs = PDFGraphicState()
        gs.linewidth = 2.5
        repr_str = repr(gs)
        assert "PDFGraphicState" in repr_str
        assert "linewidth" in repr_str
        assert "2.5" in repr_str

    def test_stroke_colors(self):
        gs = PDFGraphicState()
        gs.scolor = (1.0, 0.0, 0.0)
        gs.scs = PREDEFINED_COLORSPACE["DeviceRGB"]
        assert gs.scolor == (1.0, 0.0, 0.0)
        assert gs.scs.name == "DeviceRGB"

    def test_fill_colors(self):
        gs = PDFGraphicState()
        gs.ncolor = (0.0, 1.0, 0.0, 0.5)
        gs.ncs = PREDEFINED_COLORSPACE["DeviceCMYK"]
        assert gs.ncolor == (0.0, 1.0, 0.0, 0.5)
        assert gs.ncs.name == "DeviceCMYK"


class TestPDFResourceManager:
    """Tests for PDFResourceManager class"""

    def test_init_default_caching(self):
        rsrcmgr = PDFResourceManager()
        assert rsrcmgr.caching is True
        assert rsrcmgr._cached_fonts == {}

    def test_init_caching_disabled(self):
        rsrcmgr = PDFResourceManager(caching=False)
        assert rsrcmgr.caching is False

    def test_get_procset_pdf_literal(self):
        rsrcmgr = PDFResourceManager()
        rsrcmgr.get_procset([LITERAL_PDF])

    def test_get_procset_text_literal(self):
        rsrcmgr = PDFResourceManager()
        rsrcmgr.get_procset([LITERAL_TEXT])

    def test_get_procset_other(self):
        rsrcmgr = PDFResourceManager()
        rsrcmgr.get_procset([LIT("ImageB"), LIT("ImageC")])

    def test_get_procset_multiple(self):
        rsrcmgr = PDFResourceManager()
        rsrcmgr.get_procset([LITERAL_PDF, LITERAL_TEXT, LIT("ImageB")])

    def test_get_cmap_nonexistent_non_strict(self):
        rsrcmgr = PDFResourceManager()
        cmap = rsrcmgr.get_cmap("NonExistentCMap", strict=False)
        assert isinstance(cmap, CMapBase)

    def test_get_cmap_nonexistent_strict(self):
        rsrcmgr = PDFResourceManager()
        with pytest.raises(CMapDB.CMapNotFound):
            rsrcmgr.get_cmap("NonExistentCMap", strict=True)

    def test_get_font_with_caching(self):
        rsrcmgr = PDFResourceManager(caching=True)
        spec = {
            "Type": LITERAL_FONT,
            "Subtype": LIT("Type1"),
            "BaseFont": LIT("Helvetica"),
        }
        with (
            patch.object(rsrcmgr, "_cached_fonts", {}) as cache,
            patch(
                "pdfminer.pdfinterp.PDFType1Font", autospec=True
            ) as mock_font_class,
        ):
            mock_font = MagicMock()
            mock_font_class.return_value = mock_font
            font = rsrcmgr.get_font(1, spec)
            assert font == mock_font
            assert cache.get(1) == mock_font

    def test_get_font_from_cache(self):
        rsrcmgr = PDFResourceManager(caching=True)
        mock_font = MagicMock()
        rsrcmgr._cached_fonts[1] = mock_font
        font = rsrcmgr.get_font(1, {})
        assert font is mock_font


class TestPDFContentParser:
    """Tests for PDFContentParser class"""

    def create_stream(self, data: bytes, objid: int = 1) -> PDFStream:
        stream = PDFStream({}, data)
        stream.set_objid(objid, 0)
        return stream

    def test_init(self):
        stream = self.create_stream(b"BT ET ")
        parser = PDFContentParser([stream])
        assert parser.streams == [stream]
        assert parser.istream >= 0

    def test_parse_simple_operators(self):
        stream = self.create_stream(b"q Q ")
        parser = PDFContentParser([stream])

        _, obj = parser.nextobject()
        assert isinstance(obj, PSKeyword)
        assert obj.name == b"q"

        _, obj = parser.nextobject()
        assert isinstance(obj, PSKeyword)
        assert obj.name == b"Q"

    def test_parse_numbers(self):
        stream = self.create_stream(b"100 200 m ")
        parser = PDFContentParser([stream])

        _, obj = parser.nextobject()
        assert obj == 100

        _, obj = parser.nextobject()
        assert obj == 200

        _, obj = parser.nextobject()
        assert isinstance(obj, PSKeyword)
        assert obj.name == b"m"

    def test_parse_text_operators(self):
        stream = self.create_stream(b"BT /F1 12 Tf (Hello) Tj ET ")
        parser = PDFContentParser([stream])

        objs = []
        try:
            while True:
                _, obj = parser.nextobject()
                objs.append(obj)
        except PSEOF:
            pass

        assert any(isinstance(o, PSKeyword) and o.name == b"BT" for o in objs)
        assert any(isinstance(o, PSKeyword) and o.name == b"Tf" for o in objs)
        assert any(isinstance(o, PSKeyword) and o.name == b"Tj" for o in objs)
        assert any(isinstance(o, PSKeyword) and o.name == b"ET" for o in objs)

    def test_empty_stream(self):
        stream = self.create_stream(b"")
        parser = PDFContentParser([stream])
        with pytest.raises(PSEOF):
            parser.nextobject()

    def test_multiple_streams(self):
        stream1 = self.create_stream(b"1 ", objid=1)
        stream2 = self.create_stream(b"2 ", objid=2)
        parser = PDFContentParser([stream1, stream2])

        _, obj = parser.nextobject()
        assert obj == 1

        _, obj = parser.nextobject()
        assert obj == 2

    def test_flush(self):
        stream = self.create_stream(b"1 2 3 ")
        parser = PDFContentParser([stream])
        parser.nextobject()
        parser.nextobject()
        parser.flush()
        assert len(parser.results) >= 0

    def test_keyword_bi_id_ei_inline_image(self):
        data = b"BI /W 10 /H 10 ID 1234567890 EI "
        stream = self.create_stream(data)
        parser = PDFContentParser([stream])
        objs = []
        try:
            while True:
                _, obj = parser.nextobject()
                objs.append(obj)
        except PSEOF:
            pass
        assert any(isinstance(o, PDFStream) for o in objs)


class TestPDFPageInterpreter:
    """Tests for PDFPageInterpreter class"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def test_init(self):
        assert self.interp.rsrcmgr is self.rsrcmgr
        assert self.interp.device is self.device
        assert self.interp.stream_ids == set()
        assert self.interp.parent_stream_ids == set()

    def test_dup(self):
        dup = self.interp.dup()
        assert dup is not self.interp
        assert dup.rsrcmgr is self.rsrcmgr
        assert dup.device is self.device

    def test_subinterp(self):
        self.interp.stream_ids = {1, 2}
        self.interp.parent_stream_ids = {3, 4}

        sub = self.interp.subinterp()

        assert sub is not self.interp
        assert 1 in sub.parent_stream_ids
        assert 2 in sub.parent_stream_ids
        assert 3 in sub.parent_stream_ids
        assert 4 in sub.parent_stream_ids

    def test_init_state(self):
        ctm = (1, 0, 0, 1, 0, 0)
        self.interp.init_resources({})
        self.interp.init_state(ctm)

        assert self.interp.ctm == ctm
        assert self.interp.gstack == []
        assert isinstance(self.interp.textstate, PDFTextState)
        assert isinstance(self.interp.graphicstate, PDFGraphicState)
        assert self.interp.curpath == []
        assert self.interp.argstack == []
        self.device.set_ctm.assert_called_with(ctm)

    def test_push_pop(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

        self.interp.push(1)
        self.interp.push(2)
        self.interp.push(3)

        popped = self.interp.pop(2)
        assert popped == [2, 3]
        assert self.interp.argstack == [1]

    def test_pop_zero(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        self.interp.push(1)

        popped = self.interp.pop(0)
        assert popped == []
        assert self.interp.argstack == [1]

    def test_get_set_current_state(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

        self.interp.ctm = (2, 0, 0, 2, 100, 100)
        self.interp.textstate.fontsize = 24
        self.interp.graphicstate.linewidth = 3

        state = self.interp.get_current_state()
        assert state[0] == (2, 0, 0, 2, 100, 100)

        self.interp.init_state(MATRIX_IDENTITY)

        self.interp.set_current_state(state)
        assert self.interp.ctm == (2, 0, 0, 2, 100, 100)


class TestPDFPageInterpreterGraphicsState:
    """Tests for graphics state operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_q_save_state(self):
        self.interp.do_q()
        assert len(self.interp.gstack) == 1

    def test_do_Q_restore_state(self):
        self.interp.graphicstate.linewidth = 5
        self.interp.do_q()
        self.interp.graphicstate.linewidth = 10
        self.interp.do_Q()
        assert self.interp.graphicstate.linewidth == 5

    def test_do_Q_empty_stack(self):
        self.interp.do_Q()

    def test_do_cm_concatenate_matrix(self):
        self.interp.do_cm(2, 0, 0, 2, 10, 20)
        assert self.interp.ctm[0] == 2
        assert self.interp.ctm[3] == 2
        self.device.set_ctm.assert_called()

    def test_do_cm_invalid_values(self):
        self.interp.do_cm("invalid", 0, 0, 1, 0, 0)

    def test_do_w_set_linewidth(self):
        self.interp.do_w(2.5)
        assert self.interp.graphicstate.linewidth > 0

    def test_do_w_invalid_value(self):
        self.interp.do_w("invalid")

    def test_do_J_set_linecap(self):
        self.interp.do_J(1)
        assert self.interp.graphicstate.linecap == 1

    def test_do_j_set_linejoin(self):
        self.interp.do_j(2)
        assert self.interp.graphicstate.linejoin == 2

    def test_do_M_set_miterlimit(self):
        self.interp.do_M(10)
        assert self.interp.graphicstate.miterlimit == 10

    def test_do_d_set_dash(self):
        self.interp.do_d([3, 5], 0)
        assert self.interp.graphicstate.dash == ([3, 5], 0)

    def test_do_ri_set_intent(self):
        self.interp.do_ri(LIT("RelativeColorimetric"))
        assert self.interp.graphicstate.intent == LIT("RelativeColorimetric")

    def test_do_i_set_flatness(self):
        self.interp.do_i(0.5)
        assert self.interp.graphicstate.flatness == 0.5


class TestPDFPageInterpreterPathOperators:
    """Tests for path operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_m_moveto(self):
        self.interp.do_m(100, 200)
        assert ("m", 100.0, 200.0) in self.interp.curpath

    def test_do_m_invalid_values(self):
        self.interp.do_m("invalid", 200)
        assert len(self.interp.curpath) == 0

    def test_do_l_lineto(self):
        self.interp.do_l(100, 200)
        assert ("l", 100.0, 200.0) in self.interp.curpath

    def test_do_l_invalid_values(self):
        self.interp.do_l(100, "invalid")
        assert len(self.interp.curpath) == 0

    def test_do_c_curveto(self):
        self.interp.do_c(10, 20, 30, 40, 50, 60)
        assert ("c", 10.0, 20.0, 30.0, 40.0, 50.0, 60.0) in self.interp.curpath

    def test_do_c_invalid_values(self):
        self.interp.do_c("x", 20, 30, 40, 50, 60)
        assert len(self.interp.curpath) == 0

    def test_do_v_curveto_initial(self):
        self.interp.do_v(30, 40, 50, 60)
        assert ("v", 30.0, 40.0, 50.0, 60.0) in self.interp.curpath

    def test_do_v_invalid_values(self):
        self.interp.do_v(30, "invalid", 50, 60)
        assert len(self.interp.curpath) == 0

    def test_do_y_curveto_final(self):
        self.interp.do_y(10, 20, 50, 60)
        assert ("y", 10.0, 20.0, 50.0, 60.0) in self.interp.curpath

    def test_do_y_invalid_values(self):
        self.interp.do_y(10, 20, 50, "invalid")
        assert len(self.interp.curpath) == 0

    def test_do_h_closepath(self):
        self.interp.do_h()
        assert ("h",) in self.interp.curpath

    def test_do_re_rectangle(self):
        self.interp.do_re(10, 20, 100, 50)
        assert ("m", 10.0, 20.0) in self.interp.curpath
        assert ("l", 110.0, 20.0) in self.interp.curpath
        assert ("l", 110.0, 70.0) in self.interp.curpath
        assert ("l", 10.0, 70.0) in self.interp.curpath
        assert ("h",) in self.interp.curpath

    def test_do_re_invalid_values(self):
        self.interp.do_re("invalid", 20, 100, 50)
        assert len(self.interp.curpath) == 0

    def test_do_S_stroke(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 100)
        self.interp.do_S()
        self.device.paint_path.assert_called_once()
        call_args = self.device.paint_path.call_args
        assert call_args[0][1] is True
        assert call_args[0][2] is False
        assert self.interp.curpath == []

    def test_do_s_close_and_stroke(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 100)
        self.interp.do_s()
        self.device.paint_path.assert_called_once()
        assert self.interp.curpath == []

    def test_do_f_fill(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_l(100, 100)
        self.interp.do_h()
        self.interp.do_f()
        self.device.paint_path.assert_called_once()
        call_args = self.device.paint_path.call_args
        assert call_args[0][1] is False
        assert call_args[0][2] is True

    def test_do_f_a_fill_evenodd(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_h()
        self.interp.do_f_a()
        call_args = self.device.paint_path.call_args
        assert call_args[0][3] is True

    def test_do_B_fill_and_stroke(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_h()
        self.interp.do_B()
        call_args = self.device.paint_path.call_args
        assert call_args[0][1] is True
        assert call_args[0][2] is True

    def test_do_B_a_fill_stroke_evenodd(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_h()
        self.interp.do_B_a()
        call_args = self.device.paint_path.call_args
        assert call_args[0][1] is True
        assert call_args[0][2] is True
        assert call_args[0][3] is True

    def test_do_b_close_fill_stroke(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_b()
        self.device.paint_path.assert_called_once()

    def test_do_b_a_close_fill_stroke_evenodd(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_b_a()
        self.device.paint_path.assert_called_once()

    def test_do_n_end_path(self):
        self.interp.do_m(0, 0)
        self.interp.do_l(100, 0)
        self.interp.do_n()
        assert self.interp.curpath == []


class TestPDFPageInterpreterColorOperators:
    """Tests for color operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_CS_set_stroke_colorspace(self):
        self.interp.do_CS(LIT("DeviceRGB"))
        assert self.interp.graphicstate.scs.name == "DeviceRGB"

    def test_do_cs_set_fill_colorspace(self):
        self.interp.do_cs(LIT("DeviceCMYK"))
        assert self.interp.graphicstate.ncs.name == "DeviceCMYK"

    def test_do_G_set_gray_stroke(self):
        self.interp.do_G(0.5)
        assert self.interp.graphicstate.scolor == 0.5
        assert self.interp.graphicstate.scs.name == "DeviceGray"

    def test_do_G_invalid_value(self):
        self.interp.do_G("invalid")

    def test_do_g_set_gray_fill(self):
        self.interp.do_g(0.75)
        assert self.interp.graphicstate.ncolor == 0.75
        assert self.interp.graphicstate.ncs.name == "DeviceGray"

    def test_do_g_invalid_value(self):
        self.interp.do_g("invalid")

    def test_do_RG_set_rgb_stroke(self):
        self.interp.do_RG(1.0, 0.0, 0.0)
        assert self.interp.graphicstate.scolor == (1.0, 0.0, 0.0)
        assert self.interp.graphicstate.scs.name == "DeviceRGB"

    def test_do_RG_invalid_values(self):
        self.interp.do_RG(1.0, "invalid", 0.0)

    def test_do_rg_set_rgb_fill(self):
        self.interp.do_rg(0.0, 1.0, 0.0)
        assert self.interp.graphicstate.ncolor == (0.0, 1.0, 0.0)
        assert self.interp.graphicstate.ncs.name == "DeviceRGB"

    def test_do_rg_invalid_values(self):
        self.interp.do_rg("invalid", 1.0, 0.0)

    def test_do_K_set_cmyk_stroke(self):
        self.interp.do_K(1.0, 0.0, 0.0, 0.5)
        assert self.interp.graphicstate.scolor == (1.0, 0.0, 0.0, 0.5)
        assert self.interp.graphicstate.scs.name == "DeviceCMYK"

    def test_do_K_invalid_values(self):
        self.interp.do_K(1.0, 0.0, "invalid", 0.5)

    def test_do_k_set_cmyk_fill(self):
        self.interp.do_k(0.0, 1.0, 0.0, 0.25)
        assert self.interp.graphicstate.ncolor == (0.0, 1.0, 0.0, 0.25)
        assert self.interp.graphicstate.ncs.name == "DeviceCMYK"

    def test_do_k_invalid_values(self):
        self.interp.do_k(0.0, 1.0, 0.0, "invalid")

    def test_do_SCN_gray(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["DeviceGray"]
        self.interp.push(0.5)
        self.interp.do_SCN()
        assert self.interp.graphicstate.scolor == 0.5

    def test_do_scn_gray(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        self.interp.push(0.7)
        self.interp.do_scn()
        assert self.interp.graphicstate.ncolor == 0.7

    def test_do_SCN_rgb(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["DeviceRGB"]
        self.interp.push(1.0)
        self.interp.push(0.5)
        self.interp.push(0.0)
        self.interp.do_SCN()
        assert self.interp.graphicstate.scolor == (1.0, 0.5, 0.0)

    def test_do_scn_rgb(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["DeviceRGB"]
        self.interp.push(0.0)
        self.interp.push(0.5)
        self.interp.push(1.0)
        self.interp.do_scn()
        assert self.interp.graphicstate.ncolor == (0.0, 0.5, 1.0)

    def test_do_SCN_cmyk(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["DeviceCMYK"]
        self.interp.push(1.0)
        self.interp.push(0.5)
        self.interp.push(0.25)
        self.interp.push(0.0)
        self.interp.do_SCN()
        assert self.interp.graphicstate.scolor == (1.0, 0.5, 0.25, 0.0)

    def test_do_scn_cmyk(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["DeviceCMYK"]
        self.interp.push(0.0)
        self.interp.push(0.25)
        self.interp.push(0.5)
        self.interp.push(1.0)
        self.interp.do_scn()
        assert self.interp.graphicstate.ncolor == (0.0, 0.25, 0.5, 1.0)

    def test_do_SC_delegates_to_SCN(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["DeviceGray"]
        self.interp.push(0.3)
        self.interp.do_SC()
        assert self.interp.graphicstate.scolor == 0.3

    def test_do_sc_delegates_to_scn(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["DeviceGray"]
        self.interp.push(0.4)
        self.interp.do_sc()
        assert self.interp.graphicstate.ncolor == 0.4


class TestPDFPageInterpreterTextOperators:
    """Tests for text operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_BT_begin_text(self):
        self.interp.textstate.matrix = (2, 0, 0, 2, 100, 100)
        self.interp.textstate.linematrix = (50, 50)
        self.interp.do_BT()
        assert self.interp.textstate.matrix == MATRIX_IDENTITY
        assert self.interp.textstate.linematrix == (0, 0)

    def test_do_ET_end_text(self):
        self.interp.do_ET()

    def test_do_Tc_set_charspace(self):
        self.interp.do_Tc(2.0)
        assert self.interp.textstate.charspace == 2.0

    def test_do_Tc_invalid_value(self):
        self.interp.do_Tc("invalid")

    def test_do_Tw_set_wordspace(self):
        self.interp.do_Tw(3.0)
        assert self.interp.textstate.wordspace == 3.0

    def test_do_Tw_invalid_value(self):
        self.interp.do_Tw("invalid")

    def test_do_Tz_set_scaling(self):
        self.interp.do_Tz(150)
        assert self.interp.textstate.scaling == 150

    def test_do_Tz_invalid_value(self):
        self.interp.do_Tz("invalid")

    def test_do_TL_set_leading(self):
        self.interp.do_TL(14)
        assert self.interp.textstate.leading == -14

    def test_do_TL_invalid_value(self):
        self.interp.do_TL("invalid")

    def test_do_Tr_set_render(self):
        self.interp.do_Tr(1)
        assert self.interp.textstate.render == 1

    def test_do_Tr_invalid_value(self):
        self.interp.do_Tr("invalid")

    def test_do_Ts_set_rise(self):
        self.interp.do_Ts(5)
        assert self.interp.textstate.rise == 5

    def test_do_Ts_invalid_value(self):
        self.interp.do_Ts("invalid")

    def test_do_Td_move_text_position(self):
        self.interp.textstate.matrix = (1, 0, 0, 1, 0, 0)
        self.interp.do_Td(100, 50)
        assert self.interp.textstate.matrix[4] == 100
        assert self.interp.textstate.matrix[5] == 50
        assert self.interp.textstate.linematrix == (0, 0)

    def test_do_Td_invalid_values_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFValueError):
                self.interp.do_Td("invalid", 50)
        finally:
            settings.STRICT = original_strict

    def test_do_TD_move_text_position_set_leading(self):
        self.interp.textstate.matrix = (1, 0, 0, 1, 0, 0)
        self.interp.do_TD(100, -14)
        assert self.interp.textstate.matrix[4] == 100
        assert self.interp.textstate.matrix[5] == -14
        assert self.interp.textstate.leading == -14
        assert self.interp.textstate.linematrix == (0, 0)

    def test_do_Tm_set_text_matrix(self):
        self.interp.do_Tm(2, 0, 0, 2, 100, 200)
        assert self.interp.textstate.matrix == (2, 0, 0, 2, 100, 200)
        assert self.interp.textstate.linematrix == (0, 0)

    def test_do_Tm_invalid_values(self):
        self.interp.do_Tm("invalid", 0, 0, 1, 0, 0)

    def test_do_T_a_move_to_next_line(self):
        self.interp.textstate.matrix = (1, 0, 0, 1, 0, 0)
        self.interp.textstate.leading = -14
        self.interp.do_T_a()
        assert self.interp.textstate.matrix[5] == -14
        assert self.interp.textstate.linematrix == (0, 0)

    def test_do_TJ_show_text(self):
        mock_font = MagicMock()
        self.interp.textstate.font = mock_font
        self.interp.do_TJ([b"Hello", -100, b"World"])
        self.device.render_string.assert_called_once()

    def test_do_TJ_no_font_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            self.interp.textstate.font = None
            with pytest.raises(PDFInterpreterError):
                self.interp.do_TJ([b"Hello"])
        finally:
            settings.STRICT = original_strict

    def test_do_TJ_no_font_non_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            self.interp.textstate.font = None
            self.interp.do_TJ([b"Hello"])
        finally:
            settings.STRICT = original_strict

    def test_do_Tj_show_string(self):
        mock_font = MagicMock()
        self.interp.textstate.font = mock_font
        self.interp.do_Tj(b"Hello")
        self.device.render_string.assert_called_once()

    def test_do__q_quote_operator(self):
        mock_font = MagicMock()
        self.interp.textstate.font = mock_font
        self.interp.textstate.matrix = (1, 0, 0, 1, 0, 0)
        self.interp.textstate.leading = -14
        self.interp.do__q(b"Hello")
        self.device.render_string.assert_called_once()

    def test_do__w_double_quote_operator(self):
        mock_font = MagicMock()
        self.interp.textstate.font = mock_font
        self.interp.do__w(3.0, 0.0, b"Hello")
        assert self.interp.textstate.wordspace == 3.0
        assert self.interp.textstate.charspace == 0.0
        self.device.render_string.assert_called_once()


class TestPDFPageInterpreterMarkedContent:
    """Tests for marked content operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_MP_marked_point(self):
        tag = LIT("Tag")
        self.interp.do_MP(tag)
        self.device.do_tag.assert_called_once_with(tag)

    def test_do_MP_invalid_tag(self):
        self.interp.do_MP("invalid")
        self.device.do_tag.assert_not_called()

    def test_do_DP_marked_point_with_props(self):
        tag = LIT("Tag")
        props = {"Key": "Value"}
        self.interp.do_DP(tag, props)
        self.device.do_tag.assert_called_once_with(tag, props)

    def test_do_DP_invalid_tag(self):
        self.interp.do_DP("invalid", {})
        self.device.do_tag.assert_not_called()

    def test_do_BMC_begin_marked_content(self):
        tag = LIT("Section")
        self.interp.do_BMC(tag)
        self.device.begin_tag.assert_called_once_with(tag)

    def test_do_BMC_invalid_tag(self):
        self.interp.do_BMC("invalid")
        self.device.begin_tag.assert_not_called()

    def test_do_BDC_begin_marked_content_with_props(self):
        tag = LIT("Section")
        props = {"MCID": 1}
        self.interp.do_BDC(tag, props)
        self.device.begin_tag.assert_called_once_with(tag, props)

    def test_do_BDC_invalid_tag(self):
        self.interp.do_BDC("invalid", {})
        self.device.begin_tag.assert_not_called()

    def test_do_EMC_end_marked_content(self):
        self.interp.do_EMC()
        self.device.end_tag.assert_called_once()


class TestPDFPageInterpreterInlineImage:
    """Tests for inline image operators in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_BI(self):
        self.interp.do_BI()

    def test_do_ID(self):
        self.interp.do_ID()

    def test_do_EI_with_stream(self):
        stream = PDFStream({"W": 10, "H": 10}, b"imagedata")
        self.interp.do_EI(stream)
        self.device.begin_figure.assert_called_once()
        self.device.render_image.assert_called_once()
        self.device.end_figure.assert_called_once()

    def test_do_EI_without_dimensions(self):
        stream = PDFStream({}, b"imagedata")
        self.interp.do_EI(stream)
        self.device.begin_figure.assert_not_called()

    def test_do_EI_non_stream(self):
        self.interp.do_EI("not a stream")
        self.device.begin_figure.assert_not_called()


class TestPDFPageInterpreterXObject:
    """Tests for XObject handling in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def test_do_Do_undefined_xobject_strict(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFInterpreterError):
                self.interp.do_Do(LIT("NonExistent"))
        finally:
            settings.STRICT = original_strict

    def test_do_Do_undefined_xobject_non_strict(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            self.interp.do_Do(LIT("NonExistent"))
        finally:
            settings.STRICT = original_strict

    def test_do_Do_image_xobject(self):
        image_stream = PDFStream(
            {"Subtype": LITERAL_IMAGE, "Width": 100, "Height": 50}, b"imagedata"
        )
        image_stream.set_objid(1, 0)
        resources = {"XObject": {"Im1": image_stream}}
        self.interp.init_resources(resources)
        self.interp.init_state(MATRIX_IDENTITY)
        self.interp.do_Do(LIT("Im1"))
        self.device.begin_figure.assert_called_once()
        self.device.render_image.assert_called_once()
        self.device.end_figure.assert_called_once()


class TestPDFPageInterpreterFontHandling:
    """Tests for font handling in PDFPageInterpreter"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_Tf_set_font(self):
        mock_font = MagicMock()
        self.interp.fontmap["F1"] = mock_font
        self.interp.do_Tf(LIT("F1"), 12)
        assert self.interp.textstate.font is mock_font
        assert self.interp.textstate.fontsize == 12

    def test_do_Tf_undefined_font_strict(self):
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFInterpreterError):
                self.interp.do_Tf(LIT("UndefinedFont"), 12)
        finally:
            settings.STRICT = original_strict

    def test_do_Tf_invalid_fontsize(self):
        mock_font = MagicMock()
        self.interp.fontmap["F1"] = mock_font
        self.interp.do_Tf(LIT("F1"), "invalid")


class TestPDFPageInterpreterCompatibility:
    """Tests for compatibility section operators"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_BX_begin_compatibility(self):
        self.interp.do_BX()

    def test_do_EX_end_compatibility(self):
        self.interp.do_EX()


class TestPDFPageInterpreterClipping:
    """Tests for clipping path operators"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_W_set_clipping_nonzero(self):
        self.interp.do_W()

    def test_do_W_a_set_clipping_evenodd(self):
        self.interp.do_W_a()


class TestPDFPageInterpreterShading:
    """Tests for shading operators"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_sh_paint_shading(self):
        self.interp.do_sh(LIT("ShadingName"))


class TestPDFPageInterpreterExecute:
    """Tests for execute method"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def create_stream(self, data: bytes, objid: int = 1) -> PDFStream:
        stream = PDFStream({}, data)
        stream.set_objid(objid, 0)
        return stream

    def test_execute_simple_operators(self):
        stream = self.create_stream(b"q Q ")
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        self.interp.execute([stream])

    def test_execute_circular_reference_detection(self):
        stream = self.create_stream(b"q Q ", objid=1)
        self.interp.parent_stream_ids = {1}
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        self.interp.execute([stream])

    def test_execute_empty_stream(self):
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)
        self.interp.execute([])


class TestPDFPageInterpreterRenderContents:
    """Tests for render_contents method"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def create_stream(self, data: bytes, objid: int = 1) -> PDFStream:
        stream = PDFStream({}, data)
        stream.set_objid(objid, 0)
        return stream

    def test_render_contents_basic(self):
        stream = self.create_stream(b"q Q ")
        ctm = (1, 0, 0, 1, 0, 0)
        self.interp.render_contents({}, [stream], ctm=ctm)
        self.device.set_ctm.assert_called()

    def test_render_contents_with_resources(self):
        stream = self.create_stream(b"0.5 g ")
        resources = {"ProcSet": [LITERAL_PDF, LITERAL_TEXT]}
        self.interp.render_contents(resources, [stream])


class TestPDFPageInterpreterInitResources:
    """Tests for init_resources method"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def test_init_resources_empty(self):
        self.interp.init_resources({})
        assert self.interp.fontmap == {}
        assert self.interp.xobjmap == {}

    def test_init_resources_with_colorspace(self):
        resources = {"ColorSpace": {"CS1": [LIT("DeviceRGB")]}}
        self.interp.init_resources(resources)
        assert "CS1" in self.interp.csmap

    def test_init_resources_with_procset(self):
        resources = {"ProcSet": [LITERAL_PDF, LITERAL_TEXT]}
        self.interp.init_resources(resources)

    def test_init_resources_with_xobject(self):
        stream = PDFStream({"Subtype": LITERAL_IMAGE}, b"data")
        stream.set_objid(1, 0)
        resources = {"XObject": {"Im1": stream}}
        self.interp.init_resources(resources)
        assert "Im1" in self.interp.xobjmap


class TestPDFResourceError:
    """Tests for PDFResourceError exception"""

    def test_exception_message(self):
        with pytest.raises(PDFResourceError) as exc_info:
            raise PDFResourceError("Resource not found")
        assert "Resource not found" in str(exc_info.value)


class TestPDFInterpreterError:
    """Tests for PDFInterpreterError exception"""

    def test_exception_message(self):
        with pytest.raises(PDFInterpreterError) as exc_info:
            raise PDFInterpreterError("Interpreter error")
        assert "Interpreter error" in str(exc_info.value)


class TestColorType:
    """Tests for Color type handling"""

    def test_grayscale_color(self):
        gs = PDFGraphicState()
        gs.ncolor = 0.5
        assert isinstance(gs.ncolor, float)

    def test_rgb_color(self):
        gs = PDFGraphicState()
        gs.ncolor = (1.0, 0.5, 0.0)
        assert isinstance(gs.ncolor, tuple)
        assert len(gs.ncolor) == 3

    def test_cmyk_color(self):
        gs = PDFGraphicState()
        gs.ncolor = (1.0, 0.5, 0.25, 0.0)
        assert isinstance(gs.ncolor, tuple)
        assert len(gs.ncolor) == 4

    def test_pattern_color(self):
        gs = PDFGraphicState()
        gs.ncolor = "PatternName"
        assert isinstance(gs.ncolor, str)

    def test_uncolored_pattern_color(self):
        gs = PDFGraphicState()
        gs.ncolor = ((0.5, 0.5, 0.5), "PatternName")
        assert isinstance(gs.ncolor, tuple)
        assert len(gs.ncolor) == 2


class TestPDFPageInterpreterProcessPage:
    """Tests for process_page method"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)

    def test_process_page_rotation_0(self):
        mock_page = MagicMock()
        mock_page.mediabox = (0, 0, 612, 792)
        mock_page.rotate = 0
        mock_page.resources = {}
        stream = PDFStream({}, b"q Q ")
        stream.set_objid(1, 0)
        mock_page.contents = [stream]

        self.interp.process_page(mock_page)

        self.device.begin_page.assert_called_once()
        self.device.end_page.assert_called_once()

    def test_process_page_rotation_90(self):
        mock_page = MagicMock()
        mock_page.mediabox = (0, 0, 612, 792)
        mock_page.rotate = 90
        mock_page.resources = {}
        stream = PDFStream({}, b"q Q ")
        stream.set_objid(1, 0)
        mock_page.contents = [stream]

        self.interp.process_page(mock_page)

        call_args = self.device.begin_page.call_args
        ctm = call_args[0][1]
        assert ctm[0] == 0
        assert ctm[1] == -1

    def test_process_page_rotation_180(self):
        mock_page = MagicMock()
        mock_page.mediabox = (0, 0, 612, 792)
        mock_page.rotate = 180
        mock_page.resources = {}
        stream = PDFStream({}, b"q Q ")
        stream.set_objid(1, 0)
        mock_page.contents = [stream]

        self.interp.process_page(mock_page)

        call_args = self.device.begin_page.call_args
        ctm = call_args[0][1]
        assert ctm[0] == -1
        assert ctm[3] == -1

    def test_process_page_rotation_270(self):
        mock_page = MagicMock()
        mock_page.mediabox = (0, 0, 612, 792)
        mock_page.rotate = 270
        mock_page.resources = {}
        stream = PDFStream({}, b"q Q ")
        stream.set_objid(1, 0)
        mock_page.contents = [stream]

        self.interp.process_page(mock_page)

        call_args = self.device.begin_page.call_args
        ctm = call_args[0][1]
        assert ctm[0] == 0
        assert ctm[1] == 1


class TestPDFGraphicStateExtendedColors:
    """Extended tests for pattern and special color handling"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_SCN_pattern_colored(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["Pattern"]
        self.interp.push(LIT("MyPattern"))
        self.interp.do_SCN()
        assert self.interp.graphicstate.scolor == "MyPattern"

    def test_do_scn_pattern_colored(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["Pattern"]
        self.interp.push(LIT("MyPattern"))
        self.interp.do_scn()
        assert self.interp.graphicstate.ncolor == "MyPattern"

    def test_do_SCN_pattern_invalid_type(self):
        self.interp.graphicstate.scs = PREDEFINED_COLORSPACE["Pattern"]
        self.interp.push(42)
        self.interp.do_SCN()

    def test_do_scn_pattern_invalid_type(self):
        self.interp.graphicstate.ncs = PREDEFINED_COLORSPACE["Pattern"]
        self.interp.push(42)
        self.interp.do_scn()


class TestPDFPageInterpreterFObsolete:
    """Tests for obsolete F operator"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_F_obsolete_fill(self):
        self.interp.do_F()


class TestContentParserSeek:
    """Tests for PDFContentParser seek functionality"""

    def create_stream(self, data: bytes, objid: int = 1) -> PDFStream:
        stream = PDFStream({}, data)
        stream.set_objid(objid, 0)
        return stream

    def test_seek_basic(self):
        stream = self.create_stream(b"12345678901234567890")
        parser = PDFContentParser([stream])
        parser.seek(0)


class TestContentParserFillbuf:
    """Tests for PDFContentParser fillbuf functionality"""

    def create_stream(self, data: bytes, objid: int = 1) -> PDFStream:
        stream = PDFStream({}, data)
        stream.set_objid(objid, 0)
        return stream

    def test_fillbuf_basic(self):
        stream = self.create_stream(b"test data here ")
        parser = PDFContentParser([stream])
        parser.fillbuf()


class TestPDFPageInterpreterGsOperator:
    """Tests for gs operator (graphics state parameter dictionary)"""

    def setup_method(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = MagicMock(spec=PDFDevice)
        self.interp = PDFPageInterpreter(self.rsrcmgr, self.device)
        self.interp.init_resources({})
        self.interp.init_state(MATRIX_IDENTITY)

    def test_do_gs(self):
        self.interp.do_gs(LIT("GS1"))
