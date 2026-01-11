"""Comprehensive tests for pdfminer/pdfcolor.py"""

import pytest

from pdfminer.pdfcolor import (
    LITERAL_DEVICE_CMYK,
    LITERAL_DEVICE_GRAY,
    LITERAL_DEVICE_RGB,
    LITERAL_INLINE_DEVICE_CMYK,
    LITERAL_INLINE_DEVICE_GRAY,
    LITERAL_INLINE_DEVICE_RGB,
    PDFColorSpace,
    PREDEFINED_COLORSPACE,
)
from pdfminer.psparser import LIT


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


class TestPDFColorSpace:
    """Tests for PDFColorSpace class"""

    def test_init_with_name_and_components(self):
        cs = PDFColorSpace("TestSpace", 3)
        assert cs.name == "TestSpace"
        assert cs.ncomponents == 3

    def test_init_grayscale(self):
        cs = PDFColorSpace("DeviceGray", 1)
        assert cs.name == "DeviceGray"
        assert cs.ncomponents == 1

    def test_init_rgb(self):
        cs = PDFColorSpace("DeviceRGB", 3)
        assert cs.name == "DeviceRGB"
        assert cs.ncomponents == 3

    def test_init_cmyk(self):
        cs = PDFColorSpace("DeviceCMYK", 4)
        assert cs.name == "DeviceCMYK"
        assert cs.ncomponents == 4

    def test_repr(self):
        cs = PDFColorSpace("TestSpace", 2)
        repr_str = repr(cs)
        assert "PDFColorSpace" in repr_str
        assert "TestSpace" in repr_str
        assert "ncomponents=2" in repr_str

    def test_repr_includes_name(self):
        cs = PDFColorSpace("CustomColorSpace", 5)
        assert "CustomColorSpace" in repr(cs)

    def test_zero_components(self):
        cs = PDFColorSpace("EmptySpace", 0)
        assert cs.ncomponents == 0

    def test_negative_components(self):
        cs = PDFColorSpace("NegativeSpace", -1)
        assert cs.ncomponents == -1


class TestPredefinedColorspaces:
    """Tests for PREDEFINED_COLORSPACE dictionary"""

    def test_device_gray_exists(self):
        assert "DeviceGray" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["DeviceGray"]
        assert cs.name == "DeviceGray"
        assert cs.ncomponents == 1

    def test_device_rgb_exists(self):
        assert "DeviceRGB" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["DeviceRGB"]
        assert cs.name == "DeviceRGB"
        assert cs.ncomponents == 3

    def test_device_cmyk_exists(self):
        assert "DeviceCMYK" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["DeviceCMYK"]
        assert cs.name == "DeviceCMYK"
        assert cs.ncomponents == 4

    def test_cal_rgb_exists(self):
        assert "CalRGB" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["CalRGB"]
        assert cs.name == "CalRGB"
        assert cs.ncomponents == 3

    def test_cal_gray_exists(self):
        assert "CalGray" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["CalGray"]
        assert cs.name == "CalGray"
        assert cs.ncomponents == 1

    def test_lab_exists(self):
        assert "Lab" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["Lab"]
        assert cs.name == "Lab"
        assert cs.ncomponents == 3

    def test_separation_exists(self):
        assert "Separation" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["Separation"]
        assert cs.name == "Separation"
        assert cs.ncomponents == 1

    def test_indexed_exists(self):
        assert "Indexed" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["Indexed"]
        assert cs.name == "Indexed"
        assert cs.ncomponents == 1

    def test_pattern_exists(self):
        assert "Pattern" in PREDEFINED_COLORSPACE
        cs = PREDEFINED_COLORSPACE["Pattern"]
        assert cs.name == "Pattern"
        assert cs.ncomponents == 1

    def test_predefined_count(self):
        assert len(PREDEFINED_COLORSPACE) == 9

    def test_predefined_is_ordered(self):
        keys = list(PREDEFINED_COLORSPACE.keys())
        assert keys[0] == "DeviceGray"

    def test_all_values_are_pdfcolorspace(self):
        for name, cs in PREDEFINED_COLORSPACE.items():
            assert isinstance(cs, PDFColorSpace)
            assert cs.name == name


class TestDeviceLiterals:
    """Tests for device literal constants"""

    def test_literal_device_gray(self):
        assert isinstance(LITERAL_DEVICE_GRAY, type(LIT("test")))
        assert LITERAL_DEVICE_GRAY.name == "DeviceGray"

    def test_literal_device_rgb(self):
        assert isinstance(LITERAL_DEVICE_RGB, type(LIT("test")))
        assert LITERAL_DEVICE_RGB.name == "DeviceRGB"

    def test_literal_device_cmyk(self):
        assert isinstance(LITERAL_DEVICE_CMYK, type(LIT("test")))
        assert LITERAL_DEVICE_CMYK.name == "DeviceCMYK"


class TestInlineDeviceLiterals:
    """Tests for inline device literal constants (abbreviations for inline images)"""

    def test_literal_inline_device_gray(self):
        assert isinstance(LITERAL_INLINE_DEVICE_GRAY, type(LIT("test")))
        assert LITERAL_INLINE_DEVICE_GRAY.name == "G"

    def test_literal_inline_device_rgb(self):
        assert isinstance(LITERAL_INLINE_DEVICE_RGB, type(LIT("test")))
        assert LITERAL_INLINE_DEVICE_RGB.name == "RGB"

    def test_literal_inline_device_cmyk(self):
        assert isinstance(LITERAL_INLINE_DEVICE_CMYK, type(LIT("test")))
        assert LITERAL_INLINE_DEVICE_CMYK.name == "CMYK"

    def test_inline_abbreviations_differ_from_full_names(self):
        assert LITERAL_INLINE_DEVICE_GRAY.name != LITERAL_DEVICE_GRAY.name
        assert LITERAL_INLINE_DEVICE_RGB.name != LITERAL_DEVICE_RGB.name
        assert LITERAL_INLINE_DEVICE_CMYK.name != LITERAL_DEVICE_CMYK.name


class TestColorSpaceComponentCounts:
    """Tests verifying correct component counts for standard colorspaces"""

    @pytest.mark.parametrize(
        "name,expected_components",
        [
            ("DeviceGray", 1),
            ("CalGray", 1),
            ("Separation", 1),
            ("Indexed", 1),
            ("Pattern", 1),
            ("DeviceRGB", 3),
            ("CalRGB", 3),
            ("Lab", 3),
            ("DeviceCMYK", 4),
        ],
    )
    def test_component_counts(self, name, expected_components):
        cs = PREDEFINED_COLORSPACE[name]
        assert cs.ncomponents == expected_components


class TestColorSpaceLiteralEquality:
    """Tests for literal equality and identity"""

    def test_device_gray_literal_equality(self):
        lit1 = LIT("DeviceGray")
        assert LITERAL_DEVICE_GRAY == lit1

    def test_device_rgb_literal_equality(self):
        lit1 = LIT("DeviceRGB")
        assert LITERAL_DEVICE_RGB == lit1

    def test_device_cmyk_literal_equality(self):
        lit1 = LIT("DeviceCMYK")
        assert LITERAL_DEVICE_CMYK == lit1

    def test_inline_gray_literal_equality(self):
        lit1 = LIT("G")
        assert LITERAL_INLINE_DEVICE_GRAY == lit1

    def test_inline_rgb_literal_equality(self):
        lit1 = LIT("RGB")
        assert LITERAL_INLINE_DEVICE_RGB == lit1

    def test_inline_cmyk_literal_equality(self):
        lit1 = LIT("CMYK")
        assert LITERAL_INLINE_DEVICE_CMYK == lit1
