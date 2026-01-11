"""Comprehensive tests for pdfminer/image.py module.

This module tests:
- BMPWriter class for bitmap image creation
- ImageWriter class for exporting images from PDFs
- align32 helper function
- Various image format handling (JPEG, JPEG2000, JBIG2, BMP, raw)
"""

import os
import tempfile
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pdfminer.image import PIL_ERROR_MESSAGE, BMPWriter, ImageWriter, align32
from pdfminer.layout import LTImage
from pdfminer.pdfcolor import (
    LITERAL_DEVICE_CMYK,
    LITERAL_DEVICE_GRAY,
    LITERAL_DEVICE_RGB,
    LITERAL_INLINE_DEVICE_GRAY,
    LITERAL_INLINE_DEVICE_RGB,
)
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdftypes import LITERALS_JBIG2_DECODE
from pdfminer.psparser import LIT


class TestAlign32:
    """Tests for the align32 helper function."""

    def test_align32_zero(self) -> None:
        assert align32(0) == 0

    def test_align32_already_aligned(self) -> None:
        assert align32(4) == 4
        assert align32(8) == 8
        assert align32(16) == 16

    def test_align32_needs_alignment(self) -> None:
        assert align32(1) == 4
        assert align32(2) == 4
        assert align32(3) == 4
        assert align32(5) == 8
        assert align32(6) == 8
        assert align32(7) == 8

    def test_align32_large_values(self) -> None:
        assert align32(100) == 100
        assert align32(101) == 104
        assert align32(102) == 104
        assert align32(103) == 104


class TestBMPWriter:
    """Tests for the BMPWriter class."""

    def test_bmp_writer_1bit(self) -> None:
        """Test 1-bit (black and white) BMP creation."""
        fp = BytesIO()
        width, height = 8, 4
        bmp = BMPWriter(fp, bits=1, width=width, height=height)

        assert bmp.bits == 1
        assert bmp.width == width
        assert bmp.height == height

        fp.seek(0)
        header = fp.read(2)
        assert header == b"BM"

    def test_bmp_writer_8bit(self) -> None:
        """Test 8-bit grayscale BMP creation."""
        fp = BytesIO()
        width, height = 16, 8
        bmp = BMPWriter(fp, bits=8, width=width, height=height)

        assert bmp.bits == 8
        assert bmp.width == width
        assert bmp.height == height

        fp.seek(0)
        header = fp.read(2)
        assert header == b"BM"

    def test_bmp_writer_24bit(self) -> None:
        """Test 24-bit RGB BMP creation."""
        fp = BytesIO()
        width, height = 10, 10
        bmp = BMPWriter(fp, bits=24, width=width, height=height)

        assert bmp.bits == 24
        assert bmp.width == width
        assert bmp.height == height

        fp.seek(0)
        header = fp.read(2)
        assert header == b"BM"

    def test_bmp_writer_invalid_bits(self) -> None:
        """Test that invalid bit depth raises PDFValueError."""
        fp = BytesIO()
        with pytest.raises(PDFValueError):
            BMPWriter(fp, bits=16, width=10, height=10)

        with pytest.raises(PDFValueError):
            BMPWriter(fp, bits=32, width=10, height=10)

    def test_bmp_writer_linesize_calculation(self) -> None:
        """Test that linesize is correctly aligned to 4-byte boundary."""
        fp = BytesIO()
        bmp = BMPWriter(fp, bits=24, width=1, height=1)
        assert bmp.linesize == 4

        fp = BytesIO()
        bmp = BMPWriter(fp, bits=24, width=2, height=1)
        assert bmp.linesize == 8

        fp = BytesIO()
        bmp = BMPWriter(fp, bits=8, width=5, height=1)
        assert bmp.linesize == 8

    def test_bmp_writer_write_line(self) -> None:
        """Test writing image data lines."""
        fp = BytesIO()
        width, height = 4, 2
        bmp = BMPWriter(fp, bits=8, width=width, height=height)

        line_data = b"\x00\x55\xAA\xFF"
        bmp.write_line(0, line_data)
        bmp.write_line(1, line_data)

        fp.seek(bmp.pos0)
        written_data = fp.read(bmp.datasize)
        assert len(written_data) == bmp.datasize

    def test_bmp_writer_header_size(self) -> None:
        """Test that BMP headers are correct size."""
        fp = BytesIO()
        BMPWriter(fp, bits=8, width=10, height=10)

        fp.seek(0)
        file_header = fp.read(14)
        assert len(file_header) == 14
        assert file_header[0:2] == b"BM"

        info_header = fp.read(40)
        assert len(info_header) == 40

    def test_bmp_writer_color_table_1bit(self) -> None:
        """Test 1-bit BMP has correct color table (2 entries)."""
        fp = BytesIO()
        BMPWriter(fp, bits=1, width=8, height=1)

        fp.seek(54)
        color_table = fp.read(8)
        assert color_table == b"\x00\x00\x00\x00\xff\xff\xff\x00"

    def test_bmp_writer_color_table_8bit(self) -> None:
        """Test 8-bit BMP has correct color table (256 entries)."""
        fp = BytesIO()
        BMPWriter(fp, bits=8, width=8, height=1)

        fp.seek(54)
        color_table = fp.read(256 * 4)
        assert len(color_table) == 1024

        assert color_table[0:4] == b"\x00\x00\x00\x00"
        assert color_table[255 * 4 : 256 * 4] == b"\xff\xff\xff\x00"


class MockPDFStream:
    """Mock PDFStream for testing purposes."""

    def __init__(
        self,
        data: bytes = b"",
        width: int = 10,
        height: int = 10,
        bits: int = 8,
        colorspace: Any = None,
        filters: list[tuple[Any, dict[str, Any]]] | None = None,
    ) -> None:
        self._data = data
        self._width = width
        self._height = height
        self._bits = bits
        self._colorspace = colorspace
        self._filters = filters or []
        self.attrs: dict[str, Any] = {}

    def get_data(self) -> bytes:
        return self._data

    def get_any(self, names: tuple[str, ...], default: Any = None) -> Any:
        if names == ("W", "Width"):
            return self._width
        elif names == ("H", "Height"):
            return self._height
        elif names == ("IM", "ImageMask"):
            return None
        elif names == ("BPC", "BitsPerComponent"):
            return self._bits
        elif names == ("CS", "ColorSpace"):
            return self._colorspace
        return default

    def get_filters(self) -> list[tuple[Any, dict[str, Any]]]:
        return self._filters


def create_mock_ltimage(
    name: str = "test_img",
    data: bytes = b"",
    width: int = 10,
    height: int = 10,
    bits: int = 8,
    colorspace: Any = None,
    filters: list[tuple[Any, dict[str, Any]]] | None = None,
) -> LTImage:
    """Create a mock LTImage for testing."""
    stream = MockPDFStream(
        data=data,
        width=width,
        height=height,
        bits=bits,
        colorspace=colorspace,
        filters=filters,
    )
    return LTImage(name=name, stream=stream, bbox=(0, 0, width, height))  # type: ignore[arg-type]


class TestImageWriter:
    """Tests for the ImageWriter class."""

    def test_init_creates_directory(self) -> None:
        """Test that ImageWriter creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "new_subdir")
            assert not os.path.exists(outdir)

            writer = ImageWriter(outdir)
            assert os.path.exists(outdir)
            assert writer.outdir == outdir

    def test_init_existing_directory(self) -> None:
        """Test that ImageWriter works with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            assert writer.outdir == tmpdir

    def test_create_unique_image_name_basic(self) -> None:
        """Test unique image name generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            image = create_mock_ltimage(name="img1")

            name, path = writer._create_unique_image_name(image, ".bmp")
            assert name == "img1.bmp"
            assert path == os.path.join(tmpdir, "img1.bmp")

    def test_create_unique_image_name_collision(self) -> None:
        """Test unique image name generation with existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, "img1.bmp")
            with open(existing_file, "w") as f:
                f.write("existing")

            writer = ImageWriter(tmpdir)
            image = create_mock_ltimage(name="img1")

            name, path = writer._create_unique_image_name(image, ".bmp")
            assert name == "img1.0.bmp"
            assert path == os.path.join(tmpdir, "img1.0.bmp")

    def test_create_unique_image_name_multiple_collisions(self) -> None:
        """Test unique name generation with multiple existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in ["", ".0", ".1", ".2"]:
                existing_file = os.path.join(tmpdir, f"img1{i}.bmp")
                with open(existing_file, "w") as f:
                    f.write("existing")

            writer = ImageWriter(tmpdir)
            image = create_mock_ltimage(name="img1")

            name, path = writer._create_unique_image_name(image, ".bmp")
            assert name == "img1.3.bmp"
            assert path == os.path.join(tmpdir, "img1.3.bmp")

    def test_is_jbig2_image_true(self) -> None:
        """Test JBIG2 detection for JBIG2 encoded images."""
        image = create_mock_ltimage(
            filters=[(LIT("JBIG2Decode"), {})]
        )
        assert ImageWriter._is_jbig2_iamge(image) is True

    def test_is_jbig2_image_false(self) -> None:
        """Test JBIG2 detection for non-JBIG2 images."""
        image = create_mock_ltimage(filters=[])
        assert ImageWriter._is_jbig2_iamge(image) is False

        image = create_mock_ltimage(
            filters=[(LIT("DCTDecode"), {})]
        )
        assert ImageWriter._is_jbig2_iamge(image) is False

    def test_save_raw(self) -> None:
        """Test saving raw image data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            raw_data = b"\x00\x11\x22\x33\x44\x55"
            image = create_mock_ltimage(
                name="raw_img",
                data=raw_data,
                width=2,
                height=3,
                bits=8,
            )

            name = writer._save_raw(image)

            assert name.startswith("raw_img")
            assert name.endswith(".img")
            assert "8.2x3" in name

            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)
            with open(saved_path, "rb") as f:
                assert f.read() == raw_data

    def test_save_bmp_1bit(self) -> None:
        """Test saving 1-bit BMP image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 8, 4
            bytes_per_line = (width + 7) // 8
            data = b"\xFF" * (bytes_per_line * height)

            image = create_mock_ltimage(
                name="bmp1_img",
                data=data,
                width=width,
                height=height,
                bits=1,
            )

            name = writer._save_bmp(image, width, height, bytes_per_line, 1)

            assert name == "bmp1_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

            with open(saved_path, "rb") as f:
                header = f.read(2)
                assert header == b"BM"

    def test_save_bmp_8bit_grayscale(self) -> None:
        """Test saving 8-bit grayscale BMP image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 4, 4
            bytes_per_line = width
            data = bytes(range(16))

            image = create_mock_ltimage(
                name="bmp8_img",
                data=data,
                width=width,
                height=height,
                bits=8,
            )

            name = writer._save_bmp(image, width, height, bytes_per_line, 8)

            assert name == "bmp8_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

    def test_save_bmp_24bit_rgb(self) -> None:
        """Test saving 24-bit RGB BMP image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 2, 2
            bytes_per_line = width * 3
            data = b"\xFF\x00\x00" * 4

            image = create_mock_ltimage(
                name="bmp24_img",
                data=data,
                width=width,
                height=height,
                bits=8,
            )

            name = writer._save_bmp(image, width, height, bytes_per_line, 24)

            assert name == "bmp24_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

    def test_export_image_no_filters_with_pillow(self) -> None:
        """Test exporting image with no filters (raw bytes) requires PIL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 2, 2
            data = b"\x00" * (width * height)

            image = create_mock_ltimage(
                name="no_filter",
                data=data,
                width=width,
                height=height,
                bits=8,
                filters=[],
            )

            pil_mocks = {"PIL": None, "PIL.Image": None, "PIL.ImageOps": None}
            with (
                patch.dict("sys.modules", pil_mocks),
                pytest.raises(ImportError) as exc,
            ):
                writer.export_image(image)
            assert "Pillow" in str(exc.value) or "PIL" in str(exc.value)

    def test_export_image_1bit(self) -> None:
        """Test exporting 1-bit image as BMP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 8, 4
            bytes_per_line = (width + 7) // 8
            data = b"\xFF" * (bytes_per_line * height)

            image = create_mock_ltimage(
                name="bw_img",
                data=data,
                width=width,
                height=height,
                bits=1,
                filters=[(LIT("FlateDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "bw_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

    def test_export_image_8bit_rgb(self) -> None:
        """Test exporting 8-bit RGB image as BMP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 4, 4
            data = b"\xFF\x00\x00" * (width * height)

            image = create_mock_ltimage(
                name="rgb_img",
                data=data,
                width=width,
                height=height,
                bits=8,
                colorspace=LITERAL_DEVICE_RGB,
                filters=[(LIT("FlateDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "rgb_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

    def test_export_image_8bit_gray(self) -> None:
        """Test exporting 8-bit grayscale image as BMP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 4, 4
            data = b"\x80" * (width * height)

            image = create_mock_ltimage(
                name="gray_img",
                data=data,
                width=width,
                height=height,
                bits=8,
                colorspace=LITERAL_DEVICE_GRAY,
                filters=[(LIT("FlateDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "gray_img.bmp"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

    def test_export_image_inline_rgb(self) -> None:
        """Test exporting inline RGB image as BMP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 4, 4
            data = b"\x00\xFF\x00" * (width * height)

            image = create_mock_ltimage(
                name="inline_rgb",
                data=data,
                width=width,
                height=height,
                bits=8,
                colorspace=LITERAL_INLINE_DEVICE_RGB,
                filters=[(LIT("FlateDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "inline_rgb.bmp"

    def test_export_image_inline_gray(self) -> None:
        """Test exporting inline grayscale image as BMP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 4, 4
            data = b"\xCC" * (width * height)

            image = create_mock_ltimage(
                name="inline_gray",
                data=data,
                width=width,
                height=height,
                bits=8,
                colorspace=LITERAL_INLINE_DEVICE_GRAY,
                filters=[(LIT("FlateDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "inline_gray.bmp"

    def test_export_image_flate_decode_fallback(self) -> None:
        """Test FlateDecode images with unhandled colorspace use _save_bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            width, height = 2, 2
            data = b"\x00" * (width * height)

            image = create_mock_ltimage(
                name="flate_img",
                data=data,
                width=width,
                height=height,
                bits=8,
                colorspace=None,
                filters=[(LIT("FlateDecode"), {})],
            )

            pil_mocks = {"PIL": None, "PIL.Image": None}
            with patch.dict("sys.modules", pil_mocks), pytest.raises(ImportError):
                writer.export_image(image)

    def test_export_image_unknown_filter_save_raw(self) -> None:
        """Test that unknown filter combination falls back to raw save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            raw_data = b"\x12\x34\x56\x78"

            image = create_mock_ltimage(
                name="unknown_filter",
                data=raw_data,
                width=2,
                height=2,
                bits=8,
                colorspace=LITERAL_DEVICE_CMYK,
                filters=[(LIT("SomeUnknownFilter"), {}), (LIT("AnotherFilter"), {})],
            )

            name = writer.export_image(image)

            assert "unknown_filter" in name
            assert name.endswith(".img")


class TestImageWriterJPEG:
    """Tests for JPEG image handling."""

    def test_save_jpeg_basic(self) -> None:
        """Test saving basic JPEG image."""
        jpeg_header = b"\xFF\xD8\xFF\xE0"
        jpeg_data = jpeg_header + b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="jpeg_img",
                data=jpeg_data,
                width=10,
                height=10,
                bits=8,
                colorspace=LITERAL_DEVICE_RGB,
                filters=[(LIT("DCTDecode"), {})],
            )

            name = writer._save_jpeg(image)

            assert name == "jpeg_img.jpg"
            saved_path = os.path.join(tmpdir, name)
            assert os.path.exists(saved_path)

            with open(saved_path, "rb") as f:
                saved_data = f.read()
                assert saved_data == jpeg_data

    def test_save_jpeg_cmyk_requires_pillow(self) -> None:
        """Test that CMYK JPEG conversion requires Pillow."""
        jpeg_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="cmyk_jpeg",
                data=jpeg_data,
                width=10,
                height=10,
                bits=8,
                colorspace=LITERAL_DEVICE_CMYK,
                filters=[(LIT("DCTDecode"), {})],
            )

            pil_mocks = {
                "PIL": None, "PIL.Image": None, "PIL.ImageChops": None
            }
            with (
                patch.dict("sys.modules", pil_mocks),
                pytest.raises(ImportError) as exc,
            ):
                writer._save_jpeg(image)
            assert "Pillow" in str(exc.value)

    def test_export_image_dct_decode(self) -> None:
        """Test export_image routes DCT encoded images to JPEG save."""
        jpeg_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="dct_img",
                data=jpeg_data,
                width=10,
                height=10,
                bits=8,
                colorspace=LITERAL_DEVICE_RGB,
                filters=[(LIT("DCTDecode"), {})],
            )

            name = writer.export_image(image)

            assert name == "dct_img.jpg"

    def test_export_image_dct_abbreviation(self) -> None:
        """Test export_image with DCT abbreviation filter."""
        jpeg_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="dct_abbrev",
                data=jpeg_data,
                width=10,
                height=10,
                bits=8,
                filters=[(LIT("DCT"), {})],
            )

            name = writer.export_image(image)

            assert name == "dct_abbrev.jpg"


class TestImageWriterJPEG2000:
    """Tests for JPEG 2000 image handling."""

    def test_save_jpeg2000_requires_pillow(self) -> None:
        """Test that JPEG 2000 save requires Pillow."""
        jp2_data = b"\x00\x00\x00\x0C\x6A\x50\x20\x20" + b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="jp2_img",
                data=jp2_data,
                width=10,
                height=10,
                bits=8,
                filters=[(LIT("JPXDecode"), {})],
            )

            with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
                with pytest.raises(ImportError) as exc_info:
                    writer._save_jpeg2000(image)
                assert "Pillow" in str(exc_info.value)

    def test_export_image_jpx_decode(self) -> None:
        """Test export_image routes JPX encoded images to JPEG 2000 save."""
        jp2_data = b"\x00\x00\x00\x0C\x6A\x50\x20\x20" + b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="jpx_img",
                data=jp2_data,
                width=10,
                height=10,
                bits=8,
                filters=[(LIT("JPXDecode"), {})],
            )

            pil_mocks = {"PIL": None, "PIL.Image": None}
            with patch.dict("sys.modules", pil_mocks), pytest.raises(ImportError):
                writer.export_image(image)


class TestImageWriterJBIG2:
    """Tests for JBIG2 image handling."""

    def test_is_jbig2_image_with_jbig2decode(self) -> None:
        """Test JBIG2 detection with JBIG2Decode filter."""
        for filter_name in LITERALS_JBIG2_DECODE:
            image = create_mock_ltimage(
                filters=[(filter_name, {})]
            )
            assert ImageWriter._is_jbig2_iamge(image) is True

    def test_is_jbig2_image_with_multiple_filters(self) -> None:
        """Test JBIG2 detection when JBIG2Decode is among multiple filters."""
        image = create_mock_ltimage(
            filters=[
                (LIT("FlateDecode"), {}),
                (LIT("JBIG2Decode"), {}),
            ]
        )
        assert ImageWriter._is_jbig2_iamge(image) is True

    def test_save_jbig2_multiple_globals_error(self) -> None:
        """Test that multiple JBIG2Globals raises error."""
        mock_global1 = MagicMock()
        mock_global1.resolve.return_value = MagicMock()
        mock_global1.resolve.return_value.get_data.return_value = b"global1"

        mock_global2 = MagicMock()
        mock_global2.resolve.return_value = MagicMock()
        mock_global2.resolve.return_value.get_data.return_value = b"global2"

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            stream = MockPDFStream(
                data=b"\x00" * 20,
                width=8,
                height=8,
                bits=1,
                filters=[
                    (LIT("JBIG2Decode"), {"JBIG2Globals": mock_global1}),
                    (LIT("JBIG2Decode"), {"JBIG2Globals": mock_global2}),
                ],
            )
            image = LTImage(name="jbig2_multi", stream=stream, bbox=(0, 0, 8, 8))  # type: ignore[arg-type]

            with pytest.raises(PDFValueError) as exc_info:
                writer._save_jbig2(image)
            assert "more than one JBIG2Globals" in str(exc_info.value)


class TestImageWriterSaveBytes:
    """Tests for _save_bytes method."""

    def test_save_bytes_requires_pillow(self) -> None:
        """Test that _save_bytes requires Pillow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)
            data = b"\x00" * 100

            image = create_mock_ltimage(
                name="bytes_img",
                data=data,
                width=10,
                height=10,
                bits=8,
            )

            pil_mocks = {"PIL": None, "PIL.Image": None, "PIL.ImageOps": None}
            with (
                patch.dict("sys.modules", pil_mocks),
                pytest.raises(ImportError) as exc,
            ):
                writer._save_bytes(image)
            assert "Pillow" in str(exc.value)


class TestPILErrorMessage:
    """Tests for PIL error message constant."""

    def test_pil_error_message_content(self) -> None:
        """Test that PIL error message contains helpful information."""
        assert "Pillow" in PIL_ERROR_MESSAGE
        assert "pip install" in PIL_ERROR_MESSAGE
        assert "pdfminer.six[image]" in PIL_ERROR_MESSAGE


class TestImageWriterEdgeCases:
    """Edge case tests for ImageWriter."""

    def test_empty_image_data(self) -> None:
        """Test handling of empty image data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="empty_img",
                data=b"",
                width=0,
                height=0,
                bits=8,
            )

            name = writer._save_raw(image)
            assert "empty_img" in name

    def test_very_large_dimensions(self) -> None:
        """Test image with large dimension values in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="large_img",
                data=b"\x00" * 10,
                width=10000,
                height=10000,
                bits=16,
            )

            name = writer._save_raw(image)
            assert "10000x10000" in name
            assert "16" in name

    def test_special_characters_in_name(self) -> None:
        """Test image with special characters in name are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="img_test-123",
                data=b"\x00" * 4,
                width=2,
                height=2,
                bits=8,
            )

            name, _path = writer._create_unique_image_name(image, ".bmp")
            assert name == "img_test-123.bmp"

    def test_multiple_filter_chain_last_filter_dct(self) -> None:
        """Test that filter chain uses last filter for format detection."""
        jpeg_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ImageWriter(tmpdir)

            image = create_mock_ltimage(
                name="chained_img",
                data=jpeg_data,
                width=10,
                height=10,
                bits=8,
                filters=[
                    (LIT("FlateDecode"), {}),
                    (LIT("DCTDecode"), {}),
                ],
            )

            name = writer.export_image(image)
            assert name.endswith(".jpg")


class TestBMPWriterDataSize:
    """Tests for BMP data size calculations."""

    def test_datasize_calculation_1bit(self) -> None:
        """Test data size calculation for 1-bit images."""
        fp = BytesIO()
        width, height = 32, 10
        bmp = BMPWriter(fp, bits=1, width=width, height=height)

        expected_linesize = 4
        expected_datasize = expected_linesize * height
        assert bmp.datasize == expected_datasize

    def test_datasize_calculation_8bit(self) -> None:
        """Test data size calculation for 8-bit images."""
        fp = BytesIO()
        width, height = 10, 10
        bmp = BMPWriter(fp, bits=8, width=width, height=height)

        expected_linesize = 12
        expected_datasize = expected_linesize * height
        assert bmp.datasize == expected_datasize

    def test_datasize_calculation_24bit(self) -> None:
        """Test data size calculation for 24-bit images."""
        fp = BytesIO()
        width, height = 10, 10
        bmp = BMPWriter(fp, bits=24, width=width, height=height)

        expected_linesize = 32
        expected_datasize = expected_linesize * height
        assert bmp.datasize == expected_datasize

    def test_pos0_pos1_relationship(self) -> None:
        """Test that pos0 and pos1 are correctly set."""
        fp = BytesIO()
        width, height = 8, 4
        bmp = BMPWriter(fp, bits=8, width=width, height=height)

        assert bmp.pos1 == bmp.pos0 + bmp.datasize
        assert bmp.pos1 - bmp.pos0 == bmp.linesize * height
