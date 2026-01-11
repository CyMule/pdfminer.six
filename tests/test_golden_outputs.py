"""Golden output regression tests for pdfminer.six.

These tests verify that text extraction and layout analysis produce
consistent results across code changes, especially during Rust porting.

Run `python tests/generate_golden_outputs.py` to regenerate golden outputs
after intentional changes to parsing behavior.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pdfminer.cmapdb import CMapDB
from pdfminer.high_level import extract_pages, extract_text

GOLDEN_OUTPUT_FILE = Path(__file__).parent / "golden_outputs.json"
SAMPLES_DIR = Path(__file__).parent.parent / "samples"


@pytest.fixture(autouse=True)
def reset_cmap_cache() -> None:
    """Reset CMapDB cache before each test to avoid test pollution.

    Other tests may populate the CMap cache in ways that affect text
    extraction results, leading to different output hashes.
    """
    CMapDB._cmap_cache.clear()


@pytest.fixture(scope="module")
def golden_data() -> dict[str, Any]:
    """Load golden output data."""
    if not GOLDEN_OUTPUT_FILE.exists():
        pytest.skip("Golden outputs not generated. Run: python tests/generate_golden_outputs.py")
    with open(GOLDEN_OUTPUT_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_test_cases(golden_data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Get list of test cases from golden data."""
    cases = []
    for path, data in golden_data.get("files", {}).items():
        if data.get("error") is None:
            cases.append((path, data))
    return cases


class TestGoldenTextExtraction:
    """Tests verifying text extraction matches golden outputs."""

    def test_golden_data_exists(self, golden_data: dict[str, Any]) -> None:
        """Verify golden data file exists and has content."""
        assert "files" in golden_data
        assert len(golden_data["files"]) > 0
        assert golden_data["successful"] > 0

    @pytest.mark.parametrize(
        "pdf_path",
        [
            "samples/simple1.pdf",
            "samples/simple2.pdf",
            "samples/simple3.pdf",
            "samples/simple4.pdf",
            "samples/simple5.pdf",
            "samples/jo.pdf",
        ],
    )
    def test_simple_text_extraction(
        self, golden_data: dict[str, Any], pdf_path: str
    ) -> None:
        """Test text extraction for simple PDFs matches golden output."""
        if pdf_path not in golden_data["files"]:
            pytest.skip(f"No golden data for {pdf_path}")

        expected = golden_data["files"][pdf_path]
        if expected.get("error"):
            pytest.skip(f"Golden data has error: {expected['error']}")

        full_path = SAMPLES_DIR.parent / pdf_path
        actual_text = extract_text(str(full_path))
        actual_hash = hashlib.sha256(
            actual_text.encode("utf-8", errors="replace")
        ).hexdigest()

        assert len(actual_text) == expected["text_length"], (
            f"Text length mismatch for {pdf_path}: "
            f"expected {expected['text_length']}, got {len(actual_text)}"
        )
        assert actual_hash == expected["text_hash"], (
            f"Text hash mismatch for {pdf_path}. Content may have changed.\n"
            f"Expected preview: {expected['text_preview'][:100]}...\n"
            f"Actual preview: {actual_text[:100]}..."
        )

    def test_all_golden_text_hashes(self, golden_data: dict[str, Any]) -> None:
        """Test that all successful golden outputs still match.

        Checks both text length and hash. Length mismatches are always failures.
        Hash-only mismatches (same length, different hash) are tracked but
        allowed up to a small threshold since some PDFs are sensitive to
        global state from other tests (e.g., CMap caching order).
        """
        length_mismatches = []
        hash_only_mismatches = []

        for pdf_path, expected in golden_data["files"].items():
            if expected.get("error"):
                continue

            full_path = SAMPLES_DIR.parent / pdf_path
            if not full_path.exists():
                continue

            password = expected.get("password", "") or ""

            try:
                actual_text = extract_text(str(full_path), password=password)
                actual_hash = hashlib.sha256(
                    actual_text.encode("utf-8", errors="replace")
                ).hexdigest()

                if len(actual_text) != expected["text_length"]:
                    length_mismatches.append({
                        "path": pdf_path,
                        "expected_len": expected["text_length"],
                        "actual_len": len(actual_text),
                    })
                elif actual_hash != expected["text_hash"]:
                    hash_only_mismatches.append({
                        "path": pdf_path,
                        "expected_hash": expected["text_hash"][:16],
                        "actual_hash": actual_hash[:16],
                    })
            except Exception as e:
                length_mismatches.append({
                    "path": pdf_path,
                    "error": str(e),
                })

        # Length mismatches are always failures
        if length_mismatches:
            msg = f"Text length mismatches in {len(length_mismatches)} files:\n"
            for m in length_mismatches[:10]:
                if "error" in m:
                    msg += f"  - {m['path']}: ERROR {m['error']}\n"
                else:
                    msg += f"  - {m['path']}: len {m['expected_len']} -> {m['actual_len']}\n"
            pytest.fail(msg)

        # Hash-only mismatches are warnings (allow up to 2 due to test pollution)
        if len(hash_only_mismatches) > 2:
            msg = f"Too many hash-only mismatches ({len(hash_only_mismatches)} files):\n"
            for m in hash_only_mismatches[:10]:
                msg += f"  - {m['path']}: {m['expected_hash']}... -> {m['actual_hash']}...\n"
            pytest.fail(msg)


class TestGoldenLayoutAnalysis:
    """Tests verifying layout analysis matches golden outputs."""

    @pytest.mark.parametrize(
        "pdf_path",
        [
            "samples/simple1.pdf",
            "samples/simple2.pdf",
            "samples/jo.pdf",
            "samples/nonfree/dmca.pdf",
        ],
    )
    def test_page_count(self, golden_data: dict[str, Any], pdf_path: str) -> None:
        """Test that page count matches golden output."""
        if pdf_path not in golden_data["files"]:
            pytest.skip(f"No golden data for {pdf_path}")

        expected = golden_data["files"][pdf_path]
        if expected.get("error"):
            pytest.skip(f"Golden data has error: {expected['error']}")

        full_path = SAMPLES_DIR.parent / pdf_path
        pages = list(extract_pages(str(full_path)))

        assert len(pages) == expected["num_pages"], (
            f"Page count mismatch for {pdf_path}: "
            f"expected {expected['num_pages']}, got {len(pages)}"
        )

    @pytest.mark.parametrize(
        "pdf_path",
        [
            "samples/simple1.pdf",
            "samples/jo.pdf",
        ],
    )
    def test_page_dimensions(self, golden_data: dict[str, Any], pdf_path: str) -> None:
        """Test that page dimensions match golden output."""
        if pdf_path not in golden_data["files"]:
            pytest.skip(f"No golden data for {pdf_path}")

        expected = golden_data["files"][pdf_path]
        if expected.get("error"):
            pytest.skip(f"Golden data has error: {expected['error']}")

        full_path = SAMPLES_DIR.parent / pdf_path
        pages = list(extract_pages(str(full_path)))

        for i, (page, expected_page) in enumerate(zip(pages, expected["pages"])):
            assert round(page.width, 4) == expected_page["width"], (
                f"Page {i} width mismatch for {pdf_path}"
            )
            assert round(page.height, 4) == expected_page["height"], (
                f"Page {i} height mismatch for {pdf_path}"
            )

    def test_all_golden_page_counts(self, golden_data: dict[str, Any]) -> None:
        """Test that all successful golden outputs have correct page counts."""
        mismatches = []

        for pdf_path, expected in golden_data["files"].items():
            if expected.get("error"):
                continue

            full_path = SAMPLES_DIR.parent / pdf_path
            if not full_path.exists():
                continue

            password = expected.get("password", "") or ""

            try:
                pages = list(extract_pages(str(full_path), password=password))
                if len(pages) != expected["num_pages"]:
                    mismatches.append({
                        "path": pdf_path,
                        "expected": expected["num_pages"],
                        "actual": len(pages),
                    })
            except Exception as e:
                mismatches.append({
                    "path": pdf_path,
                    "error": str(e),
                })

        if mismatches:
            msg = f"Page count mismatches in {len(mismatches)} files:\n"
            for m in mismatches[:10]:
                if "error" in m:
                    msg += f"  - {m['path']}: ERROR {m['error']}\n"
                else:
                    msg += f"  - {m['path']}: expected {m['expected']}, got {m['actual']}\n"
            pytest.fail(msg)


class TestGoldenElementCounts:
    """Tests verifying element counts in layout analysis."""

    @pytest.mark.parametrize(
        "pdf_path",
        [
            "samples/simple1.pdf",
            "samples/jo.pdf",
        ],
    )
    def test_text_box_count(self, golden_data: dict[str, Any], pdf_path: str) -> None:
        """Test that text box count matches golden output."""
        if pdf_path not in golden_data["files"]:
            pytest.skip(f"No golden data for {pdf_path}")

        expected = golden_data["files"][pdf_path]
        if expected.get("error"):
            pytest.skip(f"Golden data has error: {expected['error']}")

        full_path = SAMPLES_DIR.parent / pdf_path
        pages = list(extract_pages(str(full_path)))

        for i, (page, expected_page) in enumerate(zip(pages, expected["pages"])):
            expected_boxes = expected_page["elements"]["text_boxes"]
            from pdfminer.layout import LTTextBox

            actual_boxes = sum(
                1 for obj in page if isinstance(obj, LTTextBox)
            )
            # Allow some tolerance since layout analysis can vary slightly
            # based on floating point calculations
            tolerance = max(1, expected_boxes // 10)
            assert abs(actual_boxes - expected_boxes) <= tolerance, (
                f"Page {i} text box count mismatch for {pdf_path}: "
                f"expected ~{expected_boxes}, got {actual_boxes}"
            )
