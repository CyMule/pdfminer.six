#!/usr/bin/env python3
"""Generate golden outputs from sample PDFs for regression testing.

This script extracts text and layout information from all sample PDFs
and saves them to a JSON file. This is used to verify that changes
(especially Rust ports) don't alter the output behavior.

Usage:
    python tests/generate_golden_outputs.py
"""

import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTComponent,
    LTContainer,
    LTCurve,
    LTFigure,
    LTImage,
    LTLine,
    LTPage,
    LTRect,
    LTTextBox,
    LTTextLine,
)

# Paths
SAMPLES_DIR = Path(__file__).parent.parent / "samples"
GOLDEN_OUTPUT_FILE = Path(__file__).parent / "golden_outputs.json"

# Known passwords for encrypted sample PDFs
KNOWN_PASSWORDS: dict[str, str] = {
    "samples/encryption/aes-128.pdf": "foo",
    "samples/encryption/aes-128-m.pdf": "foo",
    "samples/encryption/aes-256.pdf": "foo",
    "samples/encryption/aes-256-m.pdf": "foo",
    "samples/encryption/aes-256-r6.pdf": "usersecret",
    "samples/encryption/rc4-40.pdf": "foo",
    "samples/encryption/rc4-128.pdf": "foo",
}


def get_sample_pdfs() -> list[Path]:
    """Get all sample PDF files, sorted for reproducibility."""
    pdfs = list(SAMPLES_DIR.rglob("*.pdf"))
    return sorted(pdfs)


def serialize_bbox(bbox: tuple[float, float, float, float]) -> list[float]:
    """Serialize a bounding box, rounding to avoid floating point issues."""
    return [round(v, 4) for v in bbox]


def extract_layout_summary(page: LTPage) -> dict[str, Any]:
    """Extract a summary of the layout structure from a page."""
    summary: dict[str, Any] = {
        "bbox": serialize_bbox(page.bbox),
        "width": round(page.width, 4),
        "height": round(page.height, 4),
        "rotation": getattr(page, "rotate", 0),
        "elements": {
            "text_boxes": 0,
            "text_lines": 0,
            "chars": 0,
            "figures": 0,
            "images": 0,
            "curves": 0,
            "lines": 0,
            "rects": 0,
        },
        "text_boxes_sample": [],
    }

    def count_elements(obj: LTComponent) -> None:
        if isinstance(obj, LTTextBox):
            summary["elements"]["text_boxes"] += 1
            if len(summary["text_boxes_sample"]) < 5:
                summary["text_boxes_sample"].append({
                    "bbox": serialize_bbox(obj.bbox),
                    "text": obj.get_text()[:200],
                })
        elif isinstance(obj, LTTextLine):
            summary["elements"]["text_lines"] += 1
        elif isinstance(obj, LTChar):
            summary["elements"]["chars"] += 1
        elif isinstance(obj, LTFigure):
            summary["elements"]["figures"] += 1
        elif isinstance(obj, LTImage):
            summary["elements"]["images"] += 1
        elif isinstance(obj, LTLine):
            summary["elements"]["lines"] += 1
        elif isinstance(obj, LTRect):
            summary["elements"]["rects"] += 1
        elif isinstance(obj, LTCurve):
            summary["elements"]["curves"] += 1

        if isinstance(obj, LTContainer):
            for child in obj:
                count_elements(child)

    for element in page:
        count_elements(element)

    return summary


def process_pdf(pdf_path: Path) -> dict[str, Any] | None:
    """Process a single PDF and return its golden output data."""
    relative_path = pdf_path.relative_to(SAMPLES_DIR.parent)
    relative_str = str(relative_path)

    # Check if we have a known password for this PDF
    password = KNOWN_PASSWORDS.get(relative_str, "")

    try:
        # Extract text
        text = extract_text(str(pdf_path), password=password)
        text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

        # Extract layout summary for each page
        pages_summary = []
        for page in extract_pages(str(pdf_path), password=password):
            pages_summary.append(extract_layout_summary(page))

        return {
            "path": relative_str,
            "password": password if password else None,
            "text_length": len(text),
            "text_hash": text_hash,
            "text_preview": text[:500] if text else "",
            "num_pages": len(pages_summary),
            "pages": pages_summary,
            "error": None,
        }

    except Exception as e:
        # Some PDFs may fail to parse - record the error
        return {
            "path": relative_str,
            "password": password if password else None,
            "text_length": 0,
            "text_hash": None,
            "text_preview": "",
            "num_pages": 0,
            "pages": [],
            "error": f"{type(e).__name__}: {e}",
        }


def generate_golden_outputs() -> dict[str, Any]:
    """Generate golden outputs for all sample PDFs."""
    pdfs = get_sample_pdfs()
    print(f"Processing {len(pdfs)} PDF files...")

    results: dict[str, Any] = {
        "version": "1.0",
        "generator": "pdfminer.six golden output generator",
        "total_files": len(pdfs),
        "files": {},
    }

    successful = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs, 1):
        relative_path = str(pdf_path.relative_to(SAMPLES_DIR.parent))
        print(f"[{i}/{len(pdfs)}] {relative_path}...", end=" ", flush=True)

        result = process_pdf(pdf_path)
        if result:
            results["files"][relative_path] = result
            if result["error"]:
                print(f"ERROR: {result['error']}")
                failed += 1
            else:
                print(f"OK ({result['num_pages']} pages, {result['text_length']} chars)")
                successful += 1
        else:
            print("SKIPPED")

    results["successful"] = successful
    results["failed"] = failed

    return results


def main() -> int:
    """Main entry point."""
    print("Generating golden outputs for pdfminer.six regression testing")
    print("=" * 60)

    results = generate_golden_outputs()

    print("=" * 60)
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Writing to {GOLDEN_OUTPUT_FILE}...")

    with open(GOLDEN_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
