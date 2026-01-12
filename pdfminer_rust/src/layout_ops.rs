//! Layout grouping operations for PDF text analysis.
//!
//! This module provides efficient batch computation of text line grouping decisions.

use pyo3::prelude::*;

/// Bounding box representation: (x0, y0, x1, y1)
type BBox = (f64, f64, f64, f64);

/// Compute vertical overlap between two bboxes.
#[inline]
fn is_voverlap(a: &BBox, b: &BBox) -> bool {
    b.1 <= a.3 && a.1 <= b.3
}

/// Compute amount of vertical overlap.
#[inline]
fn voverlap(a: &BBox, b: &BBox) -> f64 {
    if is_voverlap(a, b) {
        (a.3 - b.1).abs().min((b.3 - a.1).abs())
    } else {
        0.0
    }
}

/// Compute horizontal distance between two bboxes.
#[inline]
fn hdistance(a: &BBox, b: &BBox) -> f64 {
    if b.0 <= a.2 && a.0 <= b.2 {
        // Overlapping
        0.0
    } else {
        (a.0 - b.2).abs().min((a.2 - b.0).abs())
    }
}

/// Compute horizontal overlap check.
#[inline]
fn is_hoverlap(a: &BBox, b: &BBox) -> bool {
    b.0 <= a.2 && a.0 <= b.2
}

/// Compute amount of horizontal overlap.
#[inline]
fn hoverlap(a: &BBox, b: &BBox) -> f64 {
    if is_hoverlap(a, b) {
        (a.2 - b.0).abs().min((b.2 - a.0).abs())
    } else {
        0.0
    }
}

/// Compute vertical distance between two bboxes.
#[inline]
fn vdistance(a: &BBox, b: &BBox) -> f64 {
    if is_voverlap(a, b) {
        0.0
    } else {
        (a.1 - b.3).abs().min((a.3 - b.1).abs())
    }
}

/// Width of a bbox.
#[inline]
fn width(bbox: &BBox) -> f64 {
    bbox.2 - bbox.0
}

/// Height of a bbox.
#[inline]
fn height(bbox: &BBox) -> f64 {
    bbox.3 - bbox.1
}

/// Alignment decision for a pair of objects.
/// 0 = no alignment
/// 1 = horizontal alignment
/// 2 = vertical alignment
/// 3 = both (should not happen in practice)
type AlignmentDecision = u8;

const ALIGN_NONE: u8 = 0;
const ALIGN_HORIZONTAL: u8 = 1;
const ALIGN_VERTICAL: u8 = 2;

/// Batch layout operations for PDF text grouping.
#[pyclass]
pub struct LayoutOps;

#[pymethods]
impl LayoutOps {
    #[new]
    pub fn new() -> Self {
        LayoutOps
    }

    /// Compute alignment decisions for adjacent pairs of bounding boxes.
    ///
    /// For each adjacent pair (i, i+1), computes whether they should be
    /// grouped horizontally or vertically based on LAParams thresholds.
    ///
    /// Args:
    ///     bboxes: List of (x0, y0, x1, y1) bounding boxes
    ///     line_overlap: LAParams.line_overlap threshold
    ///     char_margin: LAParams.char_margin threshold
    ///     detect_vertical: LAParams.detect_vertical flag
    ///
    /// Returns:
    ///     List of alignment decisions for each adjacent pair:
    ///     0 = no alignment, 1 = horizontal, 2 = vertical
    #[staticmethod]
    pub fn compute_alignments(
        bboxes: Vec<BBox>,
        line_overlap: f64,
        char_margin: f64,
        detect_vertical: bool,
    ) -> Vec<u8> {
        if bboxes.len() < 2 {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(bboxes.len() - 1);

        for i in 0..bboxes.len() - 1 {
            let obj0 = &bboxes[i];
            let obj1 = &bboxes[i + 1];

            let h0 = height(obj0);
            let h1 = height(obj1);
            let w0 = width(obj0);
            let w1 = width(obj1);

            // halign: obj0 and obj1 are horizontally aligned
            // Requires vertical overlap, sufficient overlap amount, and close enough horizontally
            let halign = is_voverlap(obj0, obj1)
                && h0.min(h1) * line_overlap < voverlap(obj0, obj1)
                && hdistance(obj0, obj1) < w0.max(w1) * char_margin;

            // valign: obj0 and obj1 are vertically aligned
            // Only checked if detect_vertical is true
            let valign = detect_vertical
                && is_hoverlap(obj0, obj1)
                && w0.min(w1) * line_overlap < hoverlap(obj0, obj1)
                && vdistance(obj0, obj1) < h0.max(h1) * char_margin;

            let decision = match (halign, valign) {
                (true, false) => ALIGN_HORIZONTAL,
                (false, true) => ALIGN_VERTICAL,
                (true, true) => ALIGN_HORIZONTAL, // Prefer horizontal when both
                (false, false) => ALIGN_NONE,
            };

            results.push(decision);
        }

        results
    }

    /// Compute grouping indices for text objects.
    ///
    /// This function implements the full grouping logic from group_objects(),
    /// returning indices that indicate which line each object belongs to and
    /// line boundary information.
    ///
    /// Args:
    ///     bboxes: List of (x0, y0, x1, y1) bounding boxes in sorted order
    ///     line_overlap: LAParams.line_overlap threshold
    ///     char_margin: LAParams.char_margin threshold
    ///     detect_vertical: LAParams.detect_vertical flag
    ///
    /// Returns:
    ///     Tuple of:
    ///     - line_starts: List of indices where new lines start
    ///     - line_types: List of line types (1=horizontal, 2=vertical) for each line
    #[staticmethod]
    pub fn compute_line_groups(
        bboxes: Vec<BBox>,
        line_overlap: f64,
        char_margin: f64,
        detect_vertical: bool,
    ) -> (Vec<usize>, Vec<u8>) {
        if bboxes.is_empty() {
            return (vec![], vec![]);
        }

        if bboxes.len() == 1 {
            return (vec![0], vec![ALIGN_HORIZONTAL]);
        }

        let mut line_starts: Vec<usize> = vec![0];
        let mut line_types: Vec<u8> = Vec::new();
        let mut current_line_type: Option<u8> = None;

        for i in 0..bboxes.len() - 1 {
            let obj0 = &bboxes[i];
            let obj1 = &bboxes[i + 1];

            let h0 = height(obj0);
            let h1 = height(obj1);
            let w0 = width(obj0);
            let w1 = width(obj1);

            let halign = is_voverlap(obj0, obj1)
                && h0.min(h1) * line_overlap < voverlap(obj0, obj1)
                && hdistance(obj0, obj1) < w0.max(w1) * char_margin;

            let valign = detect_vertical
                && is_hoverlap(obj0, obj1)
                && w0.min(w1) * line_overlap < hoverlap(obj0, obj1)
                && vdistance(obj0, obj1) < h0.max(h1) * char_margin;

            match current_line_type {
                Some(ALIGN_HORIZONTAL) if halign => {
                    // Continue horizontal line
                }
                Some(ALIGN_VERTICAL) if valign => {
                    // Continue vertical line
                }
                Some(line_type) => {
                    // End current line
                    line_types.push(line_type);

                    if valign && !halign {
                        // Start new vertical line at obj0
                        line_starts.push(i);
                        current_line_type = Some(ALIGN_VERTICAL);
                    } else if halign && !valign {
                        // Start new horizontal line at obj0
                        line_starts.push(i);
                        current_line_type = Some(ALIGN_HORIZONTAL);
                    } else {
                        // obj0 is a single-char line, obj1 starts fresh
                        line_starts.push(i);
                        line_types.push(ALIGN_HORIZONTAL);
                        line_starts.push(i + 1);
                        current_line_type = None;
                    }
                }
                None => {
                    // Starting fresh
                    if valign && !halign {
                        current_line_type = Some(ALIGN_VERTICAL);
                    } else if halign {
                        current_line_type = Some(ALIGN_HORIZONTAL);
                    } else {
                        // Single-char line
                        line_types.push(ALIGN_HORIZONTAL);
                        line_starts.push(i + 1);
                    }
                }
            }
        }

        // Close final line
        if let Some(line_type) = current_line_type {
            line_types.push(line_type);
        } else if line_types.len() < line_starts.len() {
            line_types.push(ALIGN_HORIZONTAL);
        }

        (line_starts, line_types)
    }

    /// Batch compute distances between all adjacent pairs.
    ///
    /// Returns (hdistances, vdistances, voverlaps, hoverlaps) for each adjacent pair.
    #[staticmethod]
    pub fn compute_distances(
        bboxes: Vec<BBox>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        if bboxes.len() < 2 {
            return (vec![], vec![], vec![], vec![]);
        }

        let n = bboxes.len() - 1;
        let mut hdists = Vec::with_capacity(n);
        let mut vdists = Vec::with_capacity(n);
        let mut voverlaps = Vec::with_capacity(n);
        let mut hoverlaps = Vec::with_capacity(n);

        for i in 0..n {
            let a = &bboxes[i];
            let b = &bboxes[i + 1];
            hdists.push(hdistance(a, b));
            vdists.push(vdistance(a, b));
            voverlaps.push(voverlap(a, b));
            hoverlaps.push(hoverlap(a, b));
        }

        (hdists, vdists, voverlaps, hoverlaps)
    }
}

impl Default for LayoutOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voverlap() {
        // Overlapping vertically
        let a = (0.0, 0.0, 10.0, 10.0);
        let b = (5.0, 5.0, 15.0, 15.0);
        assert!(is_voverlap(&a, &b));
        assert!((voverlap(&a, &b) - 5.0).abs() < 1e-10);

        // Not overlapping
        let c = (0.0, 20.0, 10.0, 30.0);
        assert!(!is_voverlap(&a, &c));
        assert!((voverlap(&a, &c) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hdistance() {
        // Overlapping - distance is 0
        let a = (0.0, 0.0, 10.0, 10.0);
        let b = (5.0, 0.0, 15.0, 10.0);
        assert!((hdistance(&a, &b) - 0.0).abs() < 1e-10);

        // Gap between
        let c = (20.0, 0.0, 30.0, 10.0);
        assert!((hdistance(&a, &c) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_alignments() {
        let bboxes = vec![
            (0.0, 0.0, 10.0, 10.0),
            (12.0, 0.0, 22.0, 10.0), // Horizontally aligned with first
            (100.0, 0.0, 110.0, 10.0), // Too far
        ];

        let result = LayoutOps::compute_alignments(bboxes, 0.5, 2.0, false);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ALIGN_HORIZONTAL); // First two are aligned
        assert_eq!(result[1], ALIGN_NONE); // Second and third are not
    }
}
