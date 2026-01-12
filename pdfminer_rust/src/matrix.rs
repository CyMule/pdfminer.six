//! Batch matrix operations for PDF coordinate transformations.
//!
//! This module provides efficient batch processing of matrix transformations
//! to minimize FFI overhead when transforming many points or rectangles.

use pyo3::prelude::*;

/// A 2D affine transformation matrix represented as [a, b, c, d, e, f].
/// Transforms point (x, y) to (a*x + c*y + e, b*x + d*y + f).
type Matrix = [f64; 6];

/// Apply a matrix transformation to a single point.
#[inline]
fn transform_point(m: &Matrix, x: f64, y: f64) -> (f64, f64) {
    let [a, b, c, d, e, f] = *m;
    (a * x + c * y + e, b * x + d * y + f)
}

/// Apply a matrix transformation to a rectangle, returning the axis-aligned
/// bounding box of the transformed rectangle.
#[inline]
fn transform_rect(m: &Matrix, x0: f64, y0: f64, x1: f64, y1: f64) -> (f64, f64, f64, f64) {
    // Transform all four corners
    let (lx0, ly0) = transform_point(m, x0, y0); // left-bottom
    let (lx1, ly1) = transform_point(m, x1, y0); // right-bottom
    let (lx2, ly2) = transform_point(m, x1, y1); // right-top
    let (lx3, ly3) = transform_point(m, x0, y1); // left-top

    // Find bounding box
    let min_x = lx0.min(lx1).min(lx2).min(lx3);
    let min_y = ly0.min(ly1).min(ly2).min(ly3);
    let max_x = lx0.max(lx1).max(lx2).max(lx3);
    let max_y = ly0.max(ly1).max(ly2).max(ly3);

    (min_x, min_y, max_x, max_y)
}

/// Create a translation matrix that moves by (x, y).
#[inline]
fn translate_matrix(m: &Matrix, x: f64, y: f64) -> Matrix {
    let [a, b, c, d, e, f] = *m;
    [a, b, c, d, x * a + y * c + e, x * b + y * d + f]
}

/// Batch matrix operations for PDF rendering.
///
/// This class provides efficient batch processing of matrix transformations,
/// minimizing FFI overhead by processing many operations in a single call.
#[pyclass]
pub struct MatrixOps;

#[pymethods]
impl MatrixOps {
    #[new]
    pub fn new() -> Self {
        MatrixOps
    }

    /// Transform multiple points using the same matrix.
    ///
    /// Args:
    ///     matrix: [a, b, c, d, e, f] transformation matrix
    ///     points: List of (x, y) tuples
    ///
    /// Returns:
    ///     List of transformed (x, y) tuples
    #[staticmethod]
    pub fn transform_points(
        matrix: [f64; 6],
        points: Vec<(f64, f64)>,
    ) -> Vec<(f64, f64)> {
        points
            .iter()
            .map(|(x, y)| transform_point(&matrix, *x, *y))
            .collect()
    }

    /// Transform multiple rectangles using the same matrix.
    ///
    /// Args:
    ///     matrix: [a, b, c, d, e, f] transformation matrix
    ///     rects: List of (x0, y0, x1, y1) tuples
    ///
    /// Returns:
    ///     List of transformed (x0, y0, x1, y1) bounding boxes
    #[staticmethod]
    pub fn transform_rects(
        matrix: [f64; 6],
        rects: Vec<(f64, f64, f64, f64)>,
    ) -> Vec<(f64, f64, f64, f64)> {
        rects
            .iter()
            .map(|(x0, y0, x1, y1)| transform_rect(&matrix, *x0, *y0, *x1, *y1))
            .collect()
    }

    /// Batch process character rendering data.
    ///
    /// For each character, computes:
    /// 1. The translated matrix (base_matrix translated by x_offset)
    /// 2. The transformed bounding box
    ///
    /// This combines translate_matrix + apply_matrix_rect into one batch call.
    ///
    /// Args:
    ///     base_matrix: [a, b, c, d, e, f] base transformation matrix
    ///     char_data: List of (x_offset, y_offset, bbox) where bbox is (x0, y0, x1, y1)
    ///
    /// Returns:
    ///     List of (translated_matrix, transformed_bbox) tuples
    #[staticmethod]
    pub fn batch_transform_chars(
        base_matrix: [f64; 6],
        char_data: Vec<(f64, f64, f64, f64, f64, f64)>,
    ) -> Vec<([f64; 6], (f64, f64, f64, f64))> {
        char_data
            .iter()
            .map(|(x_off, y_off, bx0, by0, bx1, by1)| {
                let translated = translate_matrix(&base_matrix, *x_off, *y_off);
                let bbox = transform_rect(&translated, *bx0, *by0, *bx1, *by1);
                (translated, bbox)
            })
            .collect()
    }

    /// Batch compute character bounding boxes for horizontal text.
    ///
    /// Given a base matrix, starting position, and character advances,
    /// computes all translated matrices and transformed bounding boxes.
    ///
    /// Args:
    ///     base_matrix: [a, b, c, d, e, f] base transformation matrix
    ///     start_x: Starting x position
    ///     start_y: Starting y position (constant for horizontal text)
    ///     advances: List of (x_advance, local_bbox) for each character
    ///               where local_bbox is (x0, y0, x1, y1) in glyph space
    ///
    /// Returns:
    ///     List of (matrix, transformed_bbox, next_x) for each character
    #[staticmethod]
    pub fn batch_horizontal_chars(
        base_matrix: [f64; 6],
        start_x: f64,
        start_y: f64,
        advances: Vec<(f64, (f64, f64, f64, f64))>,
    ) -> Vec<([f64; 6], (f64, f64, f64, f64))> {
        let mut x = start_x;
        let y = start_y;

        advances
            .iter()
            .map(|(adv, (bx0, by0, bx1, by1))| {
                let translated = translate_matrix(&base_matrix, x, y);
                let bbox = transform_rect(&translated, *bx0, *by0, *bx1, *by1);
                x += adv;
                (translated, bbox)
            })
            .collect()
    }
}

impl Default for MatrixOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_point_identity() {
        let m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (x, y) = transform_point(&m, 10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_point_translation() {
        let m = [1.0, 0.0, 0.0, 1.0, 100.0, 200.0];
        let (x, y) = transform_point(&m, 10.0, 20.0);
        assert!((x - 110.0).abs() < 1e-10);
        assert!((y - 220.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_point_scale() {
        let m = [2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (x, y) = transform_point(&m, 10.0, 20.0);
        assert!((x - 20.0).abs() < 1e-10);
        assert!((y - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_rect_identity() {
        let m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (x0, y0, x1, y1) = transform_rect(&m, 0.0, 0.0, 10.0, 20.0);
        assert!((x0 - 0.0).abs() < 1e-10);
        assert!((y0 - 0.0).abs() < 1e-10);
        assert!((x1 - 10.0).abs() < 1e-10);
        assert!((y1 - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_translate_matrix() {
        let m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let t = translate_matrix(&m, 100.0, 200.0);
        assert!((t[4] - 100.0).abs() < 1e-10);
        assert!((t[5] - 200.0).abs() < 1e-10);
    }
}
