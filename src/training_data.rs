//! The cells and pseudobulks this crate trains on, as interfaces the caller
//! implements over its own storage: a fit reads counts and pseudobulks through
//! them and depends on no storage crate. Apps adapt data-beans' types to them.

use nalgebra::DMatrix;
use std::borrow::Cow;
use std::ops::Range;

/// Counts of `n_cells()` cells over `n_features()` features, streamed in
/// slabs of consecutive cells.
pub trait CellCountStream: Sync {
    fn n_cells(&self) -> usize;
    fn n_features(&self) -> usize;
    /// Every nonzero of the whole matrix, when known: the caller sizes its
    /// slabs from it.
    fn nnz_hint(&self) -> Option<usize>;
    /// Call `f(feature, cell - cells.start, count)` for each stored count of
    /// the cells `cells`, feature on this stream's axis (`< n_features()`).
    /// A feature may repeat within a cell (each call adds to its count); the
    /// order within a cell is the storage's own.
    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()>;
}

/// Pseudobulk levels, **coarsest → finest**, on one feature axis, with each
/// cell's pseudobulk per level. Matrices are produced on request, so a reader
/// can hold one level at a time.
pub trait PseudobulkLevels {
    /// At least one.
    fn n_levels(&self) -> usize;
    /// Level `level`'s pseudobulk rates `[n_features × n_pb]`.
    fn rates(&self, level: usize) -> Cow<'_, DMatrix<f32>>;
    /// Each cell's pseudobulk at level `level`.
    fn cell_to_pb(&self, level: usize) -> &[usize];
    /// The finest level's observed counts `[n_features × n_pb]` and each
    /// pseudobulk's size.
    fn finest_counts(&self) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)>;
    /// The finest level's per-batch gene fold δ `[n_features × n_batches]`,
    /// columns in the cells' batch-id order; `None` when none was fit.
    fn finest_delta(&self) -> Option<Cow<'_, DMatrix<f32>>>;
}

/// For tests and fixtures: a small in-memory matrix streams its own columns.
#[cfg(test)]
impl CellCountStream for nalgebra_sparse::CscMatrix<f32> {
    fn n_cells(&self) -> usize {
        self.ncols()
    }

    fn n_features(&self) -> usize {
        self.nrows()
    }

    fn nnz_hint(&self) -> Option<usize> {
        Some(self.nnz())
    }

    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cells.end <= self.ncols(),
            "cells {cells:?} past the matrix's {} columns",
            self.ncols()
        );
        for (local, c) in cells.enumerate() {
            let col = self.col(c);
            for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                f(r as u32, local as u32, v);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;
    use nalgebra_sparse::CscMatrix;

    fn three_by_four() -> CscMatrix<f32> {
        let mut coo = CooMatrix::new(3, 4);
        coo.push(0, 0, 1.0);
        coo.push(2, 0, 2.0);
        coo.push(1, 2, 3.0);
        coo.push(2, 3, 4.0);
        CscMatrix::from(&coo)
    }

    /// Cells are reported relative to the slab's first cell, with every
    /// stored count, and nothing outside the slab.
    #[test]
    fn a_slab_reports_its_own_counts_with_local_cells() {
        let m = three_by_four();
        let mut seen = Vec::new();
        m.for_each_count(1..4, &mut |f, c, v| seen.push((f, c, v)))
            .expect("in range");
        assert_eq!(seen, vec![(1, 1, 3.0), (2, 2, 4.0)]);
        assert_eq!((m.n_cells(), m.n_features(), m.nnz_hint()), (4, 3, Some(4)));
    }

    #[test]
    fn a_slab_past_the_last_cell_is_refused() {
        let m = three_by_four();
        assert!(m.for_each_count(2..5, &mut |_, _, _| {}).is_err());
    }
}
