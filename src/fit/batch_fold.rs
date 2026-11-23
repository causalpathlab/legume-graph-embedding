//! Per-batch gene fold for the phase-2 projection.
//!
//! The collapse estimates, per gene and batch, the fold `δ_gb` by which that
//! batch's observed counts depart from the batch-free pseudobulk rate `μ`
//! (`E[y_gb] = δ_gb · μ_g`). Phase 1 trains the dictionary on those batch-free
//! pseudobulks, so phase 2 puts each cell in the same frame by **dividing its
//! counts by its batch's fold** before the Poisson-MAP solve.
//!
//! Why divide rather than offset the rate: a low-rank log-linear dictionary
//! cannot reproduce a whole gene profile exactly, and at the optimum the Poisson
//! fit matches the count-weighted centroid in dictionary space. With `log δ` as a
//! rate offset that centroid is weighted by each batch's own `δ`, so two batches
//! solve two different weighted problems and land apart even when `δ` is right.
//! Dividing the counts first hands every batch the same estimating equation, so
//! the solution is batch-invariant regardless of misfit — measured at pseudobulk
//! resolution, where sparsity plays no part, the divide agreed across studies
//! markedly better than the offset.
//!
//! This module turns the collapse's `δ` (columns in the cells' batch-id order)
//! into that fold, floored so a gene a batch never measured divides by a small
//! positive number rather than zero.

use super::projection::CellBatchFold;
use nalgebra::DMatrix;

/// Floor on a posterior-mean `δ`, so a gene a batch never measured stays a finite
/// positive divisor.
pub const DELTA_FLOOR: f32 = 1e-6;

/// `δ_gb` on the unified feature axis, one row per **unified** batch id.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchGeneFold {
    /// `[n_batches × n_features]` row-major, linear scale, floored.
    pub delta: Vec<f32>,
    pub n_features: usize,
    /// Unified batch names, in row order.
    pub batch_names: Vec<Box<str>>,
}

impl BatchGeneFold {
    pub fn n_batches(&self) -> usize {
        self.batch_names.len()
    }

    /// The fold row for unified batch `b`.
    pub fn row(&self, b: usize) -> &[f32] {
        &self.delta[b * self.n_features..(b + 1) * self.n_features]
    }

    /// The per-cell view phase 2 divides by: this table plus each cell's batch.
    pub(crate) fn cell_fold<'a>(&'a self, cell_to_batch: &'a [u32]) -> CellBatchFold<'a> {
        CellBatchFold {
            fold: self,
            cell_to_batch,
        }
    }
}

/// Build the per-batch fold table from `delta` (`[n_features × n_batches]`,
/// columns in `batch_names` order), or `None` when there is a single batch
/// (nothing to correct against).
pub(crate) fn batch_gene_fold(
    delta: &DMatrix<f32>,
    batch_names: &[Box<str>],
) -> anyhow::Result<Option<BatchGeneFold>> {
    let n_batches = batch_names.len();
    if n_batches < 2 {
        return Ok(None);
    }
    anyhow::ensure!(
        delta.ncols() == n_batches,
        "δ has {} columns for {n_batches} batches",
        delta.ncols()
    );
    let n_features = delta.nrows();
    let fold = (0..n_batches)
        .flat_map(|b| (0..n_features).map(move |f| delta[(f, b)].max(DELTA_FLOOR)))
        .collect();
    Ok(Some(BatchGeneFold {
        delta: fold,
        n_features,
        batch_names: batch_names.to_vec(),
    }))
}

#[cfg(test)]
mod tests;
