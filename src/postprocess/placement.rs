//! **Is the co-embedding on the cell manifold?** The placement contract, measured.
//!
//! A co-embedded feature is a weighted mean of cells, so it should sit *inside* the cell
//! cloud, and *where* inside should not depend on how long the nearby cells are. Two
//! numbers locate the features, and three make up the contract:
//!
//! - `dist_x_r`: median distance of a feature from the cell centre, in units of the cloud's
//!   radius `R = mean_i ‖z_i − mean(z)‖` (the radius `hub_call` measures against).
//! - `dist_pct`: median over features of that distance as a percentile of the cells' own
//!   distance-from-centre distribution (100 = beyond every cell).
//! - `beyond_p99`: share of features farther out than 99% of cells. **Contract: < 0.15.**
//! - `nn_norm_pct`: mean over features of the mean distance-from-centre percentile of the
//!   feature's [`NN_K`] nearest cells. ≈ 50 when placement does not track cell norm.
//!   **Contract: within 50 ± 10.**
//! - `nn_norm_pct_q25` / `nn_norm_pct_q75`: the quartiles of that per-feature percentile.
//!   Single genes spread widely on healthy runs (pancreas: q25 31, q75 66; 72% of genes
//!   outside 50 ± 10), so the band cannot be applied gene by gene. But a mean of 50 can hide
//!   a split — half the genes at the hub, half at the edge — and that shows as quartiles in
//!   both tails. **Contract: not q25 < 10 with q75 > 90.** No healthy run measured comes
//!   close (q25 ≥ 17.7 across pancreas, erythroid and dentate gyrus).
//!
//! A non-finite feature row fails the contract outright, counts as beyond p99, and is left
//! out of the nearest-cell statistics.
//!
//! A pure function of `(cells, features)`, so a consumer can check any co-embedding it reads,
//! not only the one this crate just wrote.

#[cfg(test)]
mod tests;

use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::knn::metric::l2_sq;
use rayon::prelude::*;

/// Nearest cells per feature for `nn_norm_pct`.
const NN_K: usize = 30;
/// Cells the nearest-neighbour search runs over (evenly strided). Percentiles are still
/// taken against every cell.
const MAX_NN_CELLS: usize = 60_000;
/// Features `nn_norm_pct` averages over (evenly strided).
const MAX_NN_FEATS: usize = 3_000;
/// Upper bound on `beyond_p99` for an on-manifold co-embedding.
const MAX_BEYOND_P99: f32 = 0.15;
/// Allowed distance of `nn_norm_pct` from 50.
const NN_NORM_PCT_BAND: f32 = 10.0;
/// A split placement has its lower quartile below this and its upper quartile above
/// `100 −` this.
const SPLIT_TAIL: f32 = 10.0;

/// The centre of a cell cloud, each cell's distance from it, and the cloud's radius
/// `R = mean_i ‖z_i − centre‖` — the one definition `hub_call` and [`coembed_placement`]
/// share. `radius` is NaN when any cell is.
pub(crate) struct CellCloud {
    pub centre: Vec<f32>,
    pub dist: Vec<f32>,
    pub radius: f32,
}

impl CellCloud {
    pub fn of(cells: &DMatrix<f32>) -> Self {
        Self::of_rows(cells.transpose().as_slice(), cells.ncols())
    }

    /// From row-major `[n × h]` cells (`h > 0`).
    fn of_rows(cells: &[f32], h: usize) -> Self {
        let n = cells.len() / h;
        let mut centre = vec![0f32; h];
        for row in cells.chunks_exact(h) {
            centre.iter_mut().zip(row).for_each(|(c, &x)| *c += x);
        }
        centre.iter_mut().for_each(|c| *c /= n.max(1) as f32);
        let dist: Vec<f32> = cells
            .par_chunks_exact(h)
            .map(|row| l2_sq(row, &centre).sqrt())
            .collect();
        let radius = dist.iter().sum::<f32>() / n.max(1) as f32;
        Self {
            centre,
            dist,
            radius,
        }
    }
}

/// Where a co-embedding's features sit relative to its cells. See the module doc.
#[derive(Debug, Clone, Copy)]
pub struct CoembedPlacement {
    /// The cell cloud's radius `R = mean_i ‖z_i − mean(z)‖`.
    pub radius: f32,
    /// Median feature distance from the cell centre, in units of `R`.
    pub dist_x_r: f32,
    /// Median feature distance as a percentile of the cells' distances (0–100).
    pub dist_pct: f32,
    /// Share of features farther from the centre than 99% of cells (NaN rows included).
    pub beyond_p99: f32,
    /// Feature rows with a non-finite coordinate (a diverged fit).
    pub non_finite: usize,
    /// Mean distance-from-centre percentile of each feature's nearest cells (0–100).
    pub nn_norm_pct: f32,
    /// Lower quartile of the per-feature nearest-cell percentile.
    pub nn_norm_pct_q25: f32,
    /// Upper quartile of the per-feature nearest-cell percentile.
    pub nn_norm_pct_q75: f32,
}

impl CoembedPlacement {
    /// Inside the cloud, not tracking cell norm, and not split between hub and edge. See
    /// [`CoembedPlacement::contract`].
    #[must_use]
    pub fn on_manifold(&self) -> bool {
        let split = self.nn_norm_pct_q25 < SPLIT_TAIL && self.nn_norm_pct_q75 > 100.0 - SPLIT_TAIL;
        self.non_finite == 0
            && self.beyond_p99 < MAX_BEYOND_P99
            && (self.nn_norm_pct - 50.0).abs() <= NN_NORM_PCT_BAND
            && !split
    }

    /// The contract [`CoembedPlacement::on_manifold`] checks, in words.
    #[must_use]
    pub fn contract() -> String {
        format!(
            "no non-finite rows, beyond_p99 < {MAX_BEYOND_P99}, nn_norm_pct within 50 ± {NN_NORM_PCT_BAND}, and not \
             nn quartiles {SPLIT_TAIL} / {} in both tails",
            100.0 - SPLIT_TAIL
        )
    }
}

impl std::fmt::Display for CoembedPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dist {:.2}×R (R={:.3}), dist_pct {:.1}, beyond_p99 {:.3}, nn_norm_pct {:.1} \
             (quartiles {:.1}–{:.1})",
            self.dist_x_r,
            self.radius,
            self.dist_pct,
            self.beyond_p99,
            self.nn_norm_pct,
            self.nn_norm_pct_q25,
            self.nn_norm_pct_q75
        )?;
        if self.non_finite > 0 {
            write!(f, ", {} non-finite rows", self.non_finite)?;
        }
        Ok(())
    }
}

/// Measure where the features `feats` (`[D, H]`) sit relative to the cells `cells` (`[N, H]`).
///
/// Errors on mismatched or zero widths, no cells or features, and a cell cloud with no
/// radius (identical cells, a single cell, or non-finite cells).
pub fn coembed_placement(
    cells: &DMatrix<f32>,
    feats: &DMatrix<f32>,
) -> anyhow::Result<CoembedPlacement> {
    anyhow::ensure!(
        cells.ncols() == feats.ncols(),
        "coembed_placement: H mismatch (cells {}, features {})",
        cells.ncols(),
        feats.ncols()
    );
    placement_of_rows(
        cells.transpose().as_slice(),
        feats.transpose().as_slice(),
        cells.ncols(),
    )
}

/// [`coembed_placement`] on `[N, H]` / `[D, H]` tensors, read row-major as they are.
pub(crate) fn coembed_placement_of_tensors(
    cells: &Tensor,
    feats: &Tensor,
) -> anyhow::Result<CoembedPlacement> {
    let (_, h) = cells.dims2()?;
    let (_, h2) = feats.dims2()?;
    anyhow::ensure!(
        h == h2,
        "coembed_placement: H mismatch (cells {h}, features {h2})"
    );
    let rows = |t: &Tensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
    };
    placement_of_rows(&rows(cells)?, &rows(feats)?, h)
}

/// The measurement, on row-major `[n × h]` cells and `[d × h]` features.
fn placement_of_rows(cells: &[f32], feats: &[f32], h: usize) -> anyhow::Result<CoembedPlacement> {
    anyhow::ensure!(h > 0, "coembed_placement: zero-width rows");
    let (n, d) = (cells.len() / h, feats.len() / h);
    anyhow::ensure!(
        n > 0 && d > 0,
        "coembed_placement: need ≥1 cell and ≥1 feature (got {n}, {d})"
    );
    let cloud = CellCloud::of_rows(cells, h);
    let radius = cloud.radius;
    anyhow::ensure!(
        radius.is_finite() && radius > 0.0,
        "coembed_placement: the cell cloud has no radius (R = {radius}); the cells are \
         identical, a single cell, or not finite"
    );

    let feat_dist: Vec<f32> = feats
        .par_chunks_exact(h)
        .map(|row| l2_sq(row, &cloud.centre).sqrt())
        .collect();

    let mut sorted = cloud.dist.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    // Share of cells strictly closer to the centre than `x`, in percent.
    let pct = |x: f32| sorted.partition_point(|&c| c < x) as f32 / n as f32 * 100.0;

    let p99 = quantile_sorted(&sorted, 0.99);
    // A NaN row counts as beyond: it is not on the cloud.
    let beyond_p99 = feat_dist.iter().filter(|&&x| x.is_nan() || x > p99).count() as f32 / d as f32;
    let mut finite_dist: Vec<f32> = feat_dist
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    finite_dist.sort_unstable_by(f32::total_cmp);
    // `pct` is monotone, so the percentiles of sorted distances come out sorted.
    let finite_pct: Vec<f32> = finite_dist.iter().map(|&x| pct(x)).collect();
    let dist_x_r = quantile_sorted(&finite_dist, 0.5) / radius;
    let dist_pct = quantile_sorted(&finite_pct, 0.5);

    // Nearest cells, on strided subsamples of both sides, finite feature rows only.
    let cell_idx = super::strided_indices(n, MAX_NN_CELLS);
    let sub_cells: Vec<&[f32]> = cell_idx
        .iter()
        .map(|&i| &cells[i * h..(i + 1) * h])
        .collect();
    let sub_cell_pct: Vec<f32> = cell_idx.iter().map(|&i| pct(cloud.dist[i])).collect();
    let k = NN_K.min(cell_idx.len());
    let mut per_feat: Vec<f32> = super::strided_indices(d, MAX_NN_FEATS)
        .into_par_iter()
        .filter(|&g| feat_dist[g].is_finite())
        .map(|g| {
            let e = &feats[g * h..(g + 1) * h];
            let mut d2: Vec<(f32, usize)> = sub_cells
                .iter()
                .enumerate()
                .map(|(c, z)| (l2_sq(z, e), c))
                .collect();
            d2.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            d2[..k].iter().map(|&(_, c)| sub_cell_pct[c]).sum::<f32>() / k as f32
        })
        .collect();
    per_feat.sort_unstable_by(f32::total_cmp);

    Ok(CoembedPlacement {
        radius,
        dist_x_r,
        dist_pct,
        beyond_p99,
        non_finite: feat_dist.iter().filter(|x| !x.is_finite()).count(),
        // An empty `per_feat` gives 0/0 = NaN, which fails every gate.
        nn_norm_pct: per_feat.iter().sum::<f32>() / per_feat.len() as f32,
        nn_norm_pct_q25: quantile_sorted(&per_feat, 0.25),
        nn_norm_pct_q75: quantile_sorted(&per_feat, 0.75),
    })
}

/// Linear-interpolated quantile of an ascending slice; NaN when it is empty.
fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f32::NAN;
    };
    let pos = q * last as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(last);
    sorted[lo] + (pos - lo as f32) * (sorted[hi] - sorted[lo])
}
