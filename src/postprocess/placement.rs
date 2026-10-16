//! **Is the co-embedding on the cell manifold?** The placement contract, measured.
//!
//! A co-embedded feature is a weighted mean of cells, so it should sit *inside* the cell
//! cloud, and *where* inside should not depend on how long the nearby cells are. Two
//! numbers per run check this, plus two that locate the failure:
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
//! A NaN feature row counts as beyond p99 and is left out of the nearest-cell statistics,
//! so a diverged co-embedding fails rather than reading as placed.
//!
//! A pure function of `(cells, features)`, so a consumer can check any co-embedding it reads,
//! not only the one this crate just wrote.

#[cfg(test)]
mod tests;

use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::utils::median;
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
        let (n, h) = cells.shape();
        let centre: Vec<f32> = (0..h).map(|j| cells.column(j).mean()).collect();
        let dist: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| distance(cells, i, &centre))
            .collect();
        let radius = dist.iter().sum::<f32>() / n.max(1) as f32;
        Self {
            centre,
            dist,
            radius,
        }
    }
}

/// `‖m[i, :] − centre‖`.
fn distance(m: &DMatrix<f32>, i: usize, centre: &[f32]) -> f32 {
    centre
        .iter()
        .enumerate()
        .map(|(j, &c)| (m[(i, j)] - c).powi(2))
        .sum::<f32>()
        .sqrt()
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
        self.beyond_p99 < MAX_BEYOND_P99
            && (self.nn_norm_pct - 50.0).abs() <= NN_NORM_PCT_BAND
            && !split
    }

    /// The contract [`CoembedPlacement::on_manifold`] checks, in words.
    #[must_use]
    pub fn contract() -> String {
        format!(
            "beyond_p99 < {MAX_BEYOND_P99}, nn_norm_pct within 50 ± {NN_NORM_PCT_BAND}, and not \
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
        )
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
    let (n, h) = cells.shape();
    let (d, h2) = feats.shape();
    anyhow::ensure!(
        h == h2 && h > 0,
        "coembed_placement: need equal, non-zero widths (cells {h}, features {h2})"
    );
    anyhow::ensure!(
        n > 0 && d > 0,
        "coembed_placement: need ≥1 cell and ≥1 feature (got {n}, {d})"
    );
    let cloud = CellCloud::of(cells);
    let radius = cloud.radius;
    anyhow::ensure!(
        radius.is_finite() && radius > 0.0,
        "coembed_placement: the cell cloud has no radius (R = {radius}); the cells are \
         identical, a single cell, or not finite"
    );

    let feat_dist: Vec<f32> = (0..d)
        .into_par_iter()
        .map(|i| distance(feats, i, &cloud.centre))
        .collect();
    let finite_dist: Vec<f32> = feat_dist
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    let mut sorted = cloud.dist.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    // Share of cells strictly closer to the centre than `x`, in percent.
    let pct = |x: f32| sorted.partition_point(|&c| c < x) as f32 / n as f32 * 100.0;

    let p99 = quantile_sorted(&sorted, 0.99);
    // A NaN row counts as beyond: it is not on the cloud.
    let beyond_p99 = feat_dist.iter().filter(|&&x| x.is_nan() || x > p99).count() as f32 / d as f32;
    let (dist_x_r, dist_pct) = if finite_dist.is_empty() {
        (f32::NAN, f32::NAN)
    } else {
        let pcts: Vec<f32> = finite_dist.iter().map(|&x| pct(x)).collect();
        (median(&finite_dist) / radius, median(&pcts))
    };

    // Nearest cells, on strided subsamples of both sides, finite feature rows only.
    // Row-major copies so the inner distance loop reads contiguous memory.
    let cell_idx = strided(n, MAX_NN_CELLS);
    let feat_idx: Vec<usize> = strided(d, MAX_NN_FEATS)
        .into_iter()
        .filter(|&g| feat_dist[g].is_finite())
        .collect();
    let sub_cells: Vec<f32> = cell_idx
        .iter()
        .flat_map(|&i| (0..h).map(move |j| cells[(i, j)]))
        .collect();
    let sub_cell_pct: Vec<f32> = cell_idx.iter().map(|&i| pct(cloud.dist[i])).collect();
    let k = NN_K.min(cell_idx.len());
    let mut per_feat: Vec<f32> = feat_idx
        .par_iter()
        .map(|&g| {
            let e: Vec<f32> = (0..h).map(|j| feats[(g, j)]).collect();
            let mut d2: Vec<(f32, usize)> = sub_cells
                .chunks_exact(h)
                .enumerate()
                .map(|(c, z)| (z.iter().zip(&e).map(|(a, b)| (a - b).powi(2)).sum(), c))
                .collect();
            d2.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            d2[..k].iter().map(|&(_, c)| sub_cell_pct[c]).sum::<f32>() / k as f32
        })
        .collect();
    per_feat.sort_unstable_by(f32::total_cmp);
    let (nn_norm_pct, nn_norm_pct_q25, nn_norm_pct_q75) = if per_feat.is_empty() {
        (f32::NAN, f32::NAN, f32::NAN)
    } else {
        (
            per_feat.iter().sum::<f32>() / per_feat.len() as f32,
            quantile_sorted(&per_feat, 0.25),
            quantile_sorted(&per_feat, 0.75),
        )
    };

    Ok(CoembedPlacement {
        radius,
        dist_x_r,
        dist_pct,
        beyond_p99,
        nn_norm_pct,
        nn_norm_pct_q25,
        nn_norm_pct_q75,
    })
}

/// `≤ max` evenly strided indices into `0..n`.
fn strided(n: usize, max: usize) -> Vec<usize> {
    if n <= max {
        return (0..n).collect();
    }
    (0..max).map(|i| i * n / max).collect()
}

/// Linear-interpolated quantile of an ascending, non-empty slice.
fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    let pos = q * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (pos - lo as f32) * (sorted[hi] - sorted[lo])
}
