//! **Is the co-embedding on the cell manifold?** The placement contract, measured.
//!
//! A co-embedded feature is a weighted mean of cells, so it should sit *inside* the cell
//! cloud, and *where* inside should not depend on how long the nearby cells are.
//!
//! - `dist_x_r`: median distance of a feature from the cell centre, in units of the cloud's
//!   radius `R = mean_i ‖z_i − mean(z)‖` (the radius `hub_call` measures against).
//! - `dist_pct`: median over features of that distance as a percentile of the cells' own
//!   distance-from-centre distribution (100 = beyond every cell).
//! - `beyond_p99`: share of features farther out than 99% of cells. **Contract: < 0.15.**
//! - `nn_norm_pct` and its quartiles: per feature, the mean distance-from-centre percentile
//!   of its [`NN_K`] nearest cells — high when features sit beside the longest cells, low
//!   when they collapse to the hub.
//!
//! **What that percentile should be is a property of the cloud, not a constant.** A feature
//! is an average of cells, and an average sits inward of the cells it averages, by an amount
//! that depends on the cloud's shape: on an elongated cloud a healthy feature's nearest cells
//! sit well below the 50th percentile. So the yardstick is the cells' own **local averages**
//! — each sampled cell replaced by the mean of its [`NN_K`] nearest cells, the smallest
//! average a feature could be — run through the same statistic.
//!
//! **The two directions are not symmetric.** An average can be pulled inward of the cells
//! it averages but never pushed outward, and a feature averages a whole cone of cells, so it
//! sits further in than a local average does. Outward is therefore judged strictly and inward
//! only at the tail; [`PlacementViolation`] states each clause.
//!
//! A non-finite feature row fails the contract outright, counts as beyond p99, and is left
//! out of the nearest-cell statistics.
//!
//! A pure function of `(cells, features)`, so a consumer can check any co-embedding it reads,
//! not only the one this crate just wrote.

#[cfg(test)]
mod tests;

use legume_numeric::candle::candle_core::{DType, Tensor};
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::knn::metric::{l2_simd, l2_sq};
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
            .map(|row| l2_simd(row, &centre))
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
    /// Quartiles (25th, 50th, 75th) of the per-feature nearest-cell percentile.
    pub nn_norm_pct_quartiles: [f32; 3],
    /// The same statistic for the cells' own local averages: its 10th, 75th and 90th
    /// percentiles. The yardstick for `nn_norm_pct_quartiles`.
    pub cell_baseline: [f32; 3],
}

/// A clause of the placement contract that a co-embedding fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementViolation {
    /// Some feature rows are NaN or infinite: the fit diverged.
    NonFinite,
    /// `beyond_p99` is at or above its bound: features sit outside the cell cloud.
    BeyondCloud,
    /// The features' median is above the local averages' 75th percentile, or their 75th above
    /// the 90th: features sit beside longer cells than an average of cells can.
    Outward,
    /// The features' 25th percentile is below the local averages' 10th: a quarter of the
    /// features sit nearer the hub than the cells' own averages reach.
    Inward,
}

impl std::fmt::Display for PlacementViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "non-finite rows, the fit diverged"),
            Self::BeyondCloud => write!(
                f,
                "beyond_p99 ≥ {MAX_BEYOND_P99}, features sit outside the cell cloud"
            ),
            Self::Outward => write!(
                f,
                "features sit beside longer cells than the cells' own local averages (median > \
                 their q75 or q75 > their q90)"
            ),
            Self::Inward => write!(
                f,
                "features sit nearer the hub than the cells' own local averages (q25 < their q10)"
            ),
        }
    }
}

impl CoembedPlacement {
    /// The clauses of the contract this placement fails (NaN statistics fail them too).
    #[must_use]
    pub fn violations(&self) -> Vec<PlacementViolation> {
        let [q25, q50, q75] = self.nn_norm_pct_quartiles;
        let [b10, b75, b90] = self.cell_baseline;
        // Each clause is written as the passing condition, so a NaN statistic fails it.
        let inside = self.beyond_p99 < MAX_BEYOND_P99;
        let not_outward = q50 <= b75 && q75 <= b90;
        let not_inward = q25 >= b10;
        [
            (self.non_finite == 0, PlacementViolation::NonFinite),
            (inside, PlacementViolation::BeyondCloud),
            (not_outward, PlacementViolation::Outward),
            (not_inward, PlacementViolation::Inward),
        ]
        .into_iter()
        .filter_map(|(ok, v)| (!ok).then_some(v))
        .collect()
    }

    /// Finite, inside the cloud, and sitting among the cells the way the cells' own local
    /// averages do: no [`PlacementViolation`].
    #[must_use]
    pub fn on_manifold(&self) -> bool {
        self.violations().is_empty()
    }
}

impl std::fmt::Display for CoembedPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            radius,
            dist_x_r,
            dist_pct,
            beyond_p99,
            nn_norm_pct,
            ..
        } = self;
        let [q25, q50, q75] = self.nn_norm_pct_quartiles;
        let [b10, b75, b90] = self.cell_baseline;
        write!(
            f,
            "dist {dist_x_r:.2}×R (R={radius:.3}), dist_pct {dist_pct:.1}, beyond_p99 \
             {beyond_p99:.3}, nn_norm_pct {nn_norm_pct:.1} (quartiles {q25:.1} / {q50:.1} / \
             {q75:.1}; cells' local averages q10 / q75 / q90 {b10:.1} / {b75:.1} / {b90:.1})"
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
        Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
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
        .map(|row| l2_simd(row, &cloud.centre))
        .collect();

    let mut sorted = cloud.dist.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    // Percentile of `x` among the cells' distances, ties at the middle of their range.
    let pct = |x: f32| {
        let below = sorted.partition_point(|&c| c < x);
        let up_to = sorted.partition_point(|&c| c <= x);
        (below + up_to) as f32 / 2.0 / n as f32 * 100.0
    };

    let p99 = quantile_sorted(&sorted, 0.99);
    // A NaN row counts as beyond: it is not on the cloud.
    let beyond_p99 = feat_dist.iter().filter(|&&x| x.is_nan() || x > p99).count() as f32 / d as f32;
    let mut finite_dist: Vec<f32> = feat_dist
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    finite_dist.sort_unstable_by(f32::total_cmp);
    let median_dist = quantile_sorted(&finite_dist, 0.5);
    let dist_x_r = median_dist / radius;
    // `pct` is monotone, so the median feature's percentile is the percentile of the median.
    let dist_pct = if median_dist.is_nan() {
        f32::NAN
    } else {
        pct(median_dist)
    };

    // Nearest cells, on strided subsamples of both sides, finite feature rows only.
    let cell_idx = super::strided_indices(n, MAX_NN_CELLS);
    let sub_cells: Vec<&[f32]> = cell_idx
        .iter()
        .map(|&i| &cells[i * h..(i + 1) * h])
        .collect();
    let sub_cell_pct: Vec<f32> = cell_idx.iter().map(|&i| pct(cloud.dist[i])).collect();
    let k = NN_K.min(cell_idx.len());
    // The `k` sampled cells nearest `e`.
    let nearest = |e: &[f32]| -> Vec<usize> {
        let mut d2: Vec<(f32, usize)> = sub_cells
            .iter()
            .enumerate()
            .map(|(c, z)| (l2_sq(z, e), c))
            .collect();
        d2.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        d2[..k].iter().map(|&(_, c)| c).collect()
    };
    let nearest_pct =
        |e: &[f32]| nearest(e).iter().map(|&c| sub_cell_pct[c]).sum::<f32>() / k as f32;

    let mut per_feat: Vec<f32> = super::strided_indices(d, MAX_NN_FEATS)
        .into_par_iter()
        .filter(|&g| feat_dist[g].is_finite())
        .map(|g| nearest_pct(&feats[g * h..(g + 1) * h]))
        .collect();
    per_feat.sort_unstable_by(f32::total_cmp);

    // The yardstick: each sampled cell's local average (the mean of its `k` nearest cells),
    // through the same statistic.
    let mut baseline: Vec<f32> = super::strided_indices(sub_cells.len(), MAX_NN_FEATS)
        .into_par_iter()
        .map(|p| {
            let mut local = vec![0f32; h];
            for c in nearest(sub_cells[p]) {
                local
                    .iter_mut()
                    .zip(sub_cells[c])
                    .for_each(|(l, &x)| *l += x);
            }
            local.iter_mut().for_each(|l| *l /= k as f32);
            nearest_pct(&local)
        })
        .collect();
    baseline.sort_unstable_by(f32::total_cmp);

    Ok(CoembedPlacement {
        radius,
        dist_x_r,
        dist_pct,
        beyond_p99,
        // A row with a NaN or infinite coordinate has a non-finite distance, so only those
        // rows need their coordinates checked.
        non_finite: (0..d)
            .filter(|&g| !feat_dist[g].is_finite())
            .filter(|&g| feats[g * h..(g + 1) * h].iter().any(|x| !x.is_finite()))
            .count(),
        // An empty `per_feat` gives 0/0 = NaN, which fails every gate.
        nn_norm_pct: per_feat.iter().sum::<f32>() / per_feat.len() as f32,
        nn_norm_pct_quartiles: [0.25, 0.5, 0.75].map(|q| quantile_sorted(&per_feat, q)),
        cell_baseline: [0.1, 0.75, 0.9].map(|q| quantile_sorted(&baseline, q)),
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
