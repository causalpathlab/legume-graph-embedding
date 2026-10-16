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
//!
//! A pure function of `(cells, features)`, so a consumer can check any co-embedding it reads,
//! not only the one this crate just wrote.

use legume_numeric::matrix::dmatrix_io::DMatrix;
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

/// Where a co-embedding's features sit relative to its cells. See the module doc.
#[derive(Debug, Clone, Copy)]
pub struct CoembedPlacement {
    /// The cell cloud's radius `R = mean_i ‖z_i − mean(z)‖`.
    pub radius: f32,
    /// Median feature distance from the cell centre, in units of `R`.
    pub dist_x_r: f32,
    /// Median feature distance as a percentile of the cells' distances (0–100).
    pub dist_pct: f32,
    /// Share of features farther from the centre than 99% of cells.
    pub beyond_p99: f32,
    /// Mean distance-from-centre percentile of each feature's nearest cells (0–100).
    pub nn_norm_pct: f32,
}

impl CoembedPlacement {
    /// Inside the cloud and not tracking cell norm: `beyond_p99 < 0.15` and
    /// `nn_norm_pct` within `50 ± 10`.
    #[must_use]
    pub fn on_manifold(&self) -> bool {
        self.beyond_p99 < MAX_BEYOND_P99 && (self.nn_norm_pct - 50.0).abs() <= NN_NORM_PCT_BAND
    }
}

impl std::fmt::Display for CoembedPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dist {:.2}×R (R={:.3}), dist_pct {:.1}, beyond_p99 {:.3}, nn_norm_pct {:.1}",
            self.dist_x_r, self.radius, self.dist_pct, self.beyond_p99, self.nn_norm_pct
        )
    }
}

/// Measure where the features `feats` (`[D, H]`) sit relative to the cells `cells` (`[N, H]`).
pub fn coembed_placement(
    cells: &DMatrix<f32>,
    feats: &DMatrix<f32>,
) -> anyhow::Result<CoembedPlacement> {
    let (n, h) = cells.shape();
    let (d, h2) = feats.shape();
    anyhow::ensure!(
        h == h2,
        "coembed_placement: H mismatch (cells {h} vs features {h2})"
    );
    anyhow::ensure!(
        n > 0 && d > 0,
        "coembed_placement: need ≥1 cell and ≥1 feature (got {n}, {d})"
    );

    let centre: Vec<f32> = (0..h).map(|j| cells.column(j).mean()).collect();
    let dist_to_centre = |m: &DMatrix<f32>, i: usize| -> f32 {
        (0..h)
            .map(|j| (m[(i, j)] - centre[j]).powi(2))
            .sum::<f32>()
            .sqrt()
    };
    let cell_dist: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| dist_to_centre(cells, i))
        .collect();
    let feat_dist: Vec<f32> = (0..d)
        .into_par_iter()
        .map(|i| dist_to_centre(feats, i))
        .collect();
    let radius = cell_dist.iter().sum::<f32>() / n as f32;

    let mut sorted = cell_dist.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    // Share of cells strictly closer to the centre than `x`, in percent.
    let pct = |x: f32| sorted.partition_point(|&c| c < x) as f32 / n as f32 * 100.0;

    let p99 = quantile_sorted(&sorted, 0.99);
    let beyond_p99 = feat_dist.iter().filter(|&&x| x > p99).count() as f32 / d as f32;
    let dist_x_r = median(feat_dist.clone()) / radius;
    let dist_pct = median(feat_dist.iter().map(|&x| pct(x)).collect());

    // Nearest cells, on strided subsamples of both sides. Row-major copies so the inner
    // distance loop reads contiguous memory.
    let cell_idx = strided(n, MAX_NN_CELLS);
    let feat_idx = strided(d, MAX_NN_FEATS);
    let row_major = |m: &DMatrix<f32>, idx: &[usize]| -> Vec<f32> {
        idx.iter()
            .flat_map(|&i| (0..h).map(move |j| m[(i, j)]))
            .collect()
    };
    let sub_cells = row_major(cells, &cell_idx);
    let sub_cell_pct: Vec<f32> = cell_idx.iter().map(|&i| pct(cell_dist[i])).collect();
    let k = NN_K.min(cell_idx.len());
    let nn_norm_pct = feat_idx
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
        .sum::<f32>()
        / feat_idx.len() as f32;

    Ok(CoembedPlacement {
        radius,
        dist_x_r,
        dist_pct,
        beyond_p99,
        nn_norm_pct,
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

/// Median of a non-empty vector (mean of the middle two for an even length).
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_unstable_by(f32::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}
