//! **Is the co-embedding on the cell manifold?** The placement contract, measured.
//!
//! A co-embedded feature is a weighted mean of cells, so it should sit *inside* the cell
//! cloud, and *where* inside should not depend on how long the nearby cells are.
//!
//! # The contract: unambiguous failures only
//!
//! [`CoembedPlacement::on_manifold`] fails on three things, each a failure no healthy
//! co-embedding produces ([`PlacementViolation`]):
//!
//! - **non-finite rows** — the fit diverged;
//! - **`beyond_p99 ≥ 0.15`** — features outside the cell cloud. A weighted mean of cells
//!   cannot leave their hull, and healthy placement puts about 1% of features past the cells'
//!   99th percentile; 15% is the gross failure the raw-cell scores produced (0.29–0.81 on the
//!   shipped runs), not a tuned bound;
//! - **hub collapse** — more than half the features at the hub as `hub_call` defines it
//!   (`CellCloud::is_at_hub`), where a flat softmax (SIMBA's fixed T run hot, or
//!   an untrained gene space) parks them. `hub_call` zeroes such rows one by one; a majority of
//!   them means the co-embedding as a whole located nothing.
//!
//! # Reported, not gated: where among the cells
//!
//! [`NearestCells`] reports, per feature, the mean distance-from-centre percentile of its
//! `NN_K` nearest cells — high beside the longest cells, low near the hub — as quartiles,
//! next to the same statistic for the cells' own `NN_K`-nearest-cell averages. That reference is
//! at a **different scale** from the co-embedding's softmax cones (20–100 cells along a
//! direction), so it is a yardstick to read, not a bound to test: a fixed `50 ± 10` and a
//! gate built on this reference both misfired on healthy clouds (elongated, topic-mixture)
//! while missing features parked beside long cells. A split — half the features at the hub,
//! half at the edge — shows as quartiles in both tails.
//!
//! **What this gives up.** A co-embedding whose features track cell norm *without* leaving the
//! cloud is not flagged: on several dentate gyrus runs the features sit beside the cells'
//! longest decile (nearest-cell percentile ~90) with `beyond_p99` near 0.02. That is visible in
//! the reported quartiles; a baseline built at the co-embedding's own averaging scale is the
//! planned way to gate it.
//!
//! A feature row with a non-finite distance (a NaN or infinite coordinate, or one so large its
//! distance overflows) counts as beyond p99 and is left out of every other statistic. The
//! share-based clauses are judged only over at least `NN_K` features.
//!
//! A pure function of `(cells, features)`, so a consumer can check any co-embedding it reads,
//! not only the one this crate just wrote.

#[cfg(test)]
mod tests;

use legume_numeric::candle::candle_core::{DType, Tensor};
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::knn::metric::{l2_simd, l2_sq};
use rayon::prelude::*;

/// Nearest cells per feature for the nearest-cell statistics.
const NN_K: usize = 30;
/// Cells the nearest-neighbour search runs over (evenly strided). Percentiles are still
/// taken against every cell.
const MAX_NN_CELLS: usize = 60_000;
/// Features the nearest-cell statistics run over (evenly strided).
const MAX_NN_FEATS: usize = 3_000;
/// Cells whose local averages form the report-only reference (evenly strided). Its quartiles
/// are read, not tested, so a smaller sample than the features' does: about ±2 percentile
/// points, at a sixth of the cost.
const MAX_NN_REF: usize = 500;
/// Upper bound on `beyond_p99` for an on-manifold co-embedding.
const MAX_BEYOND_P99: f32 = 0.15;
/// Share of rows at the hub above which the co-embedding as a whole located nothing — the
/// majority `hub_call` warns at, and the `HubCollapse` bound here.
pub(crate) const HUB_COLLAPSE_SHARE: f32 = 0.5;

/// How close to the hub a gene must be, as a fraction of the cell cloud's own radius, before it is
/// judged to carry no localizing information at all.
///
/// Deliberately small. The populations this separates are three orders of magnitude apart on real
/// data, so there is no need to cut anywhere near the live genes — and a *false positive* in
/// `hub_call` silently deletes a real marker, which is the one thing it must not do.
const HUB_RADIUS_FRAC: f32 = 0.05;

/// The centre of a cell cloud, each cell's distance from it, and the cloud's radius
/// `R = mean_i ‖z_i − centre‖` — the one definition `hub_call` and [`coembed_placement`]
/// share. `radius` is NaN when any cell is.
pub(crate) struct CellCloud {
    pub centre: Vec<f32>,
    pub dist: Vec<f32>,
    pub radius: f32,
}

impl CellCloud {
    /// Whether a row at `dist` from the centre sits at the hub: within `HUB_RADIUS_FRAC · R`,
    /// too close to every cell to localize any of them. The one rule `hub_call` zeroes rows by
    /// and `HubCollapse` counts them by.
    pub fn is_at_hub(&self, dist: f32) -> bool {
        dist < HUB_RADIUS_FRAC * self.radius
    }

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
    /// Median over features of their distance's percentile among the cells' (0–100).
    pub dist_pct: f32,
    /// Share of features farther from the centre than 99% of cells (NaN rows included).
    pub beyond_p99: f32,
    /// Feature rows with a non-finite coordinate (a diverged fit).
    pub non_finite: usize,
    /// Share of finite features at the hub (`CellCloud::is_at_hub`).
    pub hub_share: f32,
    /// Feature rows measured. Below `NN_K` the shares are not judged: a share of a handful of
    /// features is not evidence.
    pub n_feats: usize,
    /// The nearest-cell statistics, reported but not gated; `None` at `NN_K` cells or fewer,
    /// or fewer than `NN_K` finite features, where they mean nothing.
    pub nearest: Option<NearestCells>,
}

/// Per feature, the mean distance-from-centre percentile (0–100) of its `NN_K` nearest
/// cells: mean and quartiles; and the quartiles of the same statistic for the cells' own
/// local averages. See the module doc.
#[derive(Debug, Clone, Copy)]
pub struct NearestCells {
    pub mean: f32,
    pub q25: f32,
    pub median: f32,
    pub q75: f32,
    pub cells_q25: f32,
    pub cells_median: f32,
    pub cells_q75: f32,
}

/// An unambiguous placement failure. See the module doc for why each one is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementViolation {
    /// Some feature rows are NaN or infinite: the fit diverged.
    NonFinite,
    /// `beyond_p99` is at or above its bound: features sit outside the cell cloud.
    BeyondCloud,
    /// More than half the features sit at the hub: the co-embedding locates almost nothing
    /// (`hub_call` warns at the same majority).
    HubCollapse,
}

impl std::fmt::Display for PlacementViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "non-finite rows, the fit diverged"),
            Self::BeyondCloud => write!(
                f,
                "beyond_p99 ≥ {MAX_BEYOND_P99}, features sit outside the cell cloud"
            ),
            Self::HubCollapse => write!(
                f,
                "more than {HUB_COLLAPSE_SHARE} of the features within {HUB_RADIUS_FRAC}·R of \
                 the centre: it locates almost nothing (an untrained gene space, or a flat \
                 softmax)"
            ),
        }
    }
}

impl CoembedPlacement {
    /// The unambiguous failures this placement shows. The share-based clauses are judged only
    /// over at least `NN_K` features. (`hub_share` is NaN when no row has a finite distance —
    /// all non-finite, which `NonFinite` reports, or all overflowing, which counts as beyond.)
    #[must_use]
    pub fn violations(&self) -> Vec<PlacementViolation> {
        let judged = self.n_feats >= NN_K;
        [
            (self.non_finite > 0, PlacementViolation::NonFinite),
            (
                judged && self.beyond_p99 >= MAX_BEYOND_P99,
                PlacementViolation::BeyondCloud,
            ),
            (
                judged && self.hub_share > HUB_COLLAPSE_SHARE,
                PlacementViolation::HubCollapse,
            ),
        ]
        .into_iter()
        .filter_map(|(failed, v)| failed.then_some(v))
        .collect()
    }

    /// No [`PlacementViolation`].
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
            hub_share,
            non_finite,
            nearest,
            ..
        } = self;
        write!(
            f,
            "dist {dist_x_r:.2}×R (R={radius:.3}), dist_pct {dist_pct:.1}, beyond_p99 \
             {beyond_p99:.3}, hub_share {hub_share:.3}"
        )?;
        match nearest {
            Some(nn) => write!(
                f,
                ", nearest-cell percentile {:.1} (quartiles {:.1} / {:.1} / {:.1}; cells' \
                 {NN_K}-NN averages {:.1} / {:.1} / {:.1})",
                nn.mean, nn.q25, nn.median, nn.q75, nn.cells_q25, nn.cells_median, nn.cells_q75
            )?,
            None => write!(
                f,
                ", nearest-cell percentile not measured (at most {NN_K} cells, or fewer than \
                 {NN_K} finite features)"
            )?,
        }
        if *non_finite > 0 {
            write!(f, ", {non_finite} non-finite rows")?;
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
    let dist_x_r = quantile_sorted(&finite_dist, 0.5) / radius;
    // The median of per-feature percentiles: `pct` is monotone and `finite_dist` sorted, so it
    // is the mean of the middle one or two features' percentiles.
    let m = finite_dist.len();
    let dist_pct = match m {
        0 => f32::NAN,
        _ if m % 2 == 1 => pct(finite_dist[m / 2]),
        _ => 0.5 * (pct(finite_dist[m / 2 - 1]) + pct(finite_dist[m / 2])),
    };
    // `finite_dist` is sorted, so the rows at the hub are a prefix.
    let hub_share =
        finite_dist.partition_point(|&x| cloud.is_at_hub(x)) as f32 / finite_dist.len() as f32;

    // Below a neighbourhood's worth of cells or features the statistics are degenerate.
    let nearest = (n > NN_K && finite_dist.len() >= NN_K).then(|| {
        let cell_idx = super::strided_indices(n, MAX_NN_CELLS);
        let cell_pct: Vec<f32> = cell_idx.iter().map(|&i| pct(cloud.dist[i])).collect();
        let finite_rows: Vec<usize> = (0..d).filter(|&g| feat_dist[g].is_finite()).collect();
        nearest_cells(cells, &cell_idx, &cell_pct, feats, &finite_rows, h)
    });

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
        hub_share,
        n_feats: d,
        nearest,
    })
}

/// The report-only nearest-cell statistics over the sampled cells `cell_idx` (more than `NN_K`
/// of them, with distance percentiles `cell_pct`) and a strided sample of `finite_rows`.
fn nearest_cells(
    cells: &[f32],
    cell_idx: &[usize],
    cell_pct: &[f32],
    feats: &[f32],
    finite_rows: &[usize],
    h: usize,
) -> NearestCells {
    let sub_cells: Vec<&[f32]> = cell_idx
        .iter()
        .map(|&i| &cells[i * h..(i + 1) * h])
        .collect();
    // The `NN_K` sampled cells nearest `e`.
    let nearest = |e: &[f32]| -> Vec<usize> {
        let mut d2: Vec<(f32, usize)> = sub_cells
            .iter()
            .enumerate()
            .map(|(c, z)| (l2_sq(z, e), c))
            .collect();
        d2.select_nth_unstable_by(NN_K - 1, |a, b| a.0.total_cmp(&b.0));
        d2[..NN_K].iter().map(|&(_, c)| c).collect()
    };
    let nearest_pct =
        |e: &[f32]| nearest(e).iter().map(|&c| cell_pct[c]).sum::<f32>() / NN_K as f32;

    let mut per_feat: Vec<f32> = super::strided_indices(finite_rows.len(), MAX_NN_FEATS)
        .into_par_iter()
        .map(|i| finite_rows[i])
        .map(|g| nearest_pct(&feats[g * h..(g + 1) * h]))
        .collect();
    per_feat.sort_unstable_by(f32::total_cmp);

    // The reference: each sampled cell's local average (the mean of its `NN_K` nearest
    // cells), through the same statistic.
    let mut reference: Vec<f32> = super::strided_indices(sub_cells.len(), MAX_NN_REF)
        .into_par_iter()
        .map(|p| {
            let mut local = vec![0f32; h];
            for c in nearest(sub_cells[p]) {
                local
                    .iter_mut()
                    .zip(sub_cells[c])
                    .for_each(|(l, &x)| *l += x);
            }
            local.iter_mut().for_each(|l| *l /= NN_K as f32);
            nearest_pct(&local)
        })
        .collect();
    reference.sort_unstable_by(f32::total_cmp);

    let at = |v: &[f32], q| quantile_sorted(v, q);
    NearestCells {
        mean: per_feat.iter().sum::<f32>() / per_feat.len() as f32,
        q25: at(&per_feat, 0.25),
        median: at(&per_feat, 0.5),
        q75: at(&per_feat, 0.75),
        cells_q25: at(&reference, 0.25),
        cells_median: at(&reference, 0.5),
        cells_q75: at(&reference, 0.75),
    }
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
