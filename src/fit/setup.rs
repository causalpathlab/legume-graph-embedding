//! Everything upstream of training: the batch-corrected projection, the multilevel
//! collapse it feeds, and the per-level pseudobulk views phase 1's axes are built
//! from.
//!
//! One module because it is one dependency chain — the projection exists only to hash
//! cells into the collapse, and the collapse exists only to produce these blobs. None
//! of it touches a model, a Var or an optimizer.

use super::config::FitConfig;
use crate::data::UnifiedData;
use data_beans::alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, MultilevelParams,
};
use data_beans::alg::random_projection::RandProjOps;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::param::traits::Inference;
use log::info;
use nalgebra::DMatrix;

/// The collapse, ordered **coarsest → finest**, paired with the per-level pseudobulk
/// views and the cell→pb maps.
///
/// Kept as one struct rather than unpacked at the call site on purpose: `fit()` used to
/// split these three apart immediately and then thread the pieces through every
/// downstream stage separately, which is what pushed several of its would-be helper
/// functions past a defensible parameter count. They are one object; passing them as
/// one keeps the seams below it honest.
pub(super) struct Pseudobulks {
    pub collapsed_levels: Vec<data_beans::alg::collapse_data::CollapsedOut>,
    /// `cell_to_pb_per_level[l][c]` is cell `c`'s pseudobulk at level `l`.
    pub cell_to_pb_per_level: Vec<Vec<usize>>,
    /// One `UnifiedData` per level, on the unified feature axis.
    pub blobs: Vec<UnifiedData>,
    /// Feature → row of the collapse's per-row outputs (`mu_*`, `delta`,
    /// `observed_counts`). The backend row itself when the collapse read the
    /// whole backend; the row's rank among the live rows when it read only
    /// those (see [`build_pseudobulks`]).
    pub collapse_row_of_feature: Vec<usize>,
    /// The batch names the collapse registered, in the order of its `delta`
    /// columns; `None` when it registered none.
    pub batch_names: Option<Vec<Box<str>>>,
    /// Whether the collapse read only the live feature rows of a wider backend.
    pub masked: bool,
}

/// Project, collapse, and materialize the per-level pseudobulk views.
///
/// The projection and the collapse read only the backend rows of the live feature
/// axis. When the backend holds more rows than that axis (a divergent track split
/// off it, see [`crate::fit::divergence::split_divergence`]), they run on a clone of
/// the backend with the other rows masked out, so those rows cannot shape the
/// pseudobulks the fit trains on.
///
/// `sort_dim` controls how many bits of the binary-sketched projection are used to hash
/// cells into the *finest* pb-sample partition, so `2^sort_dim` bounds the number of
/// distinct codes at that level. It is exposed directly on [`FitConfig`] for parity with
/// `senna topic` / `svd` rather than derived from a target count.
pub(super) fn build_pseudobulks(
    unified: &mut UnifiedData,
    config: &FitConfig,
) -> anyhow::Result<Pseudobulks> {
    let n_features = unified.n_features();
    let batch_labels: Vec<Box<str>> = unified.batch_labels();
    let n_batches = unified.n_batches();
    let backend_rows = unified.count_backend().num_rows();
    let mut keep = vec![false; backend_rows];
    for &brow in &unified.feature_to_backend_row {
        keep[brow] = true;
    }
    let masked = keep.iter().any(|&k| !k);
    // The live rows, renumbered compactly in backend order: what a masked
    // backend calls them.
    let collapse_row_of_feature: Vec<usize> = if masked {
        let mut rank = vec![usize::MAX; backend_rows];
        let mut next = 0usize;
        for (brow, &k) in keep.iter().enumerate() {
            if k {
                rank[brow] = next;
                next += 1;
            }
        }
        unified
            .feature_to_backend_row
            .iter()
            .map(|&brow| rank[brow])
            .collect()
    } else {
        unified.feature_to_backend_row.clone()
    };
    let mut live_view = if masked {
        info!(
            "Projection and collapse on the {} live of {backend_rows} backend rows",
            n_features
        );
        let mut view = unified.count_backend().clone_for_collapse();
        view.mask_rows(&keep)?;
        Some(view)
    } else {
        None
    };
    let backend: &mut SparseIoVec = match live_view.as_mut() {
        Some(view) => view,
        None => unified.count_backend_mut(),
    };

    let proj_out = project(
        backend,
        config,
        &batch_labels,
        n_batches,
        &collapse_row_of_feature,
    )?;

    info!(
        "Multilevel collapse (sort_dim={}, {} levels requested)...",
        config.sort_dim, config.num_levels
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        backend,
        &proj_out.proj,
        &batch_labels,
        &MultilevelParams {
            knn_pb_samples: config.knn_pb_samples,
            num_levels: config.num_levels.max(1),
            sort_dim: config.sort_dim,
            num_opt_iter: config.num_opt_iter,
            refine: config.refine.clone(),
            // Only `posterior_mean()` is ever read off this, so skip the sd / log_mean /
            // log_sd planes — that is the bulk of the coarsen-stage memory at high
            // pb-sample counts.
            output_calibration: legume_numeric::param::traits::CalibrateTarget::MeanOnly,
            anchor_batches: config.anchor_batches.clone(),
            bulk_batches: config.bulk_batches.clone(),
            observe_panels: true,
            // The feature partition reads the finest level's counts
            // (`CollapsedOut::observed_counts`), and the pseudobulk
            // reference serializes them; both need the sufficient statistics.
            keep_finest_stats: true,
            pb_tree: None,
            strata: config.strata.clone(),
        },
    )?;
    let batch_names = backend.batch_names();
    let mut collapsed_levels = collapse_out.levels;
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    // The collapse emits finest-first. Reverse both so levels run coarsest..finest —
    // `senna topic` uses the same order, so its curriculum trains coarse first.
    collapsed_levels.reverse();
    cell_to_pb_per_level.reverse();

    // pb counts live on the unified feature axis: gather each feature's collapse row.
    let mut blobs: Vec<UnifiedData> = Vec::with_capacity(collapsed_levels.len());
    for collapsed in &collapsed_levels {
        let pb_full: &DMatrix<f32> = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        let pb_count_ds = gather_to_unified_axis(pb_full, &collapse_row_of_feature);
        blobs.push(UnifiedData::from_pseudobulks(
            &pb_count_ds,
            unified.feature_names.clone(),
            unified.feature_to_backend_row.clone(),
        )?);
    }

    // The flat cell↔feature edge list is intentionally NOT built. The cell axis is always
    // `PerBatchStratified`, whose sampler streams columns in `build_active_samplers` and
    // is self-contained at sample time, so `unified.triplets` stays empty for it.
    Ok(Pseudobulks {
        collapsed_levels,
        cell_to_pb_per_level,
        blobs,
        collapse_row_of_feature,
        batch_names,
        masked,
    })
}

/// The batch-corrected random projection the collapse hashes on, over `backend`
/// (the rows the collapse reads), HVG-weighted when the caller supplied weights.
/// `collapse_row_of_feature` places each feature's weight on its backend row; rows
/// no feature maps to get 0 and sit out the projection basis.
fn project(
    backend: &SparseIoVec,
    config: &FitConfig,
    batch_labels: &[Box<str>],
    n_batches: usize,
    collapse_row_of_feature: &[usize],
) -> anyhow::Result<data_beans::alg::random_projection::RandColProjOut> {
    info!(
        "Batch-corrected projection (proj_dim={}, {} batches)...",
        config.proj_dim, n_batches
    );
    let batch_arg = (n_batches > 1).then_some(batch_labels);
    let backend_w: Option<Vec<f32>> = match config.hvg_weights.as_deref() {
        None => None,
        Some(w) => {
            anyhow::ensure!(
                w.len() == collapse_row_of_feature.len(),
                "hvg_weights length {} != n_features {} (the HVG mask must be aligned to the \
                 unified feature axis BEFORE any subset/coarsening — pass full-axis weights from \
                 the wrapper)",
                w.len(),
                collapse_row_of_feature.len()
            );
            info!(
                "HVG-weighted projection: {} weighted features (>= 1.0)",
                w.iter().filter(|&&x| x > 0.0).count()
            );
            let mut backend_w = vec![0.0f32; backend.num_rows()];
            for (feature, &row) in collapse_row_of_feature.iter().enumerate() {
                backend_w[row] = w[feature];
            }
            Some(backend_w)
        }
    };

    project_backend(
        backend,
        config.proj_dim,
        config.block_size,
        batch_arg,
        backend_w.as_deref(),
        config.seed,
    )
}

/// One random projection over `backend`, weighted when `row_weights` is given (length =
/// `backend.num_rows()`).
fn project_backend<T>(
    backend: &SparseIoVec,
    proj_dim: usize,
    block_size: Option<usize>,
    batch_arg: Option<&[T]>,
    row_weights: Option<&[f32]>,
    seed: u64,
) -> anyhow::Result<data_beans::alg::random_projection::RandColProjOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    match row_weights {
        None => backend
            .project_columns_with_batch_correction_seeded(proj_dim, block_size, batch_arg, seed),
        Some(w) => {
            backend.project_columns_weighted_seeded(proj_dim, block_size, batch_arg, w, seed)
        }
    }
}

/// Gather a collapse-row matrix onto the unified feature axis: row `f` of the result
/// is row `row_of_feature[f]` of `rows`. A clone when the map is the identity, which
/// is every run without a feature subset.
pub(super) fn gather_to_unified_axis(
    rows: &DMatrix<f32>,
    row_of_feature: &[usize],
) -> DMatrix<f32> {
    let identity = rows.nrows() == row_of_feature.len()
        && row_of_feature.iter().enumerate().all(|(f, &r)| f == r);
    if identity {
        return rows.clone();
    }
    let cols = rows.ncols();
    let mut out = DMatrix::<f32>::zeros(row_of_feature.len(), cols);
    for (f, &r) in row_of_feature.iter().enumerate() {
        for s in 0..cols {
            out[(f, s)] = rows[(r, s)];
        }
    }
    out
}

#[cfg(test)]
#[path = "setup_tests.rs"]
mod tests;
