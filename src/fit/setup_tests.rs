use super::*;
use crate::data::{load_unified_data, LoadUnifiedArgs};
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};

const CELLS: usize = 240;
const GENES: usize = 30;

/// Base rows: four cell groups, each with its own block of high genes.
fn base_counts() -> DMatrix<f32> {
    DMatrix::from_fn(GENES, CELLS, |g, c| {
        let group = c % 4;
        let high = g * 4 / GENES == group;
        (1 + (c * 7 + g * 13) % 3) as f32 + if high { 12.0 } else { 0.0 }
    })
}

/// Extra rows with a different, deeper pattern: high on every other cell.
fn extra_counts() -> DMatrix<f32> {
    DMatrix::from_fn(GENES, CELLS, |g, c| {
        let high = (c / 2 + g) % 2 == 0;
        (1 + (c * 5 + g * 3) % 4) as f32 + if high { 60.0 } else { 0.0 }
    })
}

fn write_backend(dir: &std::path::Path, stem: &str, counts: &DMatrix<f32>) -> Box<str> {
    let path: Box<str> = dir
        .join(format!("{stem}.zarr"))
        .to_string_lossy()
        .into_owned()
        .into();
    let mut b = create_sparse_from_dmatrix(counts, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create backend");
    let rows: Vec<Box<str>> = (0..counts.nrows())
        .map(|r| format!("GENE{r}").into())
        .collect();
    let cols: Vec<Box<str>> = (0..counts.ncols())
        .map(|c| format!("C{c}").into())
        .collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    path
}

fn load(path: Box<str>) -> UnifiedData {
    load_unified_data(LoadUnifiedArgs {
        data_files: vec![path],
        preload: true,
        ..Default::default()
    })
    .expect("load")
}

fn config() -> FitConfig {
    FitConfig {
        embedding_dim: 4,
        anchor_batches: None,
        bulk_batches: None,
        emit_finest_collapse: false,
        num_levels: 2,
        sort_dim: 4,
        knn_pb_samples: 5,
        num_opt_iter: 5,
        proj_dim: 8,
        epochs: 1,
        batches_per_epoch: None,
        batch_size: 64,
        learning_rate: 0.01,
        seed: 3,
        device: legume_numeric::candle::candle_core::Device::Cpu,
        block_size: None,
        hvg_weights: None,
        refine: data_beans::alg::refine_multilevel::RefineParams::default(),
        weight_decay: 0.0,
        unit_weight_decay: None,
        phase1_cells_per_pb: 0,
        hier_units_per_step: 64,
        hier_modules_per_unit: 2,
        module_only_min_rows: 0,
        feature_modules: None,
        displaced: None,
        preset_features: None,
        strata: None,
        cis_gates: None,
        flat_module_only: false,
        multiome: None,
    }
}

/// The pseudobulks the fit trains on depend on the live rows alone: a backend
/// that also holds rows the axis was cut away from (a split-off displaced
/// track) gives the same partition and the same pseudobulk counts as a
/// backend holding only the live rows.
#[test]
fn rows_outside_the_live_axis_do_not_shape_the_pseudobulks() {
    let dir = tempfile::tempdir().unwrap();
    let base = base_counts();
    let mut both = DMatrix::<f32>::zeros(2 * GENES, CELLS);
    both.rows_mut(0, GENES).copy_from(&base);
    both.rows_mut(GENES, GENES).copy_from(&extra_counts());

    let mut alone = load(write_backend(dir.path(), "alone", &base));
    let mut wide = load(write_backend(dir.path(), "wide", &both));
    wide.subset_features(&(0..GENES).collect::<Vec<_>>());
    assert_eq!(wide.count_backend().num_rows(), 2 * GENES);

    let a = build_pseudobulks(&mut alone, &config()).unwrap();
    let b = build_pseudobulks(&mut wide, &config()).unwrap();
    assert!(!a.masked);
    assert!(b.masked, "the wide backend is read through its live rows");
    assert_eq!(b.collapse_row_of_feature, (0..GENES).collect::<Vec<_>>());
    assert_eq!(a.cell_to_pb_per_level, b.cell_to_pb_per_level);
    for (la, lb) in a.collapsed_levels.iter().zip(&b.collapsed_levels) {
        let ma =
            gather_to_unified_axis(la.mu_observed.posterior_mean(), &a.collapse_row_of_feature);
        let mb =
            gather_to_unified_axis(lb.mu_observed.posterior_mean(), &b.collapse_row_of_feature);
        assert_eq!(ma.shape(), mb.shape());
        assert!(
            ma.iter()
                .zip(mb.iter())
                .all(|(x, y)| (x - y).abs() <= 1e-5 * (1.0 + x.abs())),
            "pseudobulk counts differ"
        );
    }
}

#[test]
fn gather_reads_each_features_row() {
    let rows = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = gather_to_unified_axis(&rows, &[2, 0]);
    assert_eq!(out, DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 1.0, 2.0]));
    assert_eq!(gather_to_unified_axis(&rows, &[0, 1, 2]), rows);
}
