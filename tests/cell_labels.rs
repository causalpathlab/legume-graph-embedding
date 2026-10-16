//! The per-cell label contract: `{prefix}.argmax.parquet` with columns `cell`, `cell_type`
//! and `probability`, written with the TSV pair beside it until 0.10, by the writer and by
//! both projection paths (the firm `annotate_embeddings_ora` and the deprecated
//! `annotate_embeddings`).
#![allow(deprecated)]

use graph_embedding_util::type_annotation::{
    annotate_embeddings, annotate_embeddings_ora, write_cell_labels, AnnotateProjConfig,
    InputEmbeddings, TermOraConfig, CELL_LABELS_SUFFIX, CELL_LABEL_FILE_SUFFIXES,
    TERM_ORA_OUTPUT_SUFFIXES,
};
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::parquet::read_table_columns;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("geu_cell_labels_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The parquet has one row per cell, in input order, with a non-empty label; every file in
/// [`CELL_LABEL_FILE_SUFFIXES`] exists. Returns the labels and probabilities read back.
fn assert_cell_labels(prefix: &str, cells: &[Box<str>]) -> (Vec<Box<str>>, Vec<f64>) {
    let (mut strings, mut numbers) = read_table_columns(
        &format!("{prefix}{CELL_LABELS_SUFFIX}"),
        &["cell", "cell_type"],
        &["probability"],
    )
    .unwrap();
    assert_eq!(strings[0], cells, "one row per cell, in input order");
    assert!(
        strings[1].iter().all(|label| !label.is_empty()),
        "non-empty labels"
    );
    assert_eq!(numbers[0].len(), cells.len());
    for suffix in CELL_LABEL_FILE_SUFFIXES {
        assert!(
            Path::new(&format!("{prefix}{suffix}")).exists(),
            "{suffix} not written"
        );
    }
    (strings.swap_remove(1), numbers.swap_remove(0))
}

#[test]
fn writes_argmax_parquet_beside_the_tsvs() {
    let dir = fresh_dir("write");
    let prefix = dir.join("run");
    let prefix = prefix.to_str().unwrap();

    let cells = names(&["c0", "c1", "c2"]);
    let labels = names(&["T cell", "B cell", "T cell"]);
    let probs = [0.123_456_7_f32, 1.0, f32::NAN];
    write_cell_labels(prefix, &cells, &labels, &probs).unwrap();

    assert_eq!(CELL_LABELS_SUFFIX, ".argmax.parquet");
    let (read_labels, read_probs) = assert_cell_labels(prefix, &cells);
    assert_eq!(read_labels, labels);
    assert_eq!(
        read_probs[0],
        f64::from(probs[0]),
        "probability at full f32 precision"
    );
    assert_eq!(read_probs[1], 1.0);
    assert!(read_probs[2].is_nan(), "a NaN probability stays NaN");

    let written: BTreeSet<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    let listed: BTreeSet<String> = CELL_LABEL_FILE_SUFFIXES
        .iter()
        .map(|s| format!("run{s}"))
        .collect();
    assert_eq!(written, listed, "CELL_LABEL_FILE_SUFFIXES lists every file");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn mismatched_lengths_are_an_error_before_any_file() {
    let dir = fresh_dir("lengths");
    let prefix = dir.join("run");
    let prefix = prefix.to_str().unwrap();

    let cells = names(&["c0", "c1", "c2"]);
    let labels = names(&["T cell", "B cell", "T cell"]);
    let err = write_cell_labels(prefix, &cells, &labels, &[0.5, 0.5]).unwrap_err();
    assert!(err.to_string().contains("for 3 cells"), "{err}");
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0, "no file written");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn term_ora_cleans_every_label_file() {
    for suffix in CELL_LABEL_FILE_SUFFIXES {
        assert!(
            TERM_ORA_OUTPUT_SUFFIXES.contains(suffix),
            "{suffix} missing from TERM_ORA_OUTPUT_SUFFIXES"
        );
    }
}

/// Run one projection path on a toy embedding and check its label files. 6 genes and 6 cells
/// in H = 2: GENE0..2 point +x, GENE3..5 point +y (three markers per type, the term-ORA
/// floor); three cells near each axis. `annotate` gets the input, the panel path and the
/// output prefix.
fn check_path(
    tag: &str,
    annotate: impl FnOnce(&InputEmbeddings<'_>, &str, &str) -> anyhow::Result<()>,
) {
    let dir = fresh_dir(tag);
    let feature_emb = DMatrix::<f32>::from_row_slice(
        6,
        2,
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    );
    let cell_emb = DMatrix::<f32>::from_row_slice(
        6,
        2,
        &[
            0.9, 0.1, 1.0, 0.0, 0.8, 0.2, // +x
            0.1, 0.9, 0.0, 1.0, 0.2, 0.8, // +y
        ],
    );
    let gene_names = names(&["GENE0", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]);
    let cell_names = names(&["cell0", "cell1", "cell2", "cell3", "cell4", "cell5"]);
    let markers = dir.join("markers.tsv");
    fs::write(
        &markers,
        "GENE0\tXtype\nGENE1\tXtype\nGENE2\tXtype\nGENE3\tYtype\nGENE4\tYtype\nGENE5\tYtype\n",
    )
    .unwrap();
    let prefix = dir.join("run");
    let prefix = prefix.to_str().unwrap();
    let input = InputEmbeddings {
        feature_emb: &feature_emb,
        gene_names: &gene_names,
        cell_emb: &cell_emb,
        cell_names: &cell_names,
    };
    annotate(&input, markers.to_str().unwrap(), prefix).unwrap();
    assert_cell_labels(prefix, &cell_names);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn term_ora_writes_cell_labels() {
    let cfg = TermOraConfig {
        knn: 3,
        n_perm: 20,
        ..TermOraConfig::default()
    };
    check_path("ora", |input, markers, prefix| {
        annotate_embeddings_ora(input, markers, prefix, true, &cfg)
    });
}

#[test]
fn projection_writes_cell_labels() {
    let cfg = AnnotateProjConfig {
        n_perm: 0, // raw cosine — fast & deterministic
        seed: 42,
        knn: 3,
        coarsen: true,
        ..AnnotateProjConfig::default()
    };
    check_path("proj", |input, markers, prefix| {
        annotate_embeddings(input, markers, prefix, true, &cfg).map(drop)
    });
}
