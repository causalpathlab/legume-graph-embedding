//! The per-cell label contract: `{prefix}.argmax.parquet` with columns `cell`, `cell_type`
//! and `probability`, written with the TSV pair beside it until 0.10.

use graph_embedding_util::type_annotation::{
    write_cell_labels, CELL_LABELS_SUFFIX, TERM_ORA_OUTPUT_SUFFIXES,
};
use legume_numeric::matrix::parquet::read_table_columns;
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
    let (strings, numbers) = read_table_columns(
        &format!("{prefix}{CELL_LABELS_SUFFIX}"),
        &["cell", "cell_type"],
        &["probability"],
    )
    .unwrap();
    assert_eq!(strings, [cells, labels]);
    let read = &numbers[0];
    assert_eq!(read.len(), 3);
    assert_eq!(
        read[0],
        f64::from(probs[0]),
        "probability at full f32 precision"
    );
    assert_eq!(read[1], 1.0);
    assert!(read[2].is_nan(), "a NaN probability stays NaN");

    for tsv in [".argmax.tsv", ".membership.tsv"] {
        assert!(
            Path::new(&format!("{prefix}{tsv}")).exists(),
            "{tsv} is still written until 0.10"
        );
    }
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
    for suffix in [CELL_LABELS_SUFFIX, ".argmax.tsv", ".membership.tsv"] {
        assert!(
            TERM_ORA_OUTPUT_SUFFIXES.contains(&suffix),
            "{suffix} missing from TERM_ORA_OUTPUT_SUFFIXES"
        );
    }
}
