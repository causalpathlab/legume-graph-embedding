//! End-to-end check that both projection paths, the firm `annotate_embeddings_ora` and the
//! deprecated `annotate_embeddings`, emit the per-cell label table `argmax.parquet` (cell,
//! coarse label, probability) and, until 0.10, the TSV pair beside it.
#![allow(deprecated)]

use graph_embedding_util::type_annotation::{
    annotate_embeddings, annotate_embeddings_ora, AnnotateProjConfig, InputEmbeddings,
    TermOraConfig, CELL_LABELS_SUFFIX,
};
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::parquet::read_table_columns;
use std::fs;
use std::path::{Path, PathBuf};

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

/// 6 genes and 6 cells in H = 2, two marker types along the two axes. Returns the embeddings,
/// a fresh directory and the panel path inside it.
struct Fixture {
    feature_emb: DMatrix<f32>,
    gene_names: Vec<Box<str>>,
    cell_emb: DMatrix<f32>,
    cell_names: Vec<Box<str>>,
    dir: PathBuf,
    markers: String,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        // GENE0..2 point +x, GENE3..5 point +y: three markers per type, the term-ORA floor.
        let feature_emb = DMatrix::<f32>::from_row_slice(
            6,
            2,
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        );
        // Three cells near +x, three near +y: two resolvable clusters.
        let cell_emb = DMatrix::<f32>::from_row_slice(
            6,
            2,
            &[
                0.9, 0.1, 1.0, 0.0, 0.8, 0.2, // +x
                0.1, 0.9, 0.0, 1.0, 0.2, 0.8, // +y
            ],
        );
        let dir = std::env::temp_dir().join(format!(
            "geu_annotate_cell_labels_{tag}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let markers = dir.join("markers.tsv");
        fs::write(
            &markers,
            "GENE0\tXtype\nGENE1\tXtype\nGENE2\tXtype\nGENE3\tYtype\nGENE4\tYtype\nGENE5\tYtype\n",
        )
        .unwrap();
        Self {
            feature_emb,
            gene_names: names(&["GENE0", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]),
            cell_emb,
            cell_names: names(&["cell0", "cell1", "cell2", "cell3", "cell4", "cell5"]),
            markers: markers.to_str().unwrap().to_owned(),
            dir,
        }
    }

    fn input(&self) -> InputEmbeddings<'_> {
        InputEmbeddings {
            feature_emb: &self.feature_emb,
            gene_names: &self.gene_names,
            cell_emb: &self.cell_emb,
            cell_names: &self.cell_names,
        }
    }

    fn prefix(&self) -> String {
        self.dir.join("run").to_str().unwrap().to_owned()
    }

    /// One parquet row per cell, in input order, with a label and a probability; the TSV
    /// pair beside it.
    fn assert_cell_labels(&self) {
        let prefix = self.prefix();
        let (strings, numbers) = read_table_columns(
            &format!("{prefix}{CELL_LABELS_SUFFIX}"),
            &["cell", "cell_type"],
            &["probability"],
        )
        .unwrap();
        assert_eq!(
            strings[0], self.cell_names,
            "one row per cell, in input order"
        );
        assert!(
            strings[1].iter().all(|label| !label.is_empty()),
            "non-empty coarse labels"
        );
        assert_eq!(numbers[0].len(), self.cell_names.len());
        for tsv in [".argmax.tsv", ".membership.tsv"] {
            assert!(
                Path::new(&format!("{prefix}{tsv}")).exists(),
                "{tsv} is still written until 0.10"
            );
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn term_ora_writes_cell_labels() {
    let fx = Fixture::new("ora");
    let cfg = TermOraConfig {
        knn: 3,
        n_perm: 20,
        ..TermOraConfig::default()
    };
    annotate_embeddings_ora(&fx.input(), &fx.markers, &fx.prefix(), true, &cfg)
        .expect("annotate_embeddings_ora failed");
    fx.assert_cell_labels();
}

#[test]
fn projection_writes_cell_labels() {
    let fx = Fixture::new("proj");
    let cfg = AnnotateProjConfig {
        n_perm: 0, // raw cosine — fast & deterministic
        seed: 42,
        knn: 3,
        coarsen: true,
        ..AnnotateProjConfig::default()
    };
    annotate_embeddings(&fx.input(), &fx.markers, &fx.prefix(), true, &cfg)
        .expect("annotate_embeddings failed");
    fx.assert_cell_labels();
}
