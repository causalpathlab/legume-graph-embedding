//! `type_annotation::embedding_score` against golden tables from an independent reference
//! implementation, on a synthetic panel with overlapping markers, a gene in every type (IDF
//! 0), genes off the feature axis, and a type left with too few markers.

use graph_embedding_util::type_annotation::embedding_score::{
    balanced_centre, calls, cell_covariance, scores, FeatureSpace,
};
use nalgebra::DMatrix;
use std::collections::HashMap;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/embedding_score/");

/// A TSV with a header: its column names after the first, and each row's key and values.
struct Table {
    columns: Vec<String>,
    rows: Vec<(String, Vec<String>)>,
}

fn read(name: &str) -> Table {
    let text = std::fs::read_to_string(format!("{DIR}{name}")).unwrap();
    let mut lines = text.lines().map(|l| l.split('\t').map(String::from));
    let columns = lines.next().unwrap().skip(1).collect();
    let rows = lines
        .map(|mut fields| (fields.next().unwrap(), fields.collect()))
        .collect();
    Table { columns, rows }
}

impl Table {
    fn keys(&self) -> Vec<String> {
        self.rows.iter().map(|(k, _)| k.clone()).collect()
    }

    fn matrix(&self) -> DMatrix<f64> {
        DMatrix::from_fn(self.rows.len(), self.columns.len(), |i, j| {
            self.rows[i].1[j].parse().unwrap()
        })
    }
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-9 * want.abs().max(1e-6)
}

fn assert_matrix(got: &DMatrix<f64>, want: &DMatrix<f64>, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(*a, *b), "{what}[{i}]: {a} vs {b}");
    }
}

#[test]
fn matches_the_golden_tables() {
    let feats = read("feature_embedding.tsv");
    let cells = read("cell_embedding.tsv");
    let genes: Vec<Box<str>> = feats.keys().into_iter().map(Box::from).collect();
    let (rho, z) = (feats.matrix(), cells.matrix());

    let space = FeatureSpace::new(&rho, &cell_covariance(&z).unwrap(), &genes).unwrap();
    let sigma = read("sigma.tsv");
    assert_eq!(sigma.keys(), feats.keys());
    assert_matrix(
        &DMatrix::from_column_slice(genes.len(), 1, space.sigma().as_slice()),
        &sigma.matrix(),
        "sigma",
    );

    // Weighted sets, as (type, gene) → weight. Panel types keep their spelling here; the
    // tables carry their lowercase label keys.
    let panel = space.marker_sets(&format!("{DIR}panel.tsv"), 0.0).unwrap();
    let types: Vec<String> = panel.type_names.iter().map(|t| t.to_lowercase()).collect();
    let got: HashMap<(String, String), f64> = types
        .iter()
        .zip(&panel.sets)
        .flat_map(|(t, set)| {
            set.iter()
                .map(|&(g, w)| ((t.clone(), genes[g as usize].to_string()), w))
        })
        .collect();
    let want: HashMap<(String, String), f64> = read("weighted_sets.tsv")
        .rows
        .into_iter()
        .map(|(t, v)| ((t, v[0].clone()), v[1].parse().unwrap()))
        .collect();
    assert_eq!(
        got.keys().collect::<std::collections::BTreeSet<_>>(),
        want.keys().collect(),
        "weighted set membership"
    );
    for (key, w) in &want {
        assert!(close(got[key], *w), "weight {key:?}: {} vs {w}", got[key]);
    }

    let dirs = space.directions(&panel.sets);
    let want_dirs = read("directions.tsv");
    assert_eq!(types, want_dirs.keys(), "live types, in order");
    assert_matrix(&dirs, &want_dirs.matrix(), "directions");

    let clusters = read("clusters.tsv");
    assert_eq!(clusters.keys(), cells.keys());
    let groups: Vec<usize> = clusters
        .rows
        .iter()
        .map(|(_, v)| v[0].parse().unwrap())
        .collect();
    let centre = balanced_centre(&z, &groups).unwrap();
    assert_matrix(
        &DMatrix::from_column_slice(centre.len(), 1, centre.as_slice()),
        &read("balanced_centre.tsv").matrix(),
        "balanced centre",
    );

    let s = scores(&z, &centre, &dirs);
    let want_scores = read("scores.tsv");
    assert_eq!(want_scores.columns, types);
    assert_matrix(&s, &want_scores.matrix(), "scores");

    let want_calls = read("calls.tsv");
    assert_eq!(want_calls.keys(), cells.keys());
    for (call, (cell, v)) in calls(&s, &dirs).iter().zip(&want_calls.rows) {
        let t = call.call.expect("every cell has a live type");
        assert_eq!(types[t], v[0], "call for {cell}");
        let margin: f64 = v[1].parse().unwrap();
        assert!(
            close(call.margin, margin),
            "margin for {cell}: {} vs {margin}",
            call.margin
        );
    }
}
