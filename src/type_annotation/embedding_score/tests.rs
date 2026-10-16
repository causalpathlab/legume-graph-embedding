use super::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const H: usize = 4;
const G: usize = 20;
/// Unequal clusters, so the balanced centre differs from the plain mean.
const SIZES: [usize; 3] = [40, 15, 8];

struct Fixture {
    z: DMatrix<f64>,
    rho: DMatrix<f64>,
    groups: Vec<usize>,
    genes: Vec<Box<str>>,
    sets: Vec<WeightedSet>,
}

fn fixture(seed: u64) -> Fixture {
    let mut rng = SmallRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let groups: Vec<usize> = SIZES
        .iter()
        .enumerate()
        .flat_map(|(k, &n)| std::iter::repeat_n(k, n))
        .collect();
    let means = DMatrix::from_fn(SIZES.len(), H, |_, _| 2.0 * normal.sample(&mut rng));
    let z = DMatrix::from_fn(groups.len(), H, |i, j| {
        means[(groups[i], j)] + normal.sample(&mut rng)
    });
    let rho = DMatrix::from_fn(G, H, |_, _| normal.sample(&mut rng));
    let genes = (0..G).map(|g| format!("g{g:02}").into()).collect();
    let sets = vec![
        vec![(0, 1.6), (1, 1.6), (2, 0.9), (3, 0.5)],
        vec![(3, 0.5), (4, 1.6), (5, 1.6), (6, 1.6)],
        vec![(7, 1.0), (8, 1.0), (9, 1.0), (10, 0.9), (11, 0.4)],
    ];
    Fixture {
        z,
        rho,
        groups,
        genes,
        sets,
    }
}

impl Fixture {
    fn space(&self) -> FeatureSpace {
        FeatureSpace::new(&self.rho, &cell_covariance(&self.z).unwrap(), &self.genes).unwrap()
    }

    fn scores(&self) -> DMatrix<f64> {
        let centre = balanced_centre(&self.z, &self.groups).unwrap();
        scores(&self.z, &centre, &self.space().directions(&self.sets))
    }
}

/// The score the closed form stands for, built the long way: every gene's logit
/// `ρ_g·z_i + b_g` over every cell, standardised per gene (centred at the mean of its cluster
/// means, scaled by its sample standard deviation), then each set's weighted mean.
fn materialised(f: &Fixture, bias: &[f64]) -> DMatrix<f64> {
    let mut x = &f.z * f.rho.transpose();
    for (g, mut col) in x.column_iter_mut().enumerate() {
        col.add_scalar_mut(bias[g]);
    }
    let n = x.nrows() as f64;
    let k = SIZES.len();
    for mut col in x.column_iter_mut() {
        let mut sums = vec![0.0; k];
        let mut counts = vec![0.0; k];
        for (i, &v) in col.iter().enumerate() {
            sums[f.groups[i]] += v;
            counts[f.groups[i]] += 1.0;
        }
        let centre = sums.iter().zip(&counts).map(|(s, c)| s / c).sum::<f64>() / k as f64;
        let mean = col.mean();
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        col.apply(|v| *v = (*v - centre) / sd);
    }
    DMatrix::from_fn(x.nrows(), f.sets.len(), |i, t| {
        let set = &f.sets[t];
        let total: f64 = set.iter().map(|&(_, w)| w).sum();
        set.iter()
            .map(|&(g, w)| w * x[(i, g as usize)])
            .sum::<f64>()
            / total
    })
}

fn assert_close(got: &DMatrix<f64>, want: &DMatrix<f64>, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    for (a, b) in got.iter().zip(want.iter()) {
        assert!(
            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
            "{what}: {a} vs {b}"
        );
    }
}

#[test]
fn closed_form_equals_the_materialised_gene_z_score_marker_mean() {
    let f = fixture(1);
    assert_close(&f.scores(), &materialised(&f, &[0.0; G]), "closed form");
}

/// The decoder's per-gene bias shifts a gene's logit in every cell alike; standardising
/// removes it, so the closed form needs no bias at all.
#[test]
fn the_feature_bias_cancels() {
    let f = fixture(2);
    let mut rng = SmallRng::seed_from_u64(20);
    let normal = Normal::new(0.0, 5.0).unwrap();
    let bias: Vec<f64> = (0..G).map(|_| normal.sample(&mut rng)).collect();
    assert_close(&f.scores(), &materialised(&f, &bias), "with a bias");
}

/// A rotation of the latent space, applied to cells and genes together, leaves every logit
/// unchanged, so it must leave the scores and the calls unchanged.
#[test]
fn calls_are_invariant_to_a_joint_rotation() {
    let f = fixture(3);
    let mut rng = SmallRng::seed_from_u64(30);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let q = DMatrix::from_fn(H, H, |_, _| normal.sample(&mut rng))
        .qr()
        .q();
    let rotated = Fixture {
        z: &f.z * &q,
        rho: &f.rho * &q,
        ..fixture(3)
    };
    let (before, after) = (f.scores(), rotated.scores());
    assert_close(&after, &before, "rotated scores");
    let dirs = f.space().directions(&f.sets);
    let called = |s: &DMatrix<f64>, d: &DMatrix<f64>| -> Vec<Option<usize>> {
        calls(s, d).iter().map(|c| c.call).collect()
    };
    assert_eq!(
        called(&after, &rotated.space().directions(&rotated.sets)),
        called(&before, &dirs)
    );
}

/// `σ_g` scales with `ρ_g`, so a gene's standardised logit, and with it every score, ignores
/// the length of its row.
#[test]
fn scores_are_invariant_to_rescaling_rho() {
    let f = fixture(4);
    let mut rescaled = fixture(4);
    for (g, mut row) in rescaled.rho.row_iter_mut().enumerate() {
        row *= 0.1 + 3.0 * g as f64;
    }
    assert_close(&rescaled.scores(), &f.scores(), "rescaled rho");
}

#[test]
fn an_empty_set_gives_a_zero_direction_and_is_never_called() {
    let f = fixture(5);
    let mut sets = f.sets.clone();
    sets.push(Vec::new());
    let dirs = f.space().directions(&sets);
    assert!(dirs.row(3).iter().all(|&x| x == 0.0), "{}", dirs.row(3));

    // A dead type scores 0 everywhere; a cell whose live scores are all negative must still
    // take a live type, not the dead one.
    let centre = balanced_centre(&f.z, &f.groups).unwrap();
    let s = scores(&f.z, &centre, &dirs);
    assert!(calls(&s, &dirs)
        .iter()
        .all(|c| c.call.is_some_and(|t| t < 3)));

    let none = calls(&s, &DMatrix::zeros(4, H));
    assert!(none.iter().all(|c| c.call.is_none() && c.margin.is_nan()));
}

/// A zeroed row has `σ_g = 0`: it carries no direction, and it must not dilute its set's mean
/// as though it were a marker scoring zero.
#[test]
fn a_dead_feature_row_is_left_out_of_its_set() {
    let f = fixture(6);
    let mut dead = fixture(6);
    dead.rho.row_mut(19).fill(0.0);
    let space = dead.space();
    assert_eq!(space.sigma()[19], 0.0);
    let mut with_dead = f.sets.clone();
    with_dead[0].push((19, 2.0));
    assert_close(
        &space.directions(&with_dead),
        &space.directions(&f.sets),
        "set with a dead row",
    );
}

#[test]
fn the_balanced_centre_weights_clusters_equally() {
    let cells = DMatrix::from_row_slice(5, 1, &[0.0, 0.0, 0.0, 10.0, 99.0]);
    let groups = [0, 0, 0, 1, crate::type_annotation::UNASSIGNED];
    let centre = balanced_centre(&cells, &groups).unwrap();
    assert!((centre[0] - 5.0).abs() < 1e-12, "{centre}");
}

/// The marker floor counts live markers: a type with three matches, two of them dead rows,
/// would score on one gene, so it is dropped. Dead rows leave after IDF, so `C` still counts
/// every panel type and the survivors' weights are unchanged.
#[test]
fn the_marker_floor_counts_live_markers() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!(
        "geu_embedding_score_live_floor_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("markers.tsv");
    let mut file = std::fs::File::create(&path).unwrap();
    for (genes, ty) in [
        (["g00", "g01", "g02"], "Live"),
        (["g12", "g13", "g14"], "Thin"),
        (["g03", "g04", "g05"], "Other"),
    ] {
        for g in genes {
            writeln!(file, "{g}\t{ty}").unwrap();
        }
    }
    drop(file);

    let mut f = fixture(7);
    f.rho.row_mut(13).fill(0.0);
    f.rho.row_mut(14).fill(0.0);
    let panel = f.space().marker_sets(path.to_str().unwrap(), 0.0);
    std::fs::remove_dir_all(&dir).unwrap();
    let panel = panel.unwrap();
    let names: Vec<&str> = panel.type_names.iter().map(AsRef::as_ref).collect();
    assert_eq!(names, ["Live", "Other"]);
    for set in &panel.sets {
        assert_eq!(set.len(), 3, "{set:?}");
        assert!(
            set.iter().all(|&(_, w)| (w - 3f64.ln()).abs() < 1e-12),
            "{set:?}"
        );
    }
}

/// A centre from another embedding (wrong length) is a caller error, not a silent shift.
#[test]
#[should_panic(expected = "centre")]
fn a_centre_of_the_wrong_length_is_refused() {
    let f = fixture(8);
    let dirs = f.space().directions(&f.sets);
    let _ = scores(&f.z, &DVector::zeros(H - 1), &dirs);
}

/// A NaN score is not a candidate, wherever it sits among the types: the call goes to the best
/// finite score, and a cell with none has no call.
#[test]
fn nan_scores_are_never_called() {
    let dirs = DMatrix::from_element(3, H, 1.0);
    let s = DMatrix::from_row_slice(
        3,
        3,
        &[
            f64::NAN,
            1.0,
            0.5, //
            1.0,
            0.5,
            f64::NAN, //
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ],
    );
    let got = calls(&s, &dirs);
    assert_eq!(got[0].call, Some(1));
    assert!((got[0].margin - 0.5).abs() < 1e-12, "{:?}", got[0]);
    assert_eq!(got[1].call, Some(0));
    assert!((got[1].margin - 0.5).abs() < 1e-12, "{:?}", got[1]);
    assert_eq!(got[2].call, None);
    assert!(got[2].margin.is_nan());
}
