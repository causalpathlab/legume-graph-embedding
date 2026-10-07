use super::*;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

const H: usize = 3;
const S: usize = 60;

/// Random frozen base tables.
fn frozen(rng: &mut StdRng) -> Frozen {
    let n = Normal::new(0.0f32, 0.7).unwrap();
    Frozen {
        rho: (0..S * H).map(|_| n.sample(rng)).collect(),
        b: (0..S).map(|_| n.sample(rng)).collect(),
        h: H,
    }
}

/// `n_draws` reads among the genes with log weights `logw`.
fn draw(logw: &[f32], n_draws: usize, rng: &mut StdRng) -> Vec<(u32, f32)> {
    let w: Vec<f64> = logw.iter().map(|&x| f64::from(x).exp()).collect();
    let pick = WeightedIndex::new(&w).unwrap();
    let mut counts = vec![0f32; S];
    for _ in 0..n_draws {
        counts[pick.sample(rng)] += 1.0;
    }
    counts
        .into_iter()
        .enumerate()
        .filter(|&(_, c)| c > 0.0)
        .map(|(g, c)| (g as u32, c))
        .collect()
}

/// Two groups of units, displaced `+v` and `−v`; the displaced track's gene
/// ratios shifted from the base's.
struct World {
    frozen: Frozen,
    units: Vec<UnitCounts>,
    group: Vec<usize>,
    v: [f32; H],
}

fn world(
    n_units: usize,
    base_reads: usize,
    displaced_reads: usize,
    shift: f32,
    seed: u64,
) -> World {
    let mut rng = StdRng::seed_from_u64(seed);
    let frozen = frozen(&mut rng);
    let n = Normal::new(0.0f32, 1.0).unwrap();
    let v = [0.9 * shift, -0.6 * shift, 0.4 * shift];
    let delta: Vec<f32> = (0..S).map(|_| 0.3 * n.sample(&mut rng)).collect();
    let mut units = Vec::new();
    let mut group = Vec::new();
    for u in 0..n_units {
        let gr = u % 2;
        let sign = if gr == 0 { 1.0 } else { -1.0 };
        let th: Vec<f32> = (0..H).map(|_| 0.8 * n.sample(&mut rng)).collect();
        let d: Vec<f32> = (0..H).map(|k| sign * v[k]).collect();
        let base: Vec<f32> = (0..S)
            .map(|g| dot(&th, frozen.rho_row(g)) + frozen.b[g])
            .collect();
        let disp: Vec<f32> = (0..S)
            .map(|g| base[g] + delta[g] + dot(&d, frozen.rho_row(g)))
            .collect();
        units.push(UnitCounts {
            base: draw(&base, base_reads, &mut rng),
            displaced: draw(&disp, displaced_reads, &mut rng),
        });
        group.push(gr);
    }
    World {
        frozen,
        units,
        group,
        v,
    }
}

/// The mean displacement of each group.
fn group_means(d: &DMatrix<f32>, group: &[usize]) -> [[f32; H]; 2] {
    let mut sum = [[0f32; H]; 2];
    let mut n = [0f32; 2];
    for (u, &g) in group.iter().enumerate() {
        for k in 0..H {
            sum[g][k] += d[(u, k)];
        }
        n[g] += 1.0;
    }
    for (row, &count) in sum.iter_mut().zip(&n) {
        for x in row.iter_mut() {
            *x /= count.max(1.0);
        }
    }
    sum
}

fn cosine(a: &[f32; H], b: &[f32; H]) -> f32 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-12)
}

fn norm(a: &[f32; H]) -> f32 {
    dot(a, a).sqrt()
}

fn knobs() -> DisplacedTrackConfig {
    let mut k = DisplacedTrackConfig::new(DisplacedAxis {
        base_backend_row: Vec::new(),
        displaced_backend_row: Vec::new(),
        track_name: "count/unspliced".into(),
    });
    k.pb_epochs = 150;
    k.learning_rate = 0.05;
    k.l2_pb = 0.1;
    k
}

/// The two groups' displacements, after centring, point along `±v`.
fn assert_recovered(w: &World, fit: &mut PbFit, label: &str) {
    center_units(fit, &w.frozen);
    let means = group_means(&fit.d, &w.group);
    let neg_v = [-w.v[0], -w.v[1], -w.v[2]];
    assert!(
        cosine(&means[0], &w.v) > 0.9,
        "{label}: {:?} vs {:?}",
        means[0],
        w.v
    );
    assert!(cosine(&means[1], &neg_v) > 0.9, "{label}: {:?}", means[1]);
}

#[test]
fn pseudobulk_displacements_are_recovered() {
    let w = world(60, 3000, 1500, 1.0, 7);
    let mut fit = fit_pseudobulks(&w.frozen, &w.units, &knobs(), 1, &Device::Cpu).unwrap();
    assert_recovered(&w, &mut fit, "pseudobulks");
}

#[test]
fn no_displacement_gives_small_displacements() {
    let w = world(60, 3000, 1500, 0.0, 9);
    let mut fit = fit_pseudobulks(&w.frozen, &w.units, &knobs(), 1, &Device::Cpu).unwrap();
    center_units(&mut fit, &w.frozen);
    let means = group_means(&fit.d, &w.group);
    let shifted = world(60, 3000, 1500, 1.0, 9);
    for m in &means {
        assert!(norm(m) < 0.25 * norm(&shifted.v), "{m:?}");
    }
}

#[test]
fn the_steady_anchor_is_the_ratio_at_both_ends_of_a_genes_base_score() {
    // One dimension, one gene; units ordered along it. The two lowest and two
    // highest units (5% of 40 at each end) sit at log ratio 0.7; the rest at
    // -1. The model expects ratio e^0 everywhere (κ = δ = 0), so the anchor is
    // the ends' log ratio.
    let n = 40;
    let frozen = Frozen {
        rho: vec![1.0],
        b: vec![0.0],
        h: 1,
    };
    let theta = DMatrix::from_fn(n, 1, |p, _| p as f32);
    let units: Vec<UnitCounts> = (0..n)
        .map(|p| {
            let end = p < 2 || p >= n - 2;
            let ratio: f32 = if end { 0.7f32.exp() } else { (-1f32).exp() };
            UnitCounts {
                base: vec![(0, 1000.0)],
                displaced: vec![(0, 1000.0 * ratio)],
            }
        })
        .collect();
    let fit = PbFit {
        d: DMatrix::zeros(n, 1),
        kappa: vec![0.0; n],
        delta: vec![0.0],
    };
    let anchor = steady_anchor(&frozen, &theta, &units, &fit, 0..n);
    assert!((anchor[0] - 0.7).abs() < 1e-2, "{anchor:?}");
}

#[test]
fn centering_moves_the_mean_displacement_into_the_gene_ratios_without_changing_a_score() {
    let mut rng = StdRng::seed_from_u64(5);
    let f = frozen(&mut rng);
    let n = Normal::new(0.0f32, 1.0).unwrap();
    let u = 7;
    let mut fit = PbFit {
        d: DMatrix::from_fn(u, H, |_, k| 2.0 + k as f32 + n.sample(&mut rng)),
        kappa: (0..u).map(|_| n.sample(&mut rng)).collect(),
        delta: (0..S).map(|_| n.sample(&mut rng)).collect(),
    };
    let score = |fit: &PbFit, u: usize, g: usize| -> f32 {
        let d: Vec<f32> = fit.d.row(u).iter().copied().collect();
        fit.kappa[u] + fit.delta[g] + dot(&d, f.rho_row(g))
    };
    let before: Vec<f32> = (0..u)
        .flat_map(|p| (0..S).map(move |g| (p, g)))
        .map(|(p, g)| score(&fit, p, g))
        .collect();
    center_units(&mut fit, &f);
    for k in 0..H {
        assert!(fit.d.column(k).sum().abs() < 1e-4, "d column {k}");
    }
    let after: Vec<f32> = (0..u)
        .flat_map(|p| (0..S).map(move |g| (p, g)))
        .map(|(p, g)| score(&fit, p, g))
        .collect();
    for (x, y) in before.iter().zip(&after) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

#[test]
fn the_cell_encoder_follows_its_pseudobulk() {
    cell_encoder_follows_its_pseudobulk(&Device::Cpu);
}

/// The same on the GPU, whose matmul refuses strided operands the CPU takes.
#[cfg(feature = "cuda")]
#[test]
fn the_cell_encoder_runs_on_cuda() {
    cell_encoder_follows_its_pseudobulk(&Device::new_cuda(0).expect("a CUDA device"));
}

fn cell_encoder_follows_its_pseudobulk(dev: &Device) {
    // Cells: sparse reads, each cell's target its group's true displacement.
    let w = world(400, 300, 120, 1.0, 11);
    let mut target = DMatrix::<f32>::zeros(w.units.len(), H);
    for (u, &g) in w.group.iter().enumerate() {
        let sign = if g == 0 { 1.0 } else { -1.0 };
        for k in 0..H {
            target[(u, k)] = sign * w.v[k];
        }
    }
    let mut k = knobs();
    k.distill_epochs = 30;
    k.refine_epochs = 10;
    let delta = pooled_log_ratio(&w.units, S);
    let fit = fit_cells(&w.frozen, &w.units, &target, &delta, &k, 3, dev).unwrap();
    let means = group_means(&fit.d, &w.group);
    let neg_v = [-w.v[0], -w.v[1], -w.v[2]];
    assert!(cosine(&means[0], &w.v) > 0.8, "{:?}", means[0]);
    assert!(cosine(&means[1], &neg_v) > 0.8, "{:?}", means[1]);
}

#[test]
fn groups_sum_their_members_counts() {
    let rows = vec![
        UnitCounts {
            base: vec![(0, 1.0), (2, 2.0)],
            displaced: vec![(1, 1.0)],
        },
        UnitCounts {
            base: vec![(2, 3.0)],
            displaced: vec![(1, 2.0), (3, 1.0)],
        },
    ];
    let out = sum_groups(&rows, &[vec![0, 1], vec![]]);
    assert_eq!(out[0].base, vec![(0, 1.0), (2, 5.0)]);
    assert_eq!(out[0].displaced, vec![(1, 3.0), (3, 1.0)]);
    assert!(out[1].is_empty());
}

/// Five feature rows on backend rows `3, 5, 7, 9, 11`: three genes' base rows
/// and two displaced rows, interleaved.
fn split_fixture() -> UnifiedData {
    let names: Vec<Box<str>> = ["GENE1", "GENE2", "GENE1_d", "GENE3", "GENE2_d"]
        .iter()
        .map(|&n| n.into())
        .collect();
    let counts = DMatrix::<f32>::from_fn(5, 2, |r, c| (r + c + 1) as f32);
    UnifiedData::from_pseudobulks(&counts, names, vec![3, 5, 7, 9, 11]).unwrap()
}

#[test]
fn split_displaced_maps_to_backend_rows_and_cuts_to_the_base_rows() {
    let mut unified = split_fixture();
    let axis = split_displaced(
        &mut unified,
        &[0, 1, 3],
        &[Some(2), Some(4), None],
        "count/displaced",
    )
    .unwrap();
    assert_eq!(axis.base_backend_row, vec![3, 5, 9]);
    assert_eq!(axis.displaced_backend_row, vec![7, 11, u32::MAX]);
    assert_eq!(axis.support(), vec![0, 1], "GENE3 has no displaced row");
    assert_eq!(&*axis.track_name, "count/displaced");
    let names: Vec<&str> = unified.feature_names.iter().map(|n| &**n).collect();
    assert_eq!(names, ["GENE1", "GENE2", "GENE3"]);
    assert_eq!(unified.feature_to_backend_row, vec![3, 5, 9]);
    assert_eq!(unified.n_features(), 3);
}

#[test]
fn split_displaced_refuses_rows_it_cannot_place() {
    let refuses = |base: &[usize], displaced: &[Option<usize>]| -> bool {
        split_displaced(&mut split_fixture(), base, displaced, "count/displaced").is_err()
    };
    assert!(refuses(&[0, 1], &[Some(2)]), "lengths differ");
    assert!(refuses(&[1, 0], &[None, None]), "base rows not ascending");
    assert!(refuses(&[0, 0], &[None, None]), "a base row twice");
    assert!(refuses(&[0, 5], &[None, None]), "base row out of range");
    assert!(
        refuses(&[0, 1], &[Some(5), None]),
        "displaced row out of range"
    );
    assert!(
        refuses(&[0, 1], &[Some(1), None]),
        "displaced row is a base row"
    );
    assert!(
        refuses(&[0, 1], &[Some(2), Some(2)]),
        "two genes share a displaced row"
    );
    assert!(
        !refuses(&[0, 1, 3], &[None, None, None]),
        "no displaced row at all"
    );
}
