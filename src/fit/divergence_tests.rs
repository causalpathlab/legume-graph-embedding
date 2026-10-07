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

/// `n_draws` reads among the support at position `pos` with biases `b`.
fn draw(f: &Frozen, pos: &[f32], b: &[f32], n_draws: usize, rng: &mut StdRng) -> Vec<(u32, f32)> {
    let w: Vec<f64> = (0..S)
        .map(|g| {
            let s: f32 = (0..H).map(|k| pos[k] * f.rho[g * H + k]).sum::<f32>() + b[g];
            f64::from(s).exp()
        })
        .collect();
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

/// Two groups of units, displaced `+v` and `−v`; the displaced track's biases
/// shifted from the base's. Returns θ, the counts, the group of each unit.
struct World {
    frozen: Frozen,
    theta: DMatrix<f32>,
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
    let b_disp: Vec<f32> = frozen
        .b
        .iter()
        .map(|&b| b + 0.3 * n.sample(&mut rng))
        .collect();
    let mut theta = DMatrix::<f32>::zeros(n_units, H);
    let mut units = Vec::new();
    let mut group = Vec::new();
    for u in 0..n_units {
        let gr = u % 2;
        let sign = if gr == 0 { 1.0 } else { -1.0 };
        let th: Vec<f32> = (0..H).map(|_| 0.8 * n.sample(&mut rng)).collect();
        let disp: Vec<f32> = (0..H).map(|k| th[k] + sign * v[k]).collect();
        for k in 0..H {
            theta[(u, k)] = th[k];
        }
        units.push(UnitCounts {
            base: draw(&frozen, &th, &frozen.b, base_reads, &mut rng),
            displaced: draw(&frozen, &disp, &b_disp, displaced_reads, &mut rng),
        });
        group.push(gr);
    }
    World {
        frozen,
        theta,
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
    let dot: f32 = (0..H).map(|k| a[k] * b[k]).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn norm(a: &[f32; H]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn knobs(rule_a_prob: f64) -> DisplacedTrackConfig {
    let mut k = DisplacedTrackConfig::new(DisplacedAxis {
        base_backend_row: Vec::new(),
        displaced_backend_row: Vec::new(),
        track_name: "count/unspliced".into(),
    });
    k.rule_a_prob = rule_a_prob;
    k.pb_epochs = 150;
    k.learning_rate = 0.05;
    k.l2_pb = 0.1;
    k
}

#[test]
fn pseudobulk_displacements_are_recovered_by_either_rule_or_both() {
    let w = world(60, 3000, 1500, 1.0, 7);
    let neg_v = [-w.v[0], -w.v[1], -w.v[2]];
    for p in [0.5, 1.0, 0.0] {
        let fit =
            fit_pseudobulks(&w.frozen, &w.theta, &w.units, &knobs(p), 1, &Device::Cpu).unwrap();
        let means = group_means(&fit.d, &w.group);
        assert!(
            cosine(&means[0], &w.v) > 0.9,
            "rule A prob {p}: {:?} vs {:?}",
            means[0],
            w.v
        );
        assert!(
            cosine(&means[1], &neg_v) > 0.9,
            "rule A prob {p}: {:?}",
            means[1]
        );
    }
}

#[test]
fn no_displacement_gives_small_displacements() {
    let w = world(60, 3000, 1500, 0.0, 9);
    let fit = fit_pseudobulks(&w.frozen, &w.theta, &w.units, &knobs(0.5), 1, &Device::Cpu).unwrap();
    let means = group_means(&fit.d, &w.group);
    let shifted = world(60, 3000, 1500, 1.0, 9);
    for m in &means {
        assert!(norm(m) < 0.25 * norm(&shifted.v), "{m:?}");
    }
}

#[test]
fn the_cell_encoder_follows_its_pseudobulk_and_predicts_held_out_cells() {
    // Cells: sparse reads, each cell's target its group's true displacement.
    let w = world(400, 300, 120, 1.0, 11);
    let mut target = DMatrix::<f32>::zeros(w.units.len(), H);
    for (u, &g) in w.group.iter().enumerate() {
        let sign = if g == 0 { 1.0 } else { -1.0 };
        for k in 0..H {
            target[(u, k)] = sign * w.v[k];
        }
    }
    let mut k = knobs(0.5);
    k.distill_epochs = 30;
    k.refine_epochs = 10;
    let b_disp = w.frozen.b.clone();
    let fit = fit_cells(
        &w.frozen,
        &w.theta,
        &w.units,
        &target,
        &b_disp,
        &k,
        3,
        &Device::Cpu,
    )
    .unwrap();
    let means = group_means(&fit.d, &w.group);
    let neg_v = [-w.v[0], -w.v[1], -w.v[2]];
    assert!(cosine(&means[0], &w.v) > 0.8, "{:?}", means[0]);
    assert!(cosine(&means[1], &neg_v) > 0.8, "{:?}", means[1]);
    assert!(fit.held_out.n_cells > 0);
    assert!(fit.held_out.gain_per_count > 0.0, "{:?}", fit.held_out);
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
