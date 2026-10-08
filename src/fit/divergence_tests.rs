use super::*;
use rand::RngExt;
use rand_distr::{Binomial, Distribution, Normal};

const H: usize = 3;
const S: usize = 40;

fn frozen(rng: &mut StdRng) -> Frozen {
    let n = Normal::new(0.0f32, 0.7).unwrap();
    Frozen {
        rho: DMatrix::from_fn(S, H, |_, _| n.sample(rng)),
        b: (0..S).map(|_| n.sample(rng)).collect(),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn knobs() -> DivergenceConfig {
    DivergenceConfig {
        axis: DivergenceAxis {
            base_backend_row: Vec::new(),
            divergent_backend_row: Vec::new(),
            track_name: "count/divergent".into(),
        },
        l2: 1.0,
        epochs: 80,
        learning_rate: 0.05,
    }
}

fn on(m: &DMatrix<f32>, dev: &Device) -> Tensor {
    to_device(m, dev).unwrap()
}

/// Cells with planted states, gene ratios and directions: every (cell, gene)
/// carries `reads` reads, `Binom(reads, σ(κ_c + δ_g + ⟨θ_c, η_g⟩))` of them
/// on the divergent track.
struct World {
    theta: DMatrix<f32>,
    eta: DMatrix<f32>,
    cells: Vec<CellCounts>,
}

fn world(n_cells: usize, reads: u64, seed: u64) -> World {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = Normal::new(0.0f32, 1.0).unwrap();
    let theta = DMatrix::from_fn(n_cells, H, |_, _| n.sample(&mut rng));
    let eta = DMatrix::from_fn(S, H, |_, _| 0.7 * n.sample(&mut rng));
    let delta: Vec<f32> = (0..S).map(|_| -0.5 + 0.3 * n.sample(&mut rng)).collect();
    let cells = (0..n_cells)
        .map(|c| {
            let kappa = 0.3 * n.sample(&mut rng);
            let mut cell = CellCounts::default();
            let th: Vec<f32> = theta.row(c).iter().copied().collect();
            for (g, d) in delta.iter().enumerate() {
                let et: Vec<f32> = eta.row(g).iter().copied().collect();
                let z = kappa + d + dot(&th, &et);
                let p = 1.0 / (1.0 + (-f64::from(z)).exp());
                let u = Binomial::new(reads, p).unwrap().sample(&mut rng) as f32;
                let s = reads as f32 - u;
                if s > 0.0 {
                    cell.base.push((g as u32, s));
                }
                if u > 0.0 {
                    cell.divergent.push((g as u32, u));
                }
            }
            cell
        })
        .collect();
    World { theta, eta, cells }
}

fn correlation(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let (ma, mb) = (a.mean(), b.mean());
    let (mut num, mut da, mut db) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
    }
    num / (da * db).sqrt().max(1e-12)
}

#[test]
fn the_genes_directions_are_recovered() {
    let w = world(600, 30, 7);
    let fit = fit_ratio(&on(&w.theta, &Device::Cpu), &w.cells, S, &knobs(), 1).unwrap();
    assert_eq!(fit.eta.shape(), (S, H));
    let r = correlation(&fit.eta, &w.eta);
    assert!(r > 0.95, "η recovered with correlation {r}");
    let mean_kappa = fit.kappa.iter().sum::<f32>() / fit.kappa.len() as f32;
    assert!(mean_kappa.abs() < 1e-4, "κ is centred: {mean_kappa}");
}

#[test]
fn profiled_kappa_zeroes_each_cells_own_gradient() {
    let mut rng = StdRng::seed_from_u64(3);
    let (m, s) = (5, 12);
    let a: Vec<f32> = (0..m * s)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let xb: Vec<f32> = (0..m * s).map(|_| rng.random_range(0..6) as f32).collect();
    let xd: Vec<f32> = (0..m * s).map(|_| rng.random_range(0..4) as f32).collect();
    let dev = Device::Cpu;
    let t = |v: &[f32]| Tensor::from_slice(v, (m, s), &dev).unwrap();
    let k0 = Tensor::zeros(m, DType::F32, &dev).unwrap();
    let k = profile_kappa(&t(&a), &t(&xb), &t(&xd), &k0, 30).unwrap();
    let k: Vec<f32> = k.to_vec1().unwrap();
    for (c, kc) in k.iter().enumerate() {
        let grad: f32 = (0..s)
            .map(|g| {
                let i = c * s + g;
                let p = 1.0 / (1.0 + (-(a[i] + kc)).exp());
                (xb[i] + xd[i]) * p - xd[i]
            })
            .sum();
        assert!(grad.abs() < 1e-3, "cell {c}: gradient {grad}");
    }
}

#[test]
fn the_anchor_is_the_observed_ratio_at_both_ends_of_a_genes_score() {
    // One dimension, one gene: θ_c = c, ρ = 1, so the base score is c. With
    // 40 cells the ends are cells {0, 1} and {38, 39}. The ends hold 10 base
    // and 30 divergent reads in all; the middle cells hold the opposite. With
    // κ = δ = 0 the anchor is ln((30 + ½)/(10 + ½)).
    let n = 40;
    let frozen = Frozen {
        rho: DMatrix::from_element(1, 1, 1.0),
        b: vec![0.0],
    };
    let theta = DMatrix::from_fn(n, 1, |c, _| c as f32);
    let cells: Vec<CellCounts> = (0..n)
        .map(|c| {
            let end = c < 2 || c >= n - 2;
            let (s, u) = if end { (2.5, 7.5) } else { (7.5, 2.5) };
            CellCounts {
                base: vec![(0, s)],
                divergent: vec![(0, u)],
            }
        })
        .collect();
    let fit = RatioFit {
        kappa: vec![0.0; n],
        delta: vec![0.0],
        eta: DMatrix::zeros(1, 1),
    };
    let anchor = steady_anchor(&on(&theta, &Device::Cpu), &frozen, &fit, &cells).unwrap();
    let want = (30.5f32 / 10.5).ln();
    assert!((anchor[0] - want).abs() < 1e-5, "{anchor:?} vs {want}");
}

#[test]
fn the_velocity_solves_its_weighted_least_squares() {
    // Targets made exactly by a planted velocity per cell are recovered.
    let mut rng = StdRng::seed_from_u64(11);
    let f = frozen(&mut rng);
    let n = Normal::new(0.0f32, 1.0).unwrap();
    let m = 4;
    let dev = Device::Cpu;
    let th = DMatrix::from_fn(m, H, |_, _| n.sample(&mut rng));
    let v = DMatrix::from_fn(m, H, |_, _| n.sample(&mut rng));
    let rho = &f.rho;
    let mut r = DMatrix::<f32>::zeros(m, S);
    for c in 0..m {
        let scores: Vec<f32> = (0..S)
            .map(|g| (0..H).map(|k| th[(c, k)] * rho[(g, k)]).sum::<f32>() + f.b[g])
            .collect();
        let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
        let z: f32 = scores.iter().map(|x| (x - mx).exp()).sum();
        let pi: Vec<f32> = scores.iter().map(|x| (x - mx).exp() / z).collect();
        let rho_bar: Vec<f32> = (0..H)
            .map(|k| (0..S).map(|g| pi[g] * rho[(g, k)]).sum())
            .collect();
        for g in 0..S {
            r[(c, g)] = (0..H).map(|k| v[(c, k)] * (rho[(g, k)] - rho_bar[k])).sum();
        }
    }
    let rho_d = on(rho, &dev);
    let got = solve_velocity(
        &on(&th, &dev),
        &on(&r, &dev),
        &rho_d,
        &rho_d.t().unwrap().contiguous().unwrap(),
        &Tensor::from_slice(&f.b, (1, S), &dev).unwrap(),
    )
    .unwrap();
    for (x, y) in got.iter().zip(v.iter()) {
        assert!((x - y).abs() < 1e-2, "{got} vs {v}");
    }
}

/// The fit and the velocity on the GPU, whose matmul refuses strided
/// operands the CPU takes.
#[cfg(feature = "cuda")]
#[test]
fn the_fit_runs_on_cuda() {
    let dev = Device::new_cuda(0).expect("a CUDA device");
    let w = world(200, 20, 5);
    let mut k = knobs();
    k.epochs = 5;
    let theta = on(&w.theta, &dev);
    let fit = fit_ratio(&theta, &w.cells, S, &k, 1).unwrap();
    let mut rng = StdRng::seed_from_u64(2);
    let f = frozen(&mut rng);
    let anchor = steady_anchor(&theta, &f, &fit, &w.cells).unwrap();
    let v = cell_velocity(&theta, &f, &fit, &anchor).unwrap();
    assert_eq!(v.shape(), (200, H));
    assert!(v.iter().all(|x| x.is_finite()));
}

fn split_fixture() -> UnifiedData {
    let names: Vec<Box<str>> = ["GENE1", "GENE2", "GENE1_d", "GENE3", "GENE2_d"]
        .iter()
        .map(|&n| n.into())
        .collect();
    let counts = DMatrix::<f32>::from_fn(5, 2, |r, c| (r + c + 1) as f32);
    UnifiedData::from_pseudobulks(&counts, names, vec![3, 5, 7, 9, 11]).unwrap()
}

#[test]
fn split_divergence_maps_to_backend_rows_and_cuts_to_the_base_rows() {
    let mut unified = split_fixture();
    let axis = split_divergence(
        &mut unified,
        &[0, 1, 3],
        &[Some(2), Some(4), None],
        "count/divergent",
    )
    .unwrap();
    assert_eq!(axis.base_backend_row, vec![3, 5, 9]);
    assert_eq!(axis.divergent_backend_row, vec![7, 11, u32::MAX]);
    assert_eq!(axis.support(), vec![0, 1], "GENE3 has no divergent row");
    assert_eq!(&*axis.track_name, "count/divergent");
    let names: Vec<&str> = unified.feature_names.iter().map(|n| &**n).collect();
    assert_eq!(names, ["GENE1", "GENE2", "GENE3"]);
    assert_eq!(unified.feature_to_backend_row, vec![3, 5, 9]);
    assert_eq!(unified.n_features(), 3);
}

#[test]
fn split_divergence_refuses_rows_it_cannot_place() {
    let refuses = |base: &[usize], divergent: &[Option<usize>]| -> bool {
        split_divergence(&mut split_fixture(), base, divergent, "count/divergent").is_err()
    };
    assert!(refuses(&[0, 1], &[Some(2)]), "lengths differ");
    assert!(refuses(&[1, 0], &[None, None]), "base rows not ascending");
    assert!(refuses(&[0, 0], &[None, None]), "a base row twice");
    assert!(refuses(&[0, 5], &[None, None]), "base row out of range");
    assert!(
        refuses(&[0, 1], &[Some(5), None]),
        "divergent row out of range"
    );
    assert!(
        refuses(&[0, 1], &[Some(1), None]),
        "divergent row is a base row"
    );
    assert!(
        refuses(&[0, 1], &[Some(2), Some(2)]),
        "two genes share a divergent row"
    );
    assert!(
        !refuses(&[0, 1, 3], &[None, None, None]),
        "no divergent row at all"
    );
}
