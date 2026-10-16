//! Invariants of the co-embedding's scores, its temperature calibration, and its
//! placement. The claim the calibration tests pin is that `T` is calibrated and
//! spent in the SAME units: both the calibration and the block pass score through
//! [`ScoreKeys`], and scoring them differently is shape-valid, silently wrong, and
//! invisible to every other test in the crate.

use super::*;
use legume_numeric::candle::candle_core::{DType, Device};

const N: usize = 256;
const H: usize = 8;

/// Deterministic cells, plus features whose norms span three orders of magnitude
/// — the condition that breaks a single shared temperature on raw scores.
fn fixture(d: usize) -> (Tensor, Tensor) {
    let dev = Device::Cpu;
    let cells: Vec<f32> = (0..N * H)
        .map(|i| ((i * 37) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let feats: Vec<f32> = (0..d * H)
        .map(|i| {
            let f = i / H;
            // norms spanning 1e-2 .. 1e1 across features
            let scale = 10f32.powf((f % 4) as f32 - 2.0);
            (((i * 17) % 19) as f32 * 0.1 - 0.9) * scale
        })
        .collect();
    (
        Tensor::from_vec(cells, (N, H), &dev).unwrap(),
        Tensor::from_vec(feats, (d, H), &dev).unwrap(),
    )
}

fn col_sds(t: &Tensor) -> Vec<f32> {
    let (n, b) = t.dims2().unwrap();
    let v = t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
    (0..b)
        .map(|j| {
            let col: Vec<f32> = (0..n).map(|i| v[i][j]).collect();
            let m = col.iter().sum::<f32>() / n as f32;
            (col.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n as f32).sqrt()
        })
        .collect()
}

/// `cells · featsᵀ`, the raw dot scores `[N, D]`.
fn raw_scores(cells: &Tensor, feats: &Tensor) -> Tensor {
    cells
        .matmul(&feats.t().unwrap().contiguous().unwrap())
        .unwrap()
}

fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|x| x.is_finite())
}

/// Every feature column comes out at unit scale regardless of `‖e_f‖`. This is
/// the whole mechanism: it is what lets one `T` mean the same thing for a long
/// gene and a short one.
#[test]
fn every_feature_column_is_rescaled_to_unit_sd() {
    let (cells, feats) = fixture(32);
    let raw = raw_scores(&cells, &feats);

    let raw_sds = col_sds(&raw);
    let spread = raw_sds.iter().cloned().fold(f32::MIN, f32::max)
        / raw_sds.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 100.0,
        "fixture must actually exercise a wide norm spread, got {spread:.1}×"
    );

    for (j, sd) in col_sds(&standardize_columns(&raw).unwrap())
        .iter()
        .enumerate()
    {
        assert!(
            (sd - 1.0).abs() < 1e-3,
            "column {j} rescaled to sd {sd}, want 1.0"
        );
    }
}

/// A zero-embedding feature has a constant score column and zero SD. It must
/// divide by the floor and stay finite rather than emitting `NaN` that would
/// propagate into the written parquet.
#[test]
fn a_constant_column_stays_finite() {
    let dev = Device::Cpu;
    let (cells, _) = fixture(1);
    let zero_feat = Tensor::zeros((2, H), DType::F32, &dev).unwrap();
    let out = standardize_columns(&raw_scores(&cells, &zero_feat)).unwrap();
    assert!(
        all_finite(&out),
        "a zero-embedding feature must not produce NaN/Inf"
    );
}

/// Each cell's offset from the cloud centre, scaled to unit norm, by hand — the units the
/// softmax is meant to score in.
fn centred_unit_rows(cells: &Tensor) -> Tensor {
    let centred = cells
        .broadcast_sub(&cells.mean_keepdim(0).unwrap())
        .unwrap();
    let norm = centred
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap();
    centred.broadcast_div(&norm).unwrap()
}

/// Direction-only scores, by hand: `standardize(ê_cell · e_fᵀ)`, `ê` taken from the centre.
fn direction_scores(cells: &Tensor, feats: &Tensor) -> Tensor {
    standardize_columns(&raw_scores(&centred_unit_rows(cells), feats)).unwrap()
}

/// The end-to-end contract: the `T` returned by [`feature_coembedding`], applied to
/// direction-only scores, actually lands near the requested eff-cells target. This
/// fails if the calibration scores in different units from the ones the softmax is
/// meant to use — on raw cells, a `T` calibrated against the longest cells.
#[test]
fn the_calibrated_temperature_hits_its_eff_target() {
    let target = 40.0;
    let cases = [
        ("fixture", fixture(96)),
        ("heavy-tailed", heavy_tailed_cloud(0)),
    ];
    for (name, (cells, feats)) in cases {
        let (co, t) = feature_coembedding(&cells, &feats, target).unwrap();
        assert_eq!(co.dims2().unwrap(), (feats.dim(0).unwrap(), H));
        let eff = eff_from_scores(&direction_scores(&cells, &feats), f64::from(t)).unwrap();
        assert!(
            eff > target / 1.5 && eff < target * 1.5,
            "{name}: calibrated T={t} gives eff {eff:.1} on direction scores, want within \
             1.5× of {target}"
        );
    }
}

/// The temperature is a property of the cells' directions from the cloud centre: rescaling
/// every cell's offset by its own factor (three orders of magnitude) must not move it. The
/// cells come in `±` pairs with a shared factor, so the centre stays exactly at the origin.
#[test]
fn the_temperature_ignores_cell_norm() {
    let (half, feats) = fixture(32);
    let cells = Tensor::cat(&[&half, &half.neg().unwrap()], 0).unwrap();
    let factors: Vec<f32> = (0..2 * N)
        .map(|i| 10f32.powf(((i % N) % 4) as f32 - 1.5))
        .collect();
    let scaled = cells
        .broadcast_mul(&Tensor::from_vec(factors, (2 * N, 1), &Device::Cpu).unwrap())
        .unwrap();
    let (_, t) = feature_coembedding(&cells, &feats, 40.0).unwrap();
    let (_, t_scaled) = feature_coembedding(&scaled, &feats, 40.0).unwrap();
    assert!(
        (t - t_scaled).abs() < 1e-3 * t,
        "T moved under a per-cell rescale: {t} → {t_scaled}"
    );
}

/// An all-zero cell row has no direction. It must leave the co-embedding finite.
#[test]
fn a_zero_cell_stays_finite() {
    let (cells, feats) = fixture(8);
    let zero = Tensor::zeros((1, H), DType::F32, &Device::Cpu).unwrap();
    let cells = Tensor::cat(&[&cells, &zero], 0).unwrap();
    let (co, _) = feature_coembedding(&cells, &feats, 40.0).unwrap();
    assert!(all_finite(&co));
}

/// Sizes of the synthetic clouds: cells, features, and clusters (or topics).
const N_CELLS: usize = 1500;
const N_FEATS: usize = 120;
const K: usize = 6;

fn to_tensors(cells: Vec<f32>, feats: Vec<f32>) -> (Tensor, Tensor) {
    (
        Tensor::from_vec(cells, (N_CELLS, H), &Device::Cpu).unwrap(),
        Tensor::from_vec(feats, (N_FEATS, H), &Device::Cpu).unwrap(),
    )
}

/// Cells in a few angular clusters with a heavy-tailed (Student-t, 3 df) radius —
/// max/median cell norm ~15, the regime the real runs sit in — and features of
/// mixed norm.
fn heavy_tailed_cloud(seed: u64) -> (Tensor, Tensor) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, StudentT, Uniform};
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1.0).unwrap();
    let unit = |v: Vec<f32>| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    };
    let centres: Vec<Vec<f32>> = (0..K)
        .map(|_| unit((0..H).map(|_| normal.sample(&mut rng)).collect()))
        .collect();
    let radius = StudentT::new(3f32).unwrap();
    let mut cells = Vec::with_capacity(N_CELLS * H);
    for i in 0..N_CELLS {
        let dir = unit(
            centres[i % K]
                .iter()
                .map(|&c| c + 0.2 * normal.sample(&mut rng))
                .collect(),
        );
        let r = radius.sample(&mut rng).abs() + 0.2;
        cells.extend(dir.iter().map(|&x| r * x));
    }
    let scale = Uniform::new(0.1f32, 3.0).unwrap();
    let mut feats = Vec::with_capacity(N_FEATS * H);
    for _ in 0..N_FEATS {
        let s = scale.sample(&mut rng);
        feats.extend((0..H).map(|_| s * normal.sample(&mut rng)));
    }
    to_tensors(cells, feats)
}

fn placement(cells: &Tensor, coembed: &Tensor) -> CoembedPlacement {
    coembed_placement_of_tensors(cells, coembed).unwrap()
}

/// The placement contract on a heavy-tailed cloud. Scoring on raw cells (rebuilt here by
/// hand) piles the softmax onto the longest cells: features land outside the cloud, next to
/// its longest cells — which also checks that the fixture and the diagnostic see the bias.
/// [`feature_coembedding`] must put them inside it, beside cells of ordinary norm.
#[test]
fn heavy_tailed_cells_do_not_pull_features_off_the_manifold() {
    let target = 40.0;
    for seed in 0..3 {
        let (cells, feats) = heavy_tailed_cloud(seed);

        let raw = standardize_columns(&raw_scores(&cells, &feats)).unwrap();
        let t_raw = calibrate_t_for_eff(&raw, target).unwrap();
        let on_raw = placement(&cells, &coembed_from_scores(&raw, &cells, t_raw).unwrap());
        assert!(
            on_raw.beyond_p99 > 0.3 && on_raw.nn_norm_pct > 70.0 && !on_raw.on_manifold(),
            "seed {seed}: the fixture must reproduce the bias on raw-cell scores, got {on_raw}"
        );

        let (co, _) = feature_coembedding(&cells, &feats, target).unwrap();
        let got = placement(&cells, &co);
        assert!(
            got.on_manifold(),
            "seed {seed}: features must sit on the manifold ({}), got {got}",
            CoembedPlacement::contract()
        );
    }
}

/// The same contract when the cloud does not sit at the origin. Scoring each cell's direction
/// from the origin leaves a per-cell term `⟨μ, e_f⟩ / ‖e_cell‖` in the softmax, so placement
/// tracks cell norm again; directions must be taken from the cloud's own centre.
#[test]
fn a_cloud_off_the_origin_stays_on_the_manifold() {
    let target = 40.0;
    let (cells, feats) = heavy_tailed_cloud(0);
    let norms: Vec<f32> = cells
        .sqr()
        .unwrap()
        .sum(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec1()
        .unwrap();
    let median_radius = legume_numeric::matrix::utils::median(&norms);
    for k in [1.0f32, 2.0, 4.0] {
        let mut offset = vec![0f32; H];
        offset[0] = k * median_radius;
        let shifted = cells
            .broadcast_add(&Tensor::from_vec(offset, (1, H), &Device::Cpu).unwrap())
            .unwrap();
        let (co, _) = feature_coembedding(&shifted, &feats, target).unwrap();
        let got = placement(&shifted, &co);
        assert!(
            got.on_manifold(),
            "cloud shifted {k}× its median radius: features must sit on the manifold, got {got}"
        );
    }
}

/// An elongated Gaussian cloud (one axis 6× the others) with no cluster structure.
fn elongated_cloud(seed: u64) -> (Tensor, Tensor) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1.0).unwrap();
    let cells: Vec<f32> = (0..N_CELLS * H)
        .map(|i| normal.sample(&mut rng) * if i % H == 0 { 6.0 } else { 1.0 })
        .collect();
    let scale = Uniform::new(0.1f32, 3.0).unwrap();
    let mut feats = Vec::with_capacity(N_FEATS * H);
    for _ in 0..N_FEATS {
        let s = scale.sample(&mut rng);
        feats.extend((0..H).map(|_| s * normal.sample(&mut rng)));
    }
    to_tensors(cells, feats)
}

/// A co-embedded gene is an average of cells, so it sits somewhat inward of single cells;
/// on an elongated cloud that puts the genes' nearest cells well below the 50th norm
/// percentile without any bias. The contract must judge placement against how the cells'
/// own local averages sit, not against a fixed 50.
#[test]
fn an_elongated_cloud_stays_on_the_manifold() {
    for seed in 0..2 {
        let (cells, feats) = elongated_cloud(seed);
        let (co, _) = feature_coembedding(&cells, &feats, 40.0).unwrap();
        let got = placement(&cells, &co);
        assert!(
            got.on_manifold(),
            "seed {seed}: features must sit on the manifold ({}), got {got}",
            CoembedPlacement::contract()
        );
    }
}

/// SIMBA's fixed-T path at a high temperature collapses every gene to the hub: it must fail.
#[test]
fn a_hub_collapse_is_not_on_the_manifold() {
    let (cells, feats) = heavy_tailed_cloud(0);
    let co = feature_coembedding_fixed_t(&cells, &feats, 1e3).unwrap();
    let got = placement(&cells, &co);
    assert!(!got.on_manifold(), "a hub collapse passed: {got}");
}

/// A topic-mixture cloud, `Z = θ·α` with `θ` on the simplex (Dirichlet) and `α` a `K × H`
/// dictionary — off the origin by construction, as `senna rest` writes it.
fn topic_mixture_cloud(seed: u64) -> (Tensor, Tensor) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::multi::Dirichlet;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1.0).unwrap();
    let alpha: Vec<f32> = (0..K * H).map(|_| normal.sample(&mut rng)).collect();
    let dirichlet = Dirichlet::new(&[0.3f32; K]).unwrap();
    let mut cells = Vec::with_capacity(N_CELLS * H);
    for _ in 0..N_CELLS {
        let theta: Vec<f32> = dirichlet.sample(&mut rng);
        cells.extend((0..H).map(|j| (0..K).map(|k| theta[k] * alpha[k * H + j]).sum::<f32>()));
    }
    let feats: Vec<f32> = (0..N_FEATS * H).map(|_| normal.sample(&mut rng)).collect();
    to_tensors(cells, feats)
}

/// Where the origin sits is a coordinate choice, not a property of the cells: translating
/// a topic-mixture cloud must translate its co-embedding by the same vector and leave `T`
/// alone. Directions taken from the origin break this — the further the cloud sits from it,
/// the more the scores track each cell's norm.
#[test]
fn the_coembedding_moves_with_a_topic_mixture_cloud() {
    for seed in 0..3 {
        let (cells, feats) = topic_mixture_cloud(seed);
        let shift = Tensor::from_vec(vec![3f32; H], (1, H), &Device::Cpu).unwrap();
        let moved = cells.broadcast_add(&shift).unwrap();
        let (co, t) = feature_coembedding(&cells, &feats, 40.0).unwrap();
        let (co_moved, t_moved) = feature_coembedding(&moved, &feats, 40.0).unwrap();
        let gap = co
            .broadcast_add(&shift)
            .unwrap()
            .sub(&co_moved)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (t - t_moved).abs() < 1e-3 * t && gap < 1e-3,
            "seed {seed}: translating the cells by 3·1 moved T {t} → {t_moved} and left the \
             co-embedding off by {gap}"
        );
    }
}

/// The placement diagnostic only reports. Cells it cannot measure (here, all identical)
/// must not fail the write that the co-embedding itself completed.
#[test]
fn an_unmeasurable_placement_does_not_fail_the_write() {
    let dir = std::env::temp_dir().join(format!("coembed-write-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("run");
    let cells = Tensor::ones((40, H), DType::F32, &Device::Cpu).unwrap();
    let (_, feats) = fixture(3);
    let names: Vec<Box<str>> = (0..3).map(|i| format!("g{i}").into()).collect();
    let res = crate::write_feature_coembedding(out.to_str().unwrap(), &cells, &feats, &names, 20.0);
    std::fs::remove_dir_all(&dir).ok();
    assert!(res.is_ok(), "the write failed on the diagnostic: {res:?}");
}

/// A cell exactly at the cloud centre has no direction; its key must be the floor's zero,
/// not 0/0. Integer-valued `±` pairs plus a zero row put the centre exactly at the origin.
#[test]
fn a_cell_at_the_centre_stays_finite() {
    let half: Vec<f32> = (0..64 * H).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let half = Tensor::from_vec(half, (64, H), &Device::Cpu).unwrap();
    let zero = Tensor::zeros((1, H), DType::F32, &Device::Cpu).unwrap();
    let cells = Tensor::cat(&[&half, &half.neg().unwrap(), &zero], 0).unwrap();
    let centre: Vec<f32> = cells.mean(0).unwrap().to_vec1().unwrap();
    assert!(
        centre.iter().all(|&c| c == 0.0),
        "fixture centre {centre:?}"
    );
    let (_, feats) = fixture(8);
    let (co, _) = feature_coembedding(&cells, &feats, 20.0).unwrap();
    assert!(all_finite(&co));
}

/// The diagnostic reads whatever float type the co-embedding was written in.
#[test]
fn placement_reads_non_f32_tensors() {
    let (cells, feats) = heavy_tailed_cloud(0);
    let (co, _) = feature_coembedding(&cells, &feats, 40.0).unwrap();
    let as64 = |t: &Tensor| t.to_dtype(DType::F64).unwrap();
    assert!(coembed_placement_of_tensors(&as64(&cells), &as64(&co)).is_ok());
}

/// The clamp that keeps a big-cluster dataset off the far tail of the curve,
/// where the barycentre collapses toward the global centroid.
#[test]
fn the_eff_target_is_clamped_above_and_below() {
    // One giant cluster: median size 900 must not survive as the target.
    let huge: Vec<usize> = vec![0; 1800];
    assert_eq!(target_eff_from_labels(&huge, 1), MAX_TARGET_EFF);

    // All singletons: median size 1 must be lifted to the floor.
    let singletons: Vec<usize> = (0..64).collect();
    assert_eq!(target_eff_from_labels(&singletons, 64), MIN_TARGET_EFF);
}

//////////////////////////////////////////
// PIP-driven confidence shrinkage (§8) //
//////////////////////////////////////////

#[test]
fn fixed_t_coembedding_uses_raw_scores_and_matches_a_hand_computed_softmax_average() {
    use legume_numeric::candle::candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    let cells: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.5],
        vec![2.0, 2.0],
    ];
    let feats: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, -1.0], vec![0.3, 0.3]];
    let flat = |t: &[Vec<f64>]| -> Vec<f32> { t.iter().flatten().map(|&v| v as f32).collect() };
    let e_cell = Tensor::from_vec(flat(&cells), (4, 2), &dev).unwrap();
    let e_feat = Tensor::from_vec(flat(&feats), (3, 2), &dev).unwrap();
    let t = 0.5;
    let got = feature_coembedding_fixed_t(&e_cell, &e_feat, t)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    for (f, feat) in feats.iter().enumerate() {
        let scores: Vec<f64> = cells
            .iter()
            .map(|c| c.iter().zip(feat).map(|(a, b)| a * b).sum::<f64>() / t)
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        let p: Vec<f64> = scores.iter().map(|s| (s - m).exp() / z).collect();
        for h in 0..2 {
            let want: f64 = cells.iter().zip(&p).map(|(c, w)| c[h] * w).sum();
            assert!(
                (f64::from(got[f][h]) - want).abs() < 1e-5,
                "feature {f} dim {h}: {} vs {want}",
                got[f][h]
            );
        }
    }
    // T → ∞ is the cell centroid; T → 0 is the best-scoring cell.
    let hot = feature_coembedding_fixed_t(&e_cell, &e_feat, 1e6)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let centroid = [0.5f64, 0.875];
    for row in &hot {
        for h in 0..2 {
            assert!((f64::from(row[h]) - centroid[h]).abs() < 1e-4);
        }
    }
    let cold = feature_coembedding_fixed_t(&e_cell, &e_feat, 1e-3)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    // feature 0 = (1, 0) scores cell 3 = (2, 2) highest
    assert!(
        (f64::from(cold[0][0]) - 2.0).abs() < 1e-4 && (f64::from(cold[0][1]) - 2.0).abs() < 1e-4
    );
    // feature 1 = (0, −1) scores cell 0 = (1, 0) highest (0 vs −1, −0.5, −2)
    assert!((f64::from(cold[1][0]) - 1.0).abs() < 1e-4 && f64::from(cold[1][1]).abs() < 1e-4);
}
