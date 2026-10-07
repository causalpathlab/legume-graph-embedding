//! The distilled encoder path: the sub-pseudobulk aggregate, the seeded
//! hold-out, the exact conditional intercept, and that the distillation fits a
//! planted target on held-out pseudobulks.

use super::*;
use legume_numeric::candle::candle_core::Device;

fn aggregate_rows(rows: &[FoldedRow], members: &[usize], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d];
    aggregate_into(rows, members, &mut out);
    out
}

const D: usize = 12;
const H: usize = 3;

fn rows(n: usize) -> Vec<FoldedRow> {
    (0..n)
        .map(|i| {
            let feats: Vec<u32> = (0..D as u32)
                .filter(|g| !(g + i as u32).is_multiple_of(3))
                .collect();
            let counts: Vec<f32> = feats
                .iter()
                .map(|&g| (1 + (g as usize * 7 + i * 3) % 5) as f32)
                .collect();
            FoldedRow::new(feats, counts)
        })
        .collect()
}

#[test]
fn the_aggregate_is_the_mean_of_the_members_folded_counts() {
    let r = rows(5);
    let out = aggregate_rows(&r, &[1, 3, 4], D);
    for (g, &got) in out.iter().enumerate() {
        let want: f32 = [1usize, 3, 4]
            .iter()
            .map(|&i| {
                r[i].feats
                    .iter()
                    .zip(&r[i].counts)
                    .find(|(&f, _)| f as usize == g)
                    .map_or(0.0, |(_, &c)| c)
            })
            .sum::<f32>()
            / 3.0;
        assert!((got - want).abs() < 1e-6, "gene {g}: {got} vs {want}");
    }
    assert_eq!(aggregate_rows(&r, &[], D), vec![0.0; D]);
}

#[test]
fn the_holdout_is_seeded_disjoint_and_a_tenth() {
    let (train, held) = split_holdout(40, 0.1, 7);
    assert_eq!(held.len(), 4);
    assert_eq!(train.len(), 36);
    assert!(train.iter().all(|p| !held.contains(p)));
    assert_eq!(split_holdout(40, 0.1, 7), (train.clone(), held.clone()));
    assert_ne!(split_holdout(40, 0.1, 8).1, held);
    // Too few to hold any out: everything trains.
    let (t, h) = split_holdout(5, 0.1, 1);
    assert!(h.is_empty() && t.len() == 5);
}

/// `c = ln(total) − logsumexp_f(θ·e_f + b_f)` makes the expected total count
/// under the cell's rates equal the observed total, exactly.
#[test]
fn the_intercept_matches_the_total_count() {
    let dev = Device::Cpu;
    let feat: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.05)
        .collect();
    let b: Vec<f32> = (0..D).map(|g| -((g + 1) as f32).ln()).collect();
    let dict = FrozenDict::new(&feat, &b, H, &dev).unwrap();
    let theta = Tensor::from_vec(vec![0.3f32, -0.2, 0.5, -0.4, 0.1, 0.2], (2, H), &dev).unwrap();
    let totals = [37.0f32, 1200.0];
    let c = null_intercept(&dict, &theta, &totals).unwrap();
    let th: Vec<Vec<f32>> = theta.to_vec2().unwrap();
    for n in 0..2 {
        let expected_total: f64 = (0..D)
            .map(|g| {
                let s: f32 = (0..H).map(|k| th[n][k] * feat[g * H + k]).sum::<f32>() + b[g] + c[n];
                f64::from(s).exp()
            })
            .sum();
        assert!(
            ((expected_total - f64::from(totals[n])) / f64::from(totals[n])).abs() < 1e-4,
            "row {n}: {expected_total} vs {}",
            totals[n]
        );
    }
}

/// Planted: every pseudobulk's target row is a fixed linear map of its member
/// cells' mean composition. The distilled encoder must predict held-out
/// pseudobulks far better than the zero predictor.
#[test]
fn the_distillation_fits_a_planted_target_on_held_out_pseudobulks() {
    let dev = Device::Cpu;
    // Enough pseudobulks for several steps per pass at the production budget.
    let n_pb = 600;
    let per_pb = 5;
    let n_cells = n_pb * per_pb;
    // Cells of pseudobulk p share a composition profile that varies with p.
    let mut folded = Vec::with_capacity(n_cells);
    let mut cell_to_pb = Vec::with_capacity(n_cells);
    for p in 0..n_pb {
        for c in 0..per_pb {
            let feats: Vec<u32> = (0..D as u32).collect();
            let counts: Vec<f32> = (0..D)
                .map(|g| {
                    let base = 3.0 + 2.5 * ((p as f32 * 0.37 + g as f32 * 0.9).sin());
                    let jitter = 1.0 + 0.3 * (((c * 5 + g * 3 + p) % 7) as f32 / 7.0 - 0.5);
                    (base * jitter).max(0.0).round()
                })
                .collect();
            folded.push(FoldedRow::new(feats, counts));
            cell_to_pb.push(p);
        }
    }
    // A fixed [H, D] map of the composition.
    let a: Vec<f32> = (0..H * D)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut e_pb = nalgebra::DMatrix::<f32>::zeros(n_pb, H);
    for p in 0..n_pb {
        let members: Vec<usize> = (0..n_cells).filter(|&i| cell_to_pb[i] == p).collect();
        let row = aggregate_rows(&folded, &members, D);
        let z: f32 = row.iter().sum::<f32>().max(1e-6);
        for k in 0..H {
            e_pb[(p, k)] = (0..D).map(|g| a[k * D + g] * row[g] / z).sum::<f32>() * 10.0;
        }
    }
    let feat: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    let b = vec![0f32; D];
    let dict = FrozenDict::new(&feat, &b, H, &dev).unwrap();
    let mean = gene_mean(&folded, D);
    let groups = members_by_pb(&cell_to_pb, &(0..n_cells as u32).collect::<Vec<_>>(), n_pb);
    let levels = vec![DistillTargets {
        e_pb: &e_pb,
        groups: &groups,
    }];
    let (encoder, report) = distill(dict.clone(), &mean, &levels, &folded, 3, &dev).unwrap();
    assert!(report.held_out_mse.is_finite());
    assert!(
        report.held_out_mse < 0.5 * report.held_out_target_var,
        "held-out MSE {} vs target variance {}",
        report.held_out_mse,
        report.held_out_target_var
    );
    assert!(
        report.held_out_cosine > 0.8,
        "cosine {}",
        report.held_out_cosine
    );

    // The likelihood refinement starts from the distilled map and lowers the
    // cells' own per-count NLL — the objective it trains is the one reported.
    let refined = refine(&encoder, &dict, &folded, 1.0, 3, &dev).unwrap();
    assert_eq!(refined.n_cells, n_cells);
    assert!(refined.nll_per_count_before.is_finite() && refined.nll_per_count_after.is_finite());
    assert!(
        refined.nll_per_count_after < refined.nll_per_count_before,
        "NLL/count {} → {}",
        refined.nll_per_count_before,
        refined.nll_per_count_after
    );

    // Saved and reloaded on the same dictionary, the trunk places the same
    // rows at the same points: one estimator, both halves.
    let path = std::env::temp_dir().join(format!(
        "cell_enc_roundtrip_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    encoder.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, H, &path, &dev).unwrap();
    assert_eq!(
        again.feature_mean().unwrap(),
        encoder.feature_mean().unwrap()
    );
    std::fs::remove_file(&path).ok();
    let nodes: Vec<(u32, &[u32], &[f32])> = folded[..7]
        .iter()
        .enumerate()
        .map(|(i, r)| (i as u32, r.feats.as_slice(), r.counts.as_slice()))
        .collect();
    let a = encoder.encode_edges(&nodes).unwrap();
    let b2 = again.encode_edges(&nodes).unwrap();
    assert_eq!(a.theta.len(), 7 * H);
    for (x, y) in a.theta.iter().zip(&b2.theta) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    for (x, y) in a.b_node.iter().zip(&b2.b_node) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    // And the encoded rows are not all alike: the map is informative.
    let first: Vec<f32> = a.theta[..H].to_vec();
    assert!(a.theta[H..]
        .chunks(H)
        .any(|r| r.iter().zip(&first).any(|(p, q)| (p - q).abs() > 1e-3)));

    // The gauge shift folds into the head exactly: every placement moves by
    // −shift and nothing else, and it survives the save/load round trip.
    let shift = vec![0.5f32, -1.25, 2.0];
    encoder.shift_output(&shift).unwrap();
    let shifted = encoder.encode_edges(&nodes).unwrap();
    for (row, orig) in shifted.theta.chunks(H).zip(a.theta.chunks(H)) {
        for k in 0..H {
            assert!(
                (row[k] - (orig[k] - shift[k])).abs() < 1e-5,
                "{} vs {}",
                row[k],
                orig[k] - shift[k]
            );
        }
    }
    encoder.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, H, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();
    let reloaded = again.encode_edges(&nodes).unwrap();
    for (x, y) in reloaded.theta.iter().zip(&shifted.theta) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

//////////////////////////////////
// The phase-2 fixture (parity) //
//////////////////////////////////

/// Owned pieces of the `project_cells` fixture, so the `(id, feats,
/// counts)` tuples can borrow them.
struct Phase2Fixture {
    feat: Vec<f32>,
    b_feat: Vec<f32>,
    edges: Vec<(Vec<u32>, Vec<f32>)>,
    cell_to_pb: Vec<usize>,
    e_pb: nalgebra::DMatrix<f32>,
}

const FIX_PB: usize = 12;
const FIX_PER_PB: usize = 4;
const FIX_CELLS: usize = FIX_PB * FIX_PER_PB;

/// A planted fixture: `FIX_PB` pseudobulks of `FIX_PER_PB` cells each,
/// every cell dense over `D` features, and a pseudobulk table that is a fixed
/// linear map of the members' mean composition.
fn phase2_fixture() -> Phase2Fixture {
    let mut edges = Vec::with_capacity(FIX_CELLS);
    let mut cell_to_pb = Vec::with_capacity(FIX_CELLS);
    for p in 0..FIX_PB {
        for c in 0..FIX_PER_PB {
            let feats: Vec<u32> = (0..D as u32)
                .filter(|g| !(*g as usize + c).is_multiple_of(5))
                .collect();
            let counts: Vec<f32> = feats
                .iter()
                .map(|&g| {
                    let base = 1.5 + 0.9 * ((p as f32 * 0.41 + g as f32 * 0.77).sin());
                    let jitter =
                        1.0 + 0.25 * (((c * 3 + g as usize * 5 + p) % 7) as f32 / 7.0 - 0.5);
                    (base * jitter).max(1.0).round()
                })
                .collect();
            edges.push((feats, counts));
            cell_to_pb.push(p);
        }
    }
    let folded: Vec<FoldedRow> = edges
        .iter()
        .map(|(f, c)| FoldedRow::new(f.clone(), c.clone()))
        .collect();
    let a: Vec<f32> = (0..H * D)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let mut e_pb = nalgebra::DMatrix::<f32>::zeros(FIX_PB, H);
    for p in 0..FIX_PB {
        let members: Vec<usize> = (0..FIX_CELLS).filter(|&i| cell_to_pb[i] == p).collect();
        let row = aggregate_rows(&folded, &members, D);
        let z: f32 = row.iter().sum::<f32>().max(1e-6);
        for k in 0..H {
            e_pb[(p, k)] = (0..D).map(|g| a[k * D + g] * row[g] / z).sum::<f32>();
        }
    }
    Phase2Fixture {
        feat: (0..D * H)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2)
            .collect(),
        b_feat: (0..D).map(|g| -((g + 1) as f32).ln() * 0.1).collect(),
        edges,
        cell_to_pb,
        e_pb,
    }
}

/// `project_cells` on the fixture, returning `(θ, b_cell)`.
fn project_fixture(fx: &Phase2Fixture) -> (Vec<f32>, Vec<f32>) {
    let dev = Device::Cpu;
    let cells: Vec<(u32, &[u32], &[f32])> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &fx.feat,
        b_feat: &fx.b_feat,
        h: H,
        n_cells: FIX_CELLS,
        lambda: 1.0,
        dev: &dev,
        label: "Phase 2",
        gauge_fix: true,
    };
    let levels = vec![DistillLevel {
        e_pb: &fx.e_pb,
        cell_to_pb: &fx.cell_to_pb,
    }];
    let spec = DistillSpec {
        levels: &levels,
        seed: 20_260_914,
    };
    let (out, enc) = project_cells(&input, &cells, None, &spec, None).unwrap();
    // The encoder `senna bge` persists reads the whole axis.
    assert_eq!(enc.feature_mean().unwrap().len(), D);
    (out.theta, out.b_cell)
}

/// The parity guard for the `senna bge` phase-2 path.
///
/// The numbers below were taken from `project_cells` on this same fixture —
/// the encoder's placement with the intercept exact at it, no per-cell solve
/// after it — as the mean of six runs, one process each.
///
/// The bar is 5e-4, not exact: the global gradient-norm clip sums the
/// per-parameter squares in `GradStore`'s hash-map order, which differs between
/// processes, so the clip factor — and through the refinement, the answer — can
/// move run to run *within one build*. The six runs behind the snapshot agreed
/// far inside the bar, and re-seeding the distillation off `spec.seed` moves
/// θ[0] by well over it.
#[test]
fn project_cells_matches_the_previous_output() {
    #[rustfmt::skip]
    const THETA: [f32; FIX_CELLS * H] = [
        0.203109, -0.107541, -0.157314, 0.058433, -0.086704, -0.152246,
        -0.045752, 0.085753, 0.015765, -0.041326, 0.176028, -0.036660,
        0.267939, -0.087062, -0.182925, -0.042972, -0.019994, -0.176514,
        -0.092915, 0.123936, -0.005518, -0.110193, 0.279293, 0.002435,
        0.265573, -0.143911, -0.158012, -0.065789, -0.076130, -0.149596,
        0.020219, 0.033865, -0.057489, -0.110193, 0.279293, 0.002435,
        0.194284, -0.131810, -0.127327, -0.076847, -0.010759, -0.093634,
        -0.038931, 0.031786, -0.009169, -0.077517, 0.295738, 0.015932,
        0.085314, -0.109521, -0.066885, -0.051909, 0.071052, -0.047744,
        -0.032488, 0.015211, 0.018998, -0.070097, 0.266701, 0.016663,
        0.085314, -0.109521, -0.066885, -0.057143, 0.030019, -0.030547,
        -0.063254, -0.002902, 0.119968, -0.030610, 0.238321, 0.028054,
        0.042879, -0.045137, -0.017436, -0.045018, -0.025343, 0.031198,
        -0.089959, -0.099387, 0.170223, -0.023186, 0.219146, 0.025290,
        0.043764, -0.066149, -0.046222, -0.045018, -0.025343, 0.031198,
        -0.092799, -0.135982, 0.186704, -0.068617, 0.196431, 0.028556,
        0.043764, -0.066149, -0.046222, -0.006413, -0.065482, 0.052580,
        -0.022780, -0.145474, 0.204507, -0.042115, 0.042647, 0.124450,
        0.081346, -0.078643, -0.011180, -0.002283, -0.054526, 0.030870,
        -0.022780, -0.145474, 0.204507, -0.014857, -0.005274, 0.159811,
        0.120790, -0.088061, -0.165305, -0.037615, -0.076657, 0.019209,
        -0.003736, -0.126952, 0.181781, -0.005667, -0.014257, 0.058296,
        0.085639, -0.045470, -0.133344, -0.021665, -0.069696, -0.042657,
        -0.040255, -0.105655, 0.193101, -0.005667, -0.014257, 0.058296,
    ];
    #[rustfmt::skip]
    const B_CELL: [f32; FIX_CELLS] = [
        0.375882, 0.442689, 0.501585, 0.364092, 0.433011, 0.492288,
        0.430064, 0.467982, 0.302100, 0.553641, 0.445128, 0.467982,
        0.379358, 0.496944, 0.443856, 0.347053, 0.317504, 0.372328,
        0.507619, 0.352778, 0.317504, 0.440672, 0.383090, 0.363420,
        0.244760, 0.383526, 0.311893, 0.431103, 0.163324, 0.383526,
        0.379372, 0.294358, 0.163324, 0.319245, 0.314993, 0.240584,
        0.166783, 0.244607, 0.314993, 0.318938, 0.154058, 0.316019,
        0.317954, 0.387527, 0.236552, 0.314083, 0.315606, 0.387527,
    ];
    let (theta, b_cell) = project_fixture(&phase2_fixture());
    assert_eq!(theta.len(), THETA.len());
    assert_eq!(b_cell.len(), B_CELL.len());
    for (i, (&got, &want)) in theta.iter().zip(&THETA).enumerate() {
        assert!(
            (got - want).abs() < 5e-4,
            "θ[{i}] drifted from the pre-change output: {got} vs {want}"
        );
    }
    for (i, (&got, &want)) in b_cell.iter().zip(&B_CELL).enumerate() {
        assert!(
            (got - want).abs() < 5e-4,
            "b_cell[{i}] drifted from the pre-change output: {got} vs {want}"
        );
    }
}

/// The refinement step is a dense `[step × D]` block with a backward pass, so
/// it answers to the same activation budget as every other phase-2 block: an
/// ordinary axis keeps the full step, a very wide one (a multiome axis with
/// every peak) shrinks it instead of running the device out of memory.
#[test]
fn refine_step_answers_to_the_block_budget() {
    assert_eq!(refine_cells_per_step(30_000), REFINE_CELLS_PER_STEP);
    let wide = 2_000_000;
    let step = refine_cells_per_step(wide);
    assert!(
        step < REFINE_CELLS_PER_STEP,
        "step {step} on {wide} features"
    );
    assert_eq!(step, block_sgd::block_cells(wide));
    assert!(refine_cells_per_step(usize::MAX / 1024) >= 1);
}

/// Phase 2 on a collapsed axis. Every cell's intercept is the FULL axis'
/// closed form at its `θ` (the collapse is exact for the likelihood), and the
/// saved encoder, which carries the collapse, places the run's own full-axis
/// cells where phase 2 put them: one estimator, both halves.
#[test]
fn phase_2_on_a_collapsed_axis_is_exact_on_the_full_axis() {
    let dev = Device::Cpu;
    let mut fx = phase2_fixture();
    // Rows 6..9 and 9..12 are module-only: each module's rows share one row.
    let module_only: Vec<bool> = (0..D).map(|g| g >= 6).collect();
    let labels: Vec<u32> = (0..D as u32)
        .map(|g| match g {
            0..6 => g,
            6..9 => 6,
            _ => 7,
        })
        .collect();
    for g in 6..D {
        let src = if g < 9 { 6 } else { 9 };
        for k in 0..H {
            fx.feat[g * H + k] = fx.feat[src * H + k];
        }
    }
    let collapse = RowCollapse::from_modules(&module_only, &labels).unwrap();
    assert_eq!(collapse.n_rows, 8);

    let cells: Vec<(u32, &[u32], &[f32])> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &fx.feat,
        b_feat: &fx.b_feat,
        h: H,
        n_cells: FIX_CELLS,
        lambda: 1.0,
        dev: &dev,
        label: "Phase 2",
        gauge_fix: true,
    };
    let levels = vec![DistillLevel {
        e_pb: &fx.e_pb,
        cell_to_pb: &fx.cell_to_pb,
    }];
    let spec = DistillSpec {
        levels: &levels,
        seed: 20_260_922,
    };
    let (out, enc) = project_cells(&input, &cells, None, &spec, Some(&collapse)).unwrap();
    assert_eq!(out.theta.len(), FIX_CELLS * H);
    let tm = &out.gauge.theta_mean;
    let theta = |i: usize| -> Vec<f32> { (0..H).map(|k| out.theta[i * H + k] + tm[k]).collect() };

    for (i, (_, counts)) in fx.edges.iter().enumerate() {
        let th = theta(i);
        let scores: Vec<f64> = (0..D)
            .map(|g| {
                let s: f32 = (0..H).map(|k| th[k] * fx.feat[g * H + k]).sum::<f32>() + fx.b_feat[g];
                f64::from(s)
            })
            .collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + scores.iter().map(|s| (s - mx).exp()).sum::<f64>().ln();
        let n: f64 = counts.iter().map(|&c| f64::from(c)).sum();
        let want = n.ln() - lse;
        assert!(
            (f64::from(out.b_cell[i]) - want).abs() < 1e-4,
            "cell {i}: intercept {} vs the full axis' {want}",
            out.b_cell[i]
        );
    }

    let path = std::env::temp_dir().join(format!(
        "cell_enc_phase2_collapsed_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    enc.save(&path).unwrap();
    let again = CellEncoder::load(&fx.feat, &fx.b_feat, H, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();
    let placed = again.encode_edges(&cells).unwrap();
    for i in 0..FIX_CELLS {
        for (k, want) in theta(i).into_iter().enumerate() {
            let got = placed.theta[i * H + k];
            assert!(
                (got - want).abs() < 1e-5,
                "cell {i} θ[{k}]: {got} vs phase 2's {want}"
            );
        }
    }
}

/// An encoder trained on the collapsed axis, saved with its row map and
/// reloaded against the FULL dictionary, places full-axis cells exactly as the
/// in-memory encoder places the same cells already collapsed.
#[test]
fn a_collapsed_encoder_round_trips_onto_the_full_axis() {
    let dev = Device::Cpu;
    // Full axis: rows 0, 1 residual; rows 2..6 module-only in two modules.
    let module_only = [false, false, true, true, true, true];
    let labels = [0u32, 1, 5, 5, 7, 7];
    let collapse = RowCollapse::from_modules(&module_only, &labels).unwrap();
    let h = 3;
    let mut feat = vec![0f32; 6 * h];
    for (g, row) in feat.chunks_mut(h).enumerate() {
        let src = if module_only[g] {
            labels[g] as usize
        } else {
            g
        };
        for (k, x) in row.iter_mut().enumerate() {
            *x = ((src * 5 + k * 3) % 7) as f32 * 0.1 - 0.3;
        }
    }
    let b = vec![0.2f32, -0.1, -1.0, -0.4, -0.7, -1.3];
    let (rf, rb) = collapse.reduce_dictionary(&feat, &b, h);
    let mean_red = vec![1.0f32; collapse.n_rows];
    let mut enc =
        CellEncoder::build(FrozenDict::new(&rf, &rb, h, &dev).unwrap(), &mean_red, &dev).unwrap();
    enc.collapse = Some(collapse.clone());

    let path = std::env::temp_dir().join(format!(
        "cell_enc_collapsed_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    enc.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, h, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();

    // Full-axis cells through the reloaded encoder …
    let cells: Vec<(Vec<u32>, Vec<f32>)> = vec![
        (vec![0, 2, 3, 5], vec![2.0, 1.0, 4.0, 3.0]),
        (vec![1, 4], vec![5.0, 1.0]),
    ];
    let full: Vec<(u32, &[u32], &[f32])> = cells
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let a = again.encode_edges(&full).unwrap();
    // … match the same cells collapsed by hand through the in-memory encoder.
    let reduced: Vec<(Vec<u32>, Vec<f32>)> = cells
        .iter()
        .map(|(f, c)| collapse.reduce_edges(f, c))
        .collect();
    let red: Vec<(u32, &[u32], &[f32])> = reduced
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    enc.collapse = None;
    let bx = enc.encode_edges(&red).unwrap();
    for (x, y) in a.theta.iter().zip(&bx.theta) {
        assert!((x - y).abs() < 1e-5, "θ {x} vs {y}");
    }
    for (x, y) in a.b_node.iter().zip(&bx.b_node) {
        assert!((x - y).abs() < 1e-5, "intercept {x} vs {y}");
    }
}
