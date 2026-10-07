use super::*;
use crate::data::Triplet;
use crate::fit::hier::params::{PresetGenes, PresetMode};
use crate::fit::hier::partition::{Partition, UnitModules};
use crate::fit::hier::units::UnitTable;
use crate::LoraSpec;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::convert::to_host;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

////////////////////////////////////////////////////////////////////////
// An independent reference: the module docs' formula in f64 loops    //
////////////////////////////////////////////////////////////////////////

struct Host {
    h: usize,
    e_u: Vec<f32>,
    mu: Vec<f32>,
    b_m: Vec<f32>,
    r: Vec<f32>,
    b_g: Vec<f32>,
}

fn host(p: &HierParams) -> Host {
    Host {
        h: p.h,
        e_u: to_host(p.e_u.as_tensor()).unwrap(),
        mu: to_host(p.mu.as_tensor()).unwrap(),
        b_m: to_host(p.b_m.as_tensor()).unwrap(),
        r: to_host(p.r.as_tensor()).unwrap(),
        b_g: to_host(p.b_g.as_tensor()).unwrap(),
    }
}

fn log_softmax_f64(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + scores.iter().map(|s| (s - m).exp()).sum::<f64>().ln();
    scores.iter().map(|s| s - lse).collect()
}

/// `L = Σ_u w [L₁ + Σ_k (c_k/K) L₂]`, written from the formula with no shared
/// code.
fn reference_loss(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    plan: &StepPlan,
) -> f64 {
    let hp = host(p);
    let h = hp.h;
    let n_m = part.n_modules();
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    };
    let mu_row = |m: usize| &hp.mu[m * h..(m + 1) * h];
    let r_row = |g: usize| &hp.r[g * h..(g + 1) * h];
    let mut loss = 0f64;
    for &u in &plan.units {
        let u = u as usize;
        let w = f64::from(units.weight[u]);
        let e = &hp.e_u[u * h..(u + 1) * h];
        let scores: Vec<f64> = (0..n_m)
            .map(|m| dot(e, mu_row(m)) + f64::from(hp.b_m[m]))
            .collect();
        let logp = log_softmax_f64(&scores);
        for (m, lp) in logp.iter().enumerate() {
            let q = f64::from(um.q[um.idx(u, m)]);
            loss -= w * q * lp;
        }
    }
    for (m, pairs) in &plan.pairs_by_module {
        let m = *m as usize;
        let genes: Vec<usize> = part.members[m].iter().map(|&g| g as usize).collect();
        for &(u, wt) in pairs {
            let u = u as usize;
            let scale = f64::from(units.weight[u]) * f64::from(wt);
            let e = &hp.e_u[u * h..(u + 1) * h];
            let scores: Vec<f64> = genes
                .iter()
                .map(|&g| dot(e, r_row(g)) + f64::from(hp.b_g[g]))
                .collect();
            let logp = log_softmax_f64(&scores);
            let n_um = f64::from(um.n_um[um.idx(u, m)]);
            let counts = um.by_module[u]
                .iter()
                .find(|(k, _)| *k as usize == m)
                .map(|(_, v)| v.as_slice())
                .unwrap_or(&[]);
            for &(slot, c) in counts {
                loss -= scale * f64::from(c) / n_um * logp[slot as usize];
            }
        }
    }
    loss
}

//////////////
// Fixtures //
//////////////

/// Three units, two modules, six genes, H = 2 — every quantity small enough
/// to difference numerically.
fn fixture() -> (UnitTable, Partition, UnitModules, HierParams) {
    let l0 = vec![
        t(0, 0, 4.0),
        t(0, 1, 1.0),
        t(0, 3, 2.0),
        t(1, 2, 3.0),
        t(1, 4, 5.0),
        t(1, 5, 1.0),
        t(2, 0, 1.0),
        t(2, 2, 1.0),
        t(2, 3, 6.0),
        t(2, 5, 2.0),
    ];
    let units = UnitTable::from_pseudobulks_and_cells(&[&l0], &[3], &[], None, 6);
    let part = Partition::from_labels(&[0, 0, 1, 0, 1, 1], 2);
    let um = UnitModules::new(&units, &part);
    let params = HierParams::new(3, 2, 6, 2, 11, &Device::Cpu).unwrap();
    (units, part, um, params)
}

fn plan_all() -> StepPlan {
    StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![(0, vec![(0, 1.0), (2, 1.0)]), (1, vec![(1, 1.0), (2, 1.0)])],
    }
}

fn total(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    plan: &StepPlan,
) -> (f64, Tensor) {
    let ctx = StepCtx {
        units,
        um,
        part,
        skip_module: &[],
    };
    let (s, loss) = step_loss(p, &ctx, plan, 0.0, None).unwrap();
    (s.loss_module + s.loss_gene + s.loss_ridge, loss)
}

/// Central difference of the loss in one entry of `var`.
fn finite_difference(var: &Var, flat: usize, eps: f32, loss_at: &dyn Fn() -> f64) -> f64 {
    let dims = var.dims().to_vec();
    let base = var
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let bump = |d: f32| -> f64 {
        let mut v = base.clone();
        v[flat] += d;
        var.set(&Tensor::from_vec(v, dims.as_slice(), &Device::Cpu).unwrap())
            .unwrap();
        loss_at()
    };
    let plus = bump(eps);
    let minus = bump(-eps);
    bump(0.0);
    (plus - minus) / (2.0 * f64::from(eps))
}

fn grad_of(grads: &GradStore, v: &Var) -> Vec<f32> {
    match grads.get(v) {
        Some(g) => g.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        None => vec![0.0; v.dims().iter().product()],
    }
}

///////////
// Tests //
///////////

#[test]
fn the_step_loss_matches_the_f64_reference() {
    let (units, part, um, p) = fixture();
    let plan = plan_all();
    let (got, _) = total(&p, &units, &um, &part, &plan);
    let want = reference_loss(&p, &units, &um, &part, &plan);
    assert!(
        (got - want).abs() < 1e-4 * (1.0 + want.abs()),
        "{got} vs {want}"
    );
}

/// Autograd against central differences of the same loss, on every table.
#[test]
fn autograd_matches_finite_differences() {
    let (units, part, um, p) = fixture();
    // Biases off zero, so their gradient is not read at a symmetric point.
    p.b_m
        .set(&Tensor::from_vec(vec![0.3f32, -0.2], 2, &Device::Cpu).unwrap())
        .unwrap();
    p.b_g
        .set(&Tensor::from_vec(vec![0.1f32, -0.1, 0.2, 0.0, -0.3, 0.15], 6, &Device::Cpu).unwrap())
        .unwrap();
    let plan = plan_all();
    let (_, loss) = total(&p, &units, &um, &part, &plan);
    let grads = loss.backward().unwrap();
    let loss_at = || total(&p, &units, &um, &part, &plan).0;
    let checks: Vec<(&str, &Var, usize)> = vec![
        ("e_u", &p.e_u, 3),
        ("mu", &p.mu, 2),
        ("b_m", &p.b_m, 1),
        ("r", &p.r, 7),
        ("b_g", &p.b_g, 5),
    ];
    for (name, var, flat) in checks {
        let analytic = f64::from(grad_of(&grads, var)[flat]);
        let numeric = finite_difference(var, flat, 1e-3, &loss_at);
        assert!(
            (analytic - numeric).abs() < 2e-3 * (1.0 + analytic.abs()),
            "{name}[{flat}]: autograd {analytic} vs finite difference {numeric}"
        );
    }
}

#[test]
fn pair_weight_scales_the_gene_level_term() {
    let (units, part, um, p) = fixture();
    let one = plan_all();
    let mut half = plan_all();
    for (_, pairs) in &mut half.pairs_by_module {
        for pr in pairs.iter_mut() {
            pr.1 = 0.5;
        }
    }
    let (a, _) = step_loss(
        &p,
        &StepCtx {
            units: &units,
            um: &um,
            part: &part,
            skip_module: &[],
        },
        &one,
        0.0,
        None,
    )
    .unwrap();
    let (b, _) = step_loss(
        &p,
        &StepCtx {
            units: &units,
            um: &um,
            part: &part,
            skip_module: &[],
        },
        &half,
        0.0,
        None,
    )
    .unwrap();
    assert!((b.loss_gene - 0.5 * a.loss_gene).abs() < 1e-5);
    assert!((b.loss_module - a.loss_module).abs() < 1e-6);
}

#[test]
fn twenty_steps_on_the_full_plan_lower_the_loss() {
    let (units, part, um, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers::new(&p, 0.2).unwrap();
    let before = total(&p, &units, &um, &part, &plan).0;
    for _ in 0..20 {
        let (_, loss) = total(&p, &units, &um, &part, &plan);
        let grads = loss.backward().unwrap();
        apply(&mut p, &mut opt, &grads, 0.2, 0.0, 0.0).unwrap();
    }
    let after = total(&p, &units, &um, &part, &plan).0;
    assert!(after < before * 0.9, "{before} → {after}");
}

/// Decay reaches the rows a step touched and no bias; a row the step never
/// scored keeps its value.
#[test]
fn weight_decay_shrinks_touched_rows_only() {
    let (units, part, um, mut p) = fixture();
    // Units 0 and 2, module 0 alone at the gene level: module 1's genes
    // {2, 4, 5} and unit 1 are untouched.
    let plan = StepPlan {
        units: vec![0, 2],
        pairs_by_module: vec![(0, vec![(0, 1.0), (2, 1.0)])],
    };
    // A negligible optimizer rate, so the decay is all that moves a row; the
    // decay factor itself comes from the rate handed to `apply`.
    let mut opt = Optimizers::new(&p, 1e-7).unwrap();
    let r0 = to_host(p.r.as_tensor()).unwrap();
    let b0 = to_host(p.b_g.as_tensor()).unwrap();
    let e0 = to_host(p.e_u.as_tensor()).unwrap();
    let (_, loss) = total(&p, &units, &um, &part, &plan);
    let grads = loss.backward().unwrap();
    apply(&mut p, &mut opt, &grads, 0.5, 0.2, 0.2).unwrap();
    let h = p.h;
    let r1 = to_host(p.r.as_tensor()).unwrap();
    for g in [0usize, 1, 3] {
        for k in 0..h {
            assert!(
                (r1[g * h + k] - 0.9 * r0[g * h + k]).abs() < 1e-5,
                "touched row {g} decays"
            );
        }
    }
    for g in [2usize, 4, 5] {
        for k in 0..h {
            assert_eq!(
                r1[g * h + k],
                r0[g * h + k],
                "untouched row {g} keeps its value"
            );
        }
    }
    for (a, b) in to_host(p.b_g.as_tensor()).unwrap().iter().zip(&b0) {
        assert!((a - b).abs() < 1e-5, "biases never decay");
    }
    let e1 = to_host(p.e_u.as_tensor()).unwrap();
    for k in 0..h {
        assert!((e1[k] - 0.9 * e0[k]).abs() < 1e-5);
        assert_eq!(e1[h + k], e0[h + k], "unit 1 was not in the plan");
    }
}

/// The units' decay is its own: at zero a touched unit row keeps its value
/// while the feature rows the same step touched still decay.
#[test]
fn unit_rows_can_be_spared_the_feature_decay() {
    let (units, part, um, mut p) = fixture();
    let plan = StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![(0, vec![(0, 1.0), (2, 1.0)])],
    };
    let mut opt = Optimizers::new(&p, 1e-7).unwrap();
    let r0 = to_host(p.r.as_tensor()).unwrap();
    let e0 = to_host(p.e_u.as_tensor()).unwrap();
    let (_, loss) = total(&p, &units, &um, &part, &plan);
    let grads = loss.backward().unwrap();
    apply(&mut p, &mut opt, &grads, 0.5, 0.2, 0.0).unwrap();
    let h = p.h;
    let e1 = to_host(p.e_u.as_tensor()).unwrap();
    for (a, b) in e1[..3 * h].iter().zip(&e0[..3 * h]) {
        assert!(
            (a - b).abs() < 1e-5,
            "a touched unit row decayed: {b} → {a}"
        );
    }
    let r1 = to_host(p.r.as_tensor()).unwrap();
    for k in 0..h {
        assert!(
            (r1[k] - 0.9 * r0[k]).abs() < 1e-5,
            "the touched gene row 0 still decays"
        );
    }
}

/// A pinned row takes no step, its bias does, and only the modules holding a
/// pinned row stay put.
#[test]
fn pinned_rows_hold_while_their_biases_train() {
    let (units, part, um, mut p) = fixture();
    let given = PresetGenes {
        ids: vec![0, 3],
        rows: vec![0.5, -0.5, 0.25, 0.75],
        mode: PresetMode::Freeze,
    };
    p.preset(&given, &part.module_of, &[]).unwrap();
    let plan = plan_all();
    let mut opt = Optimizers::new(&p, 0.2).unwrap();
    let r0 = to_host(p.r.as_tensor()).unwrap();
    let mu0 = to_host(p.mu.as_tensor()).unwrap();
    let b0 = to_host(p.b_g.as_tensor()).unwrap();
    for _ in 0..5 {
        let (_, loss) = total(&p, &units, &um, &part, &plan);
        let grads = loss.backward().unwrap();
        apply(&mut p, &mut opt, &grads, 0.2, 0.01, 0.01).unwrap();
    }
    let h = p.h;
    let r1 = to_host(p.r.as_tensor()).unwrap();
    for g in [0usize, 3] {
        assert_eq!(
            &r1[g * h..(g + 1) * h],
            &r0[g * h..(g + 1) * h],
            "pinned row {g} moved"
        );
    }
    assert_ne!(&r1[2 * h..3 * h], &r0[2 * h..3 * h], "a free row trains");
    // μ is held exactly where a given row sits, and trains where none does.
    let mu1 = to_host(p.mu.as_tensor()).unwrap();
    for (m, &pinned) in p.mu_pinned.iter().enumerate() {
        let (was, is) = (&mu0[m * h..(m + 1) * h], &mu1[m * h..(m + 1) * h]);
        if pinned {
            assert_eq!(is, was, "module {m} holds a given row and moved");
        } else {
            assert_ne!(is, was, "module {m} has no given row and never trained");
        }
    }
    assert!(p.mu_pinned.iter().any(|&x| x) && p.mu_pinned.iter().any(|&x| !x));
    let b1 = to_host(p.b_g.as_tensor()).unwrap();
    assert!(
        b1[0] != b0[0] || b1[3] != b0[3],
        "a pinned gene's bias still trains"
    );
}

/// Autograd against central differences on the four LoRA factors, with the
/// shared factors moved off zero so the row factors see a gradient too.
#[test]
fn autograd_matches_finite_differences_on_the_lora_factors() {
    let (units, part, um, mut p) = fixture();
    let given = PresetGenes {
        ids: vec![0, 3, 4],
        rows: vec![0.5, -0.5, 0.25, 0.75, -0.3, 0.1],
        mode: PresetMode::Lora(LoraSpec {
            rank: 1,
            lr_ratio: 1.0,
            ridge: 0.0,
        }),
    };
    p.preset(&given, &part.module_of, &[]).unwrap();
    let l = p.lora.as_ref().unwrap();
    for v in [&l.module.v, &l.gene.v] {
        v.set(&Tensor::from_vec(vec![0.2f32, -0.4], (1, 2), &Device::Cpu).unwrap())
            .unwrap();
    }
    let plan = plan_all();
    let (_, loss) = total(&p, &units, &um, &part, &plan);
    let grads = loss.backward().unwrap();
    let loss_at = || total(&p, &units, &um, &part, &plan).0;
    let l = p.lora.as_ref().unwrap();
    for (name, var, flat) in [
        ("a", &l.module.u, 1usize),
        ("v_m", &l.module.v, 0),
        ("u", &l.gene.u, 3),
        ("v_g", &l.gene.v, 1),
    ] {
        let analytic = f64::from(grad_of(&grads, var)[flat]);
        let numeric = finite_difference(var, flat, 1e-3, &loss_at);
        assert!(
            (analytic - numeric).abs() < 2e-3 * (1.0 + analytic.abs()),
            "{name}[{flat}]: autograd {analytic} vs finite difference {numeric}"
        );
        assert!(analytic != 0.0, "{name} receives a gradient");
    }
    // A free gene's `u` row is masked at the step, not in the gradient itself.
    assert_eq!(
        to_host(&l.gene.u_mask).unwrap(),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
}
