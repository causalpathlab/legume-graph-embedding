//! A count track read against the base track, gene by gene: the
//! unspliced-against-spliced phase portrait, on the base fit's cell states.
//!
//! The base track (spliced counts) alone places every cell (`θ_c`) and trains
//! the gene rows (`ρ_g`, `b_g`). For every cell `c` and gene `g`, the
//! divergent track (unspliced counts) is read against the base reads of the
//! same cell and gene: of the `x^s_cg + x^u_cg` reads, how many are on the divergent track,
//!
//! ```text
//! x^u_cg ~ Binom(x^s_cg + x^u_cg, σ(κ_c + δ_g + ⟨θ_c, η_g⟩))
//! ```
//!
//! - `κ_c`: the cell's overall divergent share (capture, depth), profiled
//!   per cell;
//! - `δ_g`: the gene's log ratio for a typical cell;
//! - `η_g`: the gene's direction in the cell-state space along which its
//!   divergent share rises, one vector per gene, ridge-shrunk. `θ_c` stays
//!   the base fit's, so nothing here moves the base space.
//!
//! In splicing kinetics, `log(u/s)` sits at the gene's steady-state ratio
//! `log γ_g` when the gene is neither induced nor repressed, above it while
//! induced, below it while repressed. [`steady_anchor`] reads each gene's
//! steady state, `ā_g`, off the observed ratio of the cells at both ends of the
//! gene's base score, as the steady-state fit of RNA velocity does. A cell's
//! log velocity ratio is `ℓ_cg = ⟨θ_c, η_g⟩ − ā_g`, and `log γ_g = δ_g + ā_g`.
//!
//! [`cell_velocity`] turns those into one vector per cell in the base space:
//! with `β = 1`, `d log s_g / dt = γ_g (e^{ℓ_cg} − 1)`, and a move `v` of
//! `θ_c` changes the base model's `log π_cg` by `⟨v, ρ_g − ρ̄_c⟩`
//! (`π_c = softmax_g(⟨θ_c, ρ_g⟩ + b_g)`, `ρ̄_c = Σ_g π_cg ρ_g`), so
//!
//! ```text
//! θ̇_c = argmin_v Σ_g π_cg (⟨v, ρ_g − ρ̄_c⟩ − γ_g (e^{ℓ_cg} − 1))²
//! ```

use crate::data::UnifiedData;
use crate::progress::new_progress_bar;
use legume_numeric::candle::candle_core::{DType, Device, Tensor, Var};
use legume_numeric::candle::candle_nn::ops::{sigmoid, softmax_last_dim};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::candle::grad_clip::clipped_backward_step;
use legume_numeric::candle::loss::log_sigmoid;
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::matrix::traits::ConvertMatOps;
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Cells per optimizer step.
const CELLS_PER_STEP: usize = 128;
const GRAD_CLIP: f64 = 5.0;
/// Newton steps on each cell's `κ` per optimizer step, from its last value.
const KAPPA_STEPS: usize = 3;
/// Newton steps on each cell's `κ` once the gene terms are fitted.
const KAPPA_FINAL_STEPS: usize = 20;
/// Share of the cells taken as steady state at EACH end of a gene's base
/// score.
const STEADY_QUANTILE: f64 = 0.05;
/// Genes per block when scoring every cell against a block of genes.
const GENES_PER_BLOCK: usize = 256;
/// Cells per block when solving the per-cell velocities.
const CELLS_PER_VELOCITY_BLOCK: usize = 16;
/// The largest log velocity ratio the velocity reads, so one runaway gene
/// cannot dominate a cell's least squares.
const MAX_LOG_RATIO: f64 = 5.0;
/// Sub-stream tag for this module's seed.
const SEED_FIT: u64 = 0x4449_5350_4345;

////////////////////
// Axis and knobs //
////////////////////

/// Where the base and the divergent counts of every gene of the live (base)
/// axis are in the count backend, after [`split_divergence`] cut the axis down
/// to the base track.
#[derive(Clone, Debug)]
pub struct DivergenceAxis {
    /// Per live row (gene): the backend row of its base-track counts.
    pub base_backend_row: Vec<u32>,
    /// Per live row (gene): the backend row of its divergent-track counts,
    /// `u32::MAX` for a gene the divergent track has no row for.
    pub divergent_backend_row: Vec<u32>,
    /// The divergent track's name, e.g. `count/unspliced`.
    pub track_name: Box<str>,
}

impl DivergenceAxis {
    /// Live rows carrying a divergent row, ascending: the divergence support.
    #[must_use]
    pub fn support(&self) -> Vec<u32> {
        self.divergent_backend_row
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r != u32::MAX)
            .map(|(g, _)| g as u32)
            .collect()
    }
}

/// How the divergent track is learned. See the module docs.
#[derive(Clone, Debug)]
pub struct DivergenceConfig {
    pub axis: DivergenceAxis,
    /// Ridge on the genes' directions, `λ/2 Σ_g ‖η_g‖²`, against the summed
    /// binomial log-likelihood of every cell.
    pub l2: f32,
    /// Passes over the cells.
    pub epochs: usize,
    pub learning_rate: f64,
}

/// Cut `unified` down to the base track's rows and say where the divergent
/// track's counts are. Call before building the fit's other per-feature inputs
/// (they then index the base axis).
///
/// Gene `i` is `base_rows[i]` on the current feature axis, and
/// `divergent_rows[i]` its divergent-track row, `None` for a gene the divergent
/// track has no row for. `base_rows` must be strictly ascending: the live axis
/// is the base rows in that order, and [`UnifiedData::subset_features`] takes a
/// selection as wide as the axis to be the axis itself. No divergent row may be
/// a base row or another gene's divergent row. The rows are mapped to backend
/// rows before the cut; the live axis's names are the base rows' names.
pub fn split_divergence(
    unified: &mut UnifiedData,
    base_rows: &[usize],
    divergent_rows: &[Option<usize>],
    track_name: &str,
) -> anyhow::Result<DivergenceAxis> {
    let n_features = unified.n_features();
    anyhow::ensure!(
        base_rows.len() == divergent_rows.len(),
        "{} base rows but {} divergent-row entries: one of each per gene",
        base_rows.len(),
        divergent_rows.len()
    );
    anyhow::ensure!(
        base_rows.windows(2).all(|w| w[0] < w[1]),
        "the base rows must be strictly ascending"
    );
    if let Some(&last) = base_rows.last() {
        anyhow::ensure!(
            last < n_features,
            "base row {last} is past the {n_features}-row feature axis"
        );
    }
    // Every row is used at most once: a base row by its gene, a divergent row
    // by one gene and never as a base row.
    let mut used = vec![false; n_features];
    for &row in base_rows {
        used[row] = true;
    }
    for (g, row) in divergent_rows.iter().enumerate() {
        let Some(row) = *row else { continue };
        anyhow::ensure!(
            row < n_features,
            "gene {g}'s divergent row {row} is past the {n_features}-row feature axis"
        );
        anyhow::ensure!(
            !used[row],
            "gene {g}'s divergent row {row} is already a base row or another gene's divergent row"
        );
        used[row] = true;
    }
    let backend = &unified.feature_to_backend_row;
    let base_backend_row: Vec<u32> = base_rows.iter().map(|&row| backend[row] as u32).collect();
    let divergent_backend_row: Vec<u32> = divergent_rows
        .iter()
        .map(|row| row.map_or(u32::MAX, |r| backend[r] as u32))
        .collect();
    unified.subset_features(base_rows);
    Ok(DivergenceAxis {
        base_backend_row,
        divergent_backend_row,
        track_name: track_name.into(),
    })
}

/////////////
// Results //
/////////////

/// What the divergent track's fit gives back.
pub struct DivergenceOutput {
    pub track_name: Box<str>,
    /// Live rows of the divergence support, ascending: the genes every
    /// per-gene vector and every row of `loading` below index.
    pub genes: Vec<u32>,
    /// Per support gene, `δ_g`: the log ratio of divergent to base reads for
    /// a typical cell (`κ` is centred over the cells).
    pub ratio: Vec<f32>,
    /// Per support gene, `ā_g`: the observed log ratio of the gene's
    /// steady-state cells against `κ_c + δ_g`. A cell's log velocity ratio is
    /// `⟨θ_c, η_g⟩ − ā_g`, and `log γ_g = δ_g + ā_g`.
    pub steady_anchor: Vec<f32>,
    /// Per support gene, `log γ_g = δ_g + ā_g`, the steady-state log ratio.
    pub log_gamma: Vec<f32>,
    /// `[S × H]` the genes' directions `η_g`.
    pub loading: DMatrix<f32>,
    /// Per cell, `κ_c` (centred over the cells with reads).
    pub kappa_cell: Vec<f32>,
    /// `[n_cells × H]` each cell's velocity `θ̇_c` in the base space; zero for
    /// a cell with no reads on the support.
    pub velocity_cell: DMatrix<f32>,
}

//////////////////////
// Host-side inputs //
//////////////////////

/// One cell's counts on the divergence support: local gene ids and counts, for
/// the base and the divergent track.
#[derive(Clone, Debug, Default)]
pub(crate) struct CellCounts {
    pub base: Vec<(u32, f32)>,
    pub divergent: Vec<(u32, f32)>,
}

impl CellCounts {
    fn is_empty(&self) -> bool {
        self.base.is_empty() && self.divergent.is_empty()
    }
}

/// The frozen base tables on the divergence support.
pub(crate) struct Frozen {
    /// `[S × H]` gene rows ρ.
    pub rho: DMatrix<f32>,
    /// `[S]` base biases b.
    pub b: Vec<f32>,
}

/// Dense `[n × s]` base and divergent blocks of `cells`.
fn dense_block(cells: &[&CellCounts], s: usize) -> (Vec<f32>, Vec<f32>) {
    let n = cells.len();
    let mut xb = vec![0f32; n * s];
    let mut xd = vec![0f32; n * s];
    xb.par_chunks_mut(s)
        .zip(xd.par_chunks_mut(s))
        .zip(cells.par_iter())
        .for_each(|((b, d), u)| {
            for &(g, v) in &u.base {
                b[g as usize] = v;
            }
            for &(g, v) in &u.divergent {
                d[g as usize] = v;
            }
        });
    (xb, xd)
}

/// Each gene's pooled log ratio over `cells`, `ln((Σ x^u + ½)/(Σ x^s + ½))`:
/// the starting `δ`.
fn pooled_log_ratio(cells: &[CellCounts], s: usize) -> Vec<f32> {
    let mut su = vec![0f64; s];
    let mut ss = vec![0f64; s];
    for u in cells {
        for &(g, v) in &u.divergent {
            su[g as usize] += f64::from(v);
        }
        for &(g, v) in &u.base {
            ss[g as usize] += f64::from(v);
        }
    }
    su.iter()
        .zip(&ss)
        .map(|(&a, &b)| ((a + 0.5) / (b + 0.5)).ln() as f32)
        .collect()
}

////////////////////
// The likelihood //
////////////////////

/// The per-cell negative log-likelihood `[n]` of the divergent reads among
/// all reads of each (cell, gene): `xd ~ Binom(xb + xd, σ(logit))`.
fn binomial_nll(logit: &Tensor, xb: &Tensor, xd: &Tensor) -> anyhow::Result<Tensor> {
    let pos = (xd * log_sigmoid(logit)?)?;
    let neg = (xb * log_sigmoid(&logit.neg()?)?)?;
    Ok((pos + neg)?.neg()?.sum(1)?)
}

/// Each cell's `κ` maximising its own likelihood given the rest of its
/// logits `a` `[n × S]`: `steps` Newton steps from `kappa` `[n]`. A cell with
/// no reads keeps its start.
fn profile_kappa(
    a: &Tensor,
    xb: &Tensor,
    xd: &Tensor,
    kappa: &Tensor,
    steps: usize,
) -> anyhow::Result<Tensor> {
    let n_reads = (xb + xd)?;
    let mut k = kappa.clone();
    for _ in 0..steps {
        let p = sigmoid(&a.broadcast_add(&k.unsqueeze(1)?)?)?;
        let grad = ((&n_reads * &p)? - xd)?.sum(1)?;
        let curv = (&n_reads * (&p * (1.0 - &p)?)?)?.sum(1)?;
        k = (k - (grad / (curv + 1e-6)?)?)?;
    }
    Ok(k)
}

//////////////
// The fit  //
//////////////

/// The gene terms and the cells' intercepts.
pub(crate) struct RatioFit {
    /// `[n_cells]`.
    pub kappa: Vec<f32>,
    /// `[S]`.
    pub delta: Vec<f32>,
    /// `[S × H]`.
    pub eta: DMatrix<f32>,
}

/// Fit `κ_c`, `δ_g` and `η_g` on the cells: `theta` `[n_cells × H]` fixed
/// (on `dev`), `cells` their counts on the support. Adam on minibatches of
/// cells over `δ` and `η`, each cell's `κ` profiled by Newton steps inside
/// every step; then `κ` centred over the cells with reads, its mean moved into
/// `δ`.
pub(crate) fn fit_ratio(
    theta: &Tensor,
    cells: &[CellCounts],
    s: usize,
    knobs: &DivergenceConfig,
    seed: u64,
) -> anyhow::Result<RatioFit> {
    let (n, h) = theta.dims2()?;
    anyhow::ensure!(n == cells.len(), "θ is not [cells × H]");
    let dev = theta.device();
    let delta = Var::from_tensor(&Tensor::from_vec(pooled_log_ratio(cells, s), (1, s), dev)?)?;
    // Stored transposed, `[H × S]`, so `θ ηᵀ` is a plain matmul.
    let eta_t = Var::zeros((h, s), DType::F32, dev)?;
    let mut kappa = vec![0f32; n];
    let mut adam = AdamW::new(
        vec![delta.clone(), eta_t.clone()],
        ParamsAdamW {
            lr: knobs.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut active: Vec<usize> = (0..n).filter(|&c| !cells[c].is_empty()).collect();
    let n_active = active.len().max(1);
    // The objective is `Σ_c NLL_c + λ/2 ‖η‖²`, divided by the number of
    // cells so a minibatch mean estimates it.
    let ridge = f64::from(knobs.l2) / 2.0 / n_active as f64;
    let mut rng = StdRng::seed_from_u64(mix_seed(seed, SEED_FIT));
    info!(
        "Divergence — fitting {} cells' {s} gene ratios and directions: {} epochs, ridge {}",
        active.len(),
        knobs.epochs,
        knobs.l2
    );
    // A minibatch: the logits without κ, the counts, and the cells' last κ.
    let batch = |ids: &[usize], kappa: &[f32]| -> anyhow::Result<[Tensor; 4]> {
        let rows: Vec<&CellCounts> = ids.iter().map(|&c| &cells[c]).collect();
        let (xb, xd) = dense_block(&rows, s);
        let m = ids.len();
        let idx = Tensor::from_vec(ids.iter().map(|&c| c as u32).collect(), m, dev)?;
        let a = theta
            .index_select(&idx, 0)?
            .matmul(eta_t.as_tensor())?
            .broadcast_add(delta.as_tensor())?;
        let k0 = Tensor::from_vec(ids.iter().map(|&c| kappa[c]).collect::<Vec<f32>>(), m, dev)?;
        Ok([
            a,
            Tensor::from_vec(xb, (m, s), dev)?,
            Tensor::from_vec(xd, (m, s), dev)?,
            k0,
        ])
    };
    let store = |ids: &[usize], k: &Tensor, kappa: &mut [f32]| -> anyhow::Result<()> {
        for (&c, v) in ids.iter().zip(k.to_vec1::<f32>()?) {
            kappa[c] = v;
        }
        Ok(())
    };
    let bar = new_progress_bar(knobs.epochs as u64);
    for _ in 0..knobs.epochs {
        active.shuffle(&mut rng);
        for chunk in active.chunks(CELLS_PER_STEP) {
            let [a, xb, xd, k0] = batch(chunk, &kappa)?;
            let k = profile_kappa(&a.detach(), &xb, &xd, &k0, KAPPA_STEPS)?.detach();
            let fit = binomial_nll(&a.broadcast_add(&k.unsqueeze(1)?)?, &xb, &xd)?;
            let loss =
                (fit.mean_all()? + eta_t.as_tensor().sqr()?.sum_all()?.affine(ridge, 0.0)?)?;
            clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
            store(chunk, &k, &mut kappa)?;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    // Every cell's κ at the fitted gene terms, then centred.
    for chunk in active.chunks(CELLS_PER_STEP) {
        let [a, xb, xd, k0] = batch(chunk, &kappa)?;
        let k = profile_kappa(&a, &xb, &xd, &k0, KAPPA_FINAL_STEPS)?;
        store(chunk, &k, &mut kappa)?;
    }
    let mean = active.iter().map(|&c| f64::from(kappa[c])).sum::<f64>() / n_active as f64;
    for &c in &active {
        kappa[c] -= mean as f32;
    }
    let delta: Vec<f32> = delta
        .as_tensor()
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|d| d + mean as f32)
        .collect();
    Ok(RatioFit {
        kappa,
        delta,
        eta: DMatrix::from_tensor(&eta_t.as_tensor().t()?)?,
    })
}

/////////////////////
// Steady state    //
/////////////////////

/// Per gene, `ā_g`: the log ratio of the divergent reads of the cells at
/// both `STEADY_QUANTILE` ends of the gene's base score `⟨θ_c, ρ_g⟩ + b_g` to
/// what `κ_c + δ_g` expects of them from their base reads,
/// `ln((Σ x^u + ½) / (Σ x^s e^{κ_c + δ_g} + ½))`. The ends are taken as the
/// gene's steady state, as the steady-state fit of RNA velocity takes the
/// cells at both extremes of a gene's spliced expression.
pub(crate) fn steady_anchor(
    theta: &Tensor,
    frozen: &Frozen,
    fit: &RatioFit,
    cells: &[CellCounts],
) -> anyhow::Result<Vec<f32>> {
    let (n, s) = (theta.dim(0)?, frozen.b.len());
    if n == 0 {
        return Ok(vec![0.0; s]);
    }
    // Each gene's base and divergent reads per cell.
    let mut base: Vec<Vec<(u32, f32)>> = vec![Vec::new(); s];
    let mut divergent: Vec<Vec<(u32, f32)>> = vec![Vec::new(); s];
    for (c, cell) in cells.iter().enumerate() {
        for &(g, v) in &cell.base {
            base[g as usize].push((c as u32, v));
        }
        for &(g, v) in &cell.divergent {
            divergent[g as usize].push((c as u32, v));
        }
    }
    let k = ((n as f64 * STEADY_QUANTILE).ceil() as usize).clamp(1, n);
    let mut anchor = vec![0f32; s];
    for start in (0..s).step_by(GENES_PER_BLOCK) {
        let end = (start + GENES_PER_BLOCK).min(s);
        let w = end - start;
        // `[w × n]`, one gene's scores per row; its bias `b_g` shifts the
        // whole row and so cannot move which cells are at its ends.
        let rho_blk = to_device(&frozen.rho.rows(start, w).into_owned(), theta.device())?;
        let score: Vec<f32> = rho_blk.matmul(&theta.t()?)?.flatten_all()?.to_vec1()?;
        let blk: Vec<f32> = (0..w)
            .into_par_iter()
            .map(|j| {
                let g = start + j;
                let row = &score[j * n..(j + 1) * n];
                let mut order = row.to_vec();
                let (_, &mut lo, upper) = order.select_nth_unstable_by(k - 1, f32::total_cmp);
                let hi = if n - k > k - 1 {
                    *upper.select_nth_unstable_by(n - k - k, f32::total_cmp).1
                } else {
                    lo
                };
                let at_end = |c: u32| row[c as usize] <= lo || row[c as usize] >= hi;
                let num: f64 = divergent[g]
                    .iter()
                    .filter(|&&(c, _)| at_end(c))
                    .map(|&(_, v)| f64::from(v))
                    .sum();
                let den: f64 = base[g]
                    .iter()
                    .filter(|&&(c, _)| at_end(c))
                    .map(|&(c, v)| {
                        f64::from(v) * f64::from(fit.kappa[c as usize] + fit.delta[g]).exp()
                    })
                    .sum();
                ((num + 0.5) / (den + 0.5)).ln() as f32
            })
            .collect();
        anchor[start..end].copy_from_slice(&blk);
    }
    Ok(anchor)
}

/////////////////////
// Cell velocity   //
/////////////////////

/// Each cell's velocity in the base space: `θ̇_c` solving the weighted least
/// squares of the module docs, with `r_cg = γ_g (e^{ℓ_cg} − 1)`,
/// `ℓ_cg = ⟨θ_c, η_g⟩ − ā_g`, `γ_g = e^{δ_g + ā_g}`. `π_c` is the base model's
/// share of each support gene, renormalised over the support.
pub(crate) fn cell_velocity(
    theta: &Tensor,
    frozen: &Frozen,
    fit: &RatioFit,
    anchor: &[f32],
) -> anyhow::Result<DMatrix<f32>> {
    let dev = theta.device();
    let (n, h) = theta.dims2()?;
    let s = frozen.b.len();
    let rho = to_device(&frozen.rho, dev)?;
    let rho_t = rho.t()?.contiguous()?;
    let b = Tensor::from_slice(&frozen.b, (1, s), dev)?;
    let eta_t = to_device(&fit.eta.transpose(), dev)?;
    let anchor_t = Tensor::from_slice(anchor, (1, s), dev)?;
    let gamma: Vec<f32> = fit
        .delta
        .iter()
        .zip(anchor)
        .map(|(d, a)| (d + a).exp())
        .collect();
    let gamma_t = Tensor::from_slice(&gamma, (1, s), dev)?;
    let mut out = DMatrix::<f32>::zeros(n, h);
    for start in (0..n).step_by(CELLS_PER_VELOCITY_BLOCK) {
        let m = (start + CELLS_PER_VELOCITY_BLOCK).min(n) - start;
        let th = theta.narrow(0, start, m)?;
        let ell = th
            .matmul(&eta_t)?
            .broadcast_sub(&anchor_t)?
            .clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)?;
        let r = (ell.exp()? - 1.0)?.broadcast_mul(&gamma_t)?;
        let v = solve_velocity(&th, &r, &rho, &rho_t, &b)?;
        out.rows_mut(start, m).copy_from(&v);
    }
    Ok(out)
}

/// For cells `th` `[m × H]` and their targets `r` `[m × S]`: per cell, the
/// `v` minimising `Σ_g π_cg (⟨v, ρ_g − ρ̄_c⟩ − r_cg)²`, `[m × H]`. A tiny
/// ridge keeps a cell whose `π` spans fewer than H directions solvable.
fn solve_velocity(
    th: &Tensor,
    r: &Tensor,
    rho: &Tensor,
    rho_t: &Tensor,
    b: &Tensor,
) -> anyhow::Result<DMatrix<f32>> {
    let (m, h) = th.dims2()?;
    let pi = softmax_last_dim(&th.matmul(rho_t)?.broadcast_add(b)?)?; // [m × S]
    let rho_bar = pi.matmul(rho)?; // [m × H]
    let w = (&pi * r)?; // [m × S]
    let rhs = (w.matmul(rho)? - rho_bar.broadcast_mul(&w.sum_keepdim(1)?)?)?; // [m × H]

    // Σ_g π_cg ρ_g ρ_gᵀ − ρ̄_c ρ̄_cᵀ, one symmetric H × H per cell.
    let weighted = pi.unsqueeze(2)?.broadcast_mul(&rho.unsqueeze(0)?)?; // [m × S × H]
    let second = rho_t.unsqueeze(0)?.broadcast_matmul(&weighted)?; // [m × H × H]
    let outer = rho_bar.unsqueeze(2)?.matmul(&rho_bar.unsqueeze(1)?)?;
    let lhs: Vec<f32> = (second - outer)?.flatten_all()?.to_vec1()?;
    let rhs = DMatrix::<f32>::from_tensor(&rhs)?;
    let rows: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map(|c| {
            let block = &lhs[c * h * h..(c + 1) * h * h];
            // Symmetric, so the row-major block reads the same column-major.
            let mut a = DMatrix::from_column_slice(h, h, block);
            let jitter = 1e-6 * (a.trace() / h as f32).max(1e-12) + 1e-12;
            for i in 0..h {
                a[(i, i)] += jitter;
            }
            let y = rhs.row(c).transpose();
            if let Some(ch) = a.clone().cholesky() {
                return ch.solve(&y).iter().copied().collect();
            }
            a.lu()
                .solve(&y)
                .map_or_else(|| vec![0.0; h], |v| v.iter().copied().collect())
        })
        .collect();
    Ok(DMatrix::from_fn(m, h, |c, k| rows[c][k]))
}

/////////////////////////
// From the fit's data //
/////////////////////////

/// Every cell's base and divergent counts on the support, streamed from the
/// count backend (raw counts). `support` are the live rows; local gene `i` is
/// `support[i]`.
pub(crate) fn read_counts(
    unified: &UnifiedData,
    axis: &DivergenceAxis,
    support: &[u32],
) -> anyhow::Result<Vec<CellCounts>> {
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    // backend row → (local gene, is divergent).
    let mut role = vec![(u32::MAX, false); data.num_rows()];
    for (i, &g) in support.iter().enumerate() {
        let g = g as usize;
        role[axis.base_backend_row[g] as usize] = (i as u32, false);
        role[axis.divergent_backend_row[g] as usize] = (i as u32, true);
    }
    let mut out: Vec<CellCounts> = (0..n_cells).map(|_| CellCounts::default()).collect();
    let chunk = (1usize << 14).min(n_cells.max(1));
    data.for_each_triplet(0..n_cells, chunk, |brow, col, v| {
        if v == 0.0 {
            return;
        }
        let (g, divergent) = role[brow as usize];
        if g == u32::MAX {
            return;
        }
        let cell = &mut out[col as usize];
        if divergent {
            cell.divergent.push((g, v));
        } else {
            cell.base.push((g, v));
        }
    })?;
    Ok(out)
}

/// Fit the divergent track after the base fit: the cells' states `theta`
/// `[n_cells × H]` (on the fit's device), the gene rows `e_feat` `[G × H]` and
/// biases `b_feat` `[G]`, all in the cells' frame.
pub(crate) fn fit_divergence(
    unified: &UnifiedData,
    knobs: &DivergenceConfig,
    theta: &Tensor,
    e_feat: &DMatrix<f32>,
    b_feat: &[f32],
    seed: u64,
) -> anyhow::Result<DivergenceOutput> {
    let support = knobs.axis.support();
    anyhow::ensure!(
        !support.is_empty(),
        "the divergent track has no rows on the base axis"
    );
    let rows: Vec<usize> = support.iter().map(|&g| g as usize).collect();
    let frozen = Frozen {
        rho: e_feat.select_rows(&rows),
        b: rows.iter().map(|&g| b_feat[g]).collect(),
    };
    let theta = theta.contiguous()?;
    let cells = read_counts(unified, &knobs.axis, &support)?;
    let fit = fit_ratio(&theta, &cells, frozen.b.len(), knobs, seed)?;
    let anchor = steady_anchor(&theta, &frozen, &fit, &cells)?;
    info!("Divergence — each cell's velocity in the base space");
    let mut velocity = cell_velocity(&theta, &frozen, &fit, &anchor)?;
    for (c, cell) in cells.iter().enumerate() {
        if cell.is_empty() {
            velocity.row_mut(c).fill(0.0);
        }
    }
    let log_gamma = fit.delta.iter().zip(&anchor).map(|(d, a)| d + a).collect();
    Ok(DivergenceOutput {
        track_name: knobs.axis.track_name.clone(),
        genes: support,
        ratio: fit.delta,
        steady_anchor: anchor,
        log_gamma,
        loading: fit.eta,
        kappa_cell: fit.kappa,
        velocity_cell: velocity,
    })
}

/// `m` on `dev` as a contiguous row-major tensor (`to_tensor` alone gives a
/// transposed view, which CUDA's matmul refuses).
fn to_device(m: &DMatrix<f32>, dev: &Device) -> anyhow::Result<Tensor> {
    Ok(m.to_tensor(dev)?.contiguous()?)
}

#[cfg(test)]
#[path = "divergence_tests.rs"]
mod divergence_tests;
