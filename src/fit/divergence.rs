//! A count track read as a **displacement** of the base track: the
//! unspliced-against-spliced phase portrait, in the base track's space.
//!
//! The base track (spliced counts) alone places every unit and trains the
//! gene features. For every unit `u` and gene `g`, the displaced track
//! (unspliced counts) is read against the base reads of the same unit and
//! gene: of the `x^s_ug + x^u_ug` reads, how many are displaced,
//!
//! ```text
//! x^u_ug ~ Binom(x^s_ug + x^u_ug, σ(κ_u + δ_g + ⟨d_u, ρ_g⟩))
//! ```
//!
//! - `δ_g`: the gene's log ratio, its displaced bias against its base bias
//!   (`b'_g − b_g`);
//! - `κ_u`: the unit's overall displaced share (capture, depth);
//! - `⟨d_u, ρ_g⟩`: the unit's displacement `d_u` read through the base
//!   track's gene features `ρ_g`, which are not trained here.
//!
//! In splicing kinetics, `log(u/s)` sits at the gene's steady-state ratio
//! `log γ_g` when the gene is neither induced nor repressed, above it while
//! induced, below it while repressed. `δ_g` is fitted across all units, so it
//! is the gene's AVERAGE ratio. [`steady_anchor`] then estimates, per gene,
//! where the steady state sits against that average, `ā_g`, from the finest
//! pseudobulks at both ends of the gene's base score. A unit's log
//! velocity ratio is `⟨d_u, ρ_g⟩ − ā_g`, and its velocity
//! `γ_g s_ug (e^{⟨d_u, ρ_g⟩ − ā_g} − 1)`.
//!
//! Nothing here reaches `θ`, `ρ` or `b`, so the displaced track cannot move
//! the base space, whatever its depth.
//!
//! Two stages, as the base fit has:
//! 1. **pseudobulks**: a free `d_p`, `κ_p` per pseudobulk of every level, and
//!    the gene ratios `δ`; ridge toward no displacement;
//! 2. **cells**: an encoder reads a cell's displaced counts through the frozen
//!    gene table and gives `(d_c, κ_c)`; distilled onto the cell's finest
//!    pseudobulk's, then refined on the cell's own counts, ridge toward the
//!    pseudobulk's.

use crate::data::UnifiedData;
use crate::progress::new_progress_bar;
use anyhow::Context;
use legume_numeric::candle::candle_core::{DType, Device, Tensor, Var};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use legume_numeric::candle::encoder::{PooledGeneEncoder, PooledGeneEncoderArgs};
use legume_numeric::candle::feature_embedding::FeatureEmbedding;
use legume_numeric::candle::grad_clip::clipped_backward_step;
use legume_numeric::matrix::rand_util::mix_seed;
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Width of the encoder trunk, as phase 2's.
const TRUNK_WIDTH: usize = 128;
/// Units per optimizer step.
const UNITS_PER_STEP: usize = 128;
const GRAD_CLIP: f64 = 5.0;
/// Share of a gene's finest pseudobulks taken as steady state at EACH end of
/// its base score.
const STEADY_QUANTILE: f64 = 0.05;
/// The var-name prefix the encoder is saved under.
const VAR_PREFIX: &str = "displaced_enc";
/// The per-gene mean's tensor name inside the saved file.
const MEAN_TENSOR: &str = "displaced_enc.feature_mean";
/// Sub-stream tags for this module's seeds.
const SEED_PB: u64 = 0x4449_5350_5042;
const SEED_CELL: u64 = 0x4449_5350_4345;

////////////////////
// Axis and knobs //
////////////////////

/// Where the base and the displaced counts of every gene of the live (base)
/// axis are in the count backend, after [`split_displaced`] cut the axis down
/// to the base track.
#[derive(Clone, Debug)]
pub struct DisplacedAxis {
    /// Per live row (gene): the backend row of its base-track counts.
    pub base_backend_row: Vec<u32>,
    /// Per live row (gene): the backend row of its displaced-track counts,
    /// `u32::MAX` for a gene the displaced track has no row for.
    pub displaced_backend_row: Vec<u32>,
    /// The displaced track's name, e.g. `count/unspliced`.
    pub track_name: Box<str>,
}

impl DisplacedAxis {
    /// Live rows carrying a displaced row, ascending: the displaced support.
    #[must_use]
    pub fn support(&self) -> Vec<u32> {
        self.displaced_backend_row
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r != u32::MAX)
            .map(|(g, _)| g as u32)
            .collect()
    }
}

/// How the displaced track is learned. See the module docs.
#[derive(Clone, Debug)]
pub struct DisplacedTrackConfig {
    pub axis: DisplacedAxis,
    /// Ridge precision on a pseudobulk's displacement (`1/σ²`).
    pub l2_pb: f32,
    /// Ridge precision of a cell's displacement around its pseudobulk's
    /// (`1/τ²`).
    pub l2_cell: f32,
    /// Passes over the pseudobulks.
    pub pb_epochs: usize,
    /// Passes over the cells distilling the encoder onto the pseudobulks.
    pub distill_epochs: usize,
    /// Passes over the cells refining the encoder on their own counts.
    pub refine_epochs: usize,
    pub learning_rate: f64,
}

impl DisplacedTrackConfig {
    /// The default knobs on `axis`.
    #[must_use]
    pub fn new(axis: DisplacedAxis) -> Self {
        Self {
            axis,
            l2_pb: 1.0,
            l2_cell: 1.0,
            pb_epochs: 50,
            distill_epochs: 20,
            refine_epochs: 10,
            learning_rate: 1e-2,
        }
    }
}

/// Cut `unified` down to the base track's rows and say where the displaced
/// track's counts are. Call before building the fit's other per-feature inputs
/// (they then index the base axis).
///
/// Gene `i` is `base_rows[i]` on the current feature axis, and
/// `displaced_rows[i]` its displaced-track row, `None` for a gene the displaced
/// track has no row for. `base_rows` must be strictly ascending: the live axis
/// is the base rows in that order, and [`UnifiedData::subset_features`] takes a
/// selection as wide as the axis to be the axis itself. No displaced row may be
/// a base row or another gene's displaced row. The rows are mapped to backend
/// rows before the cut; the live axis's names are the base rows' names.
pub fn split_displaced(
    unified: &mut UnifiedData,
    base_rows: &[usize],
    displaced_rows: &[Option<usize>],
    track_name: &str,
) -> anyhow::Result<DisplacedAxis> {
    let n_features = unified.n_features();
    anyhow::ensure!(
        base_rows.len() == displaced_rows.len(),
        "{} base rows but {} displaced-row entries: one of each per gene",
        base_rows.len(),
        displaced_rows.len()
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
    // Every row is used at most once: a base row by its gene, a displaced row
    // by one gene and never as a base row.
    let mut used = vec![false; n_features];
    for &row in base_rows {
        used[row] = true;
    }
    for (g, row) in displaced_rows.iter().enumerate() {
        let Some(row) = *row else { continue };
        anyhow::ensure!(
            row < n_features,
            "gene {g}'s displaced row {row} is past the {n_features}-row feature axis"
        );
        anyhow::ensure!(
            !used[row],
            "gene {g}'s displaced row {row} is already a base row or another gene's displaced row"
        );
        used[row] = true;
    }
    let backend = &unified.feature_to_backend_row;
    let base_backend_row: Vec<u32> = base_rows.iter().map(|&row| backend[row] as u32).collect();
    let displaced_backend_row: Vec<u32> = displaced_rows
        .iter()
        .map(|row| row.map_or(u32::MAX, |r| backend[r] as u32))
        .collect();
    unified.subset_features(base_rows);
    Ok(DisplacedAxis {
        base_backend_row,
        displaced_backend_row,
        track_name: track_name.into(),
    })
}

/////////////
// Results //
/////////////

/// What the displaced track's fit gives back.
pub struct DisplacementOutput {
    pub track_name: Box<str>,
    /// Live rows of the displaced support, ascending: the genes every
    /// per-gene vector below and every `d · ρ` readout index.
    pub genes: Vec<u32>,
    /// Per support gene, the displaced track's bias `b'_g = b_g + δ_g`.
    pub b_displaced: Vec<f32>,
    /// Per support gene, the base track's bias `b_g` (in the cells' frame).
    pub b_base: Vec<f32>,
    /// Per support gene, `ā_g`: where the gene's steady state sits against
    /// its average log ratio `δ_g`. A unit's log velocity ratio is
    /// `⟨d_u, ρ_g⟩ − ā_g`.
    pub steady_anchor: Vec<f32>,
    /// Per level (coarsest → finest), `[n_pb × H]` pseudobulk displacements.
    pub d_pb: Vec<DMatrix<f32>>,
    /// Per level, each pseudobulk's intercept `κ_p`.
    pub kappa_pb: Vec<Vec<f32>>,
    /// `[n_cells × H]` cell displacements.
    pub d_cell: DMatrix<f32>,
    /// Each cell's intercept `κ_c`.
    pub kappa_cell: Vec<f32>,
    /// Each cell's finest pseudobulk: a row of the last level of `d_pb`, or
    /// `u32::MAX` for a cell in none.
    pub cell_pb: Vec<u32>,
    /// The cell encoder, for placing new cells' displaced counts.
    pub encoder: DisplacementEncoder,
}

//////////////////////
// Host-side inputs //
//////////////////////

/// One unit's counts on the displaced support: local gene ids and counts, for
/// the base and the displaced track.
#[derive(Clone, Debug, Default)]
pub(crate) struct UnitCounts {
    pub base: Vec<(u32, f32)>,
    pub displaced: Vec<(u32, f32)>,
}

impl UnitCounts {
    fn is_empty(&self) -> bool {
        self.base.is_empty() && self.displaced.is_empty()
    }
}

/// The frozen base tables on the displaced support.
pub(crate) struct Frozen {
    /// `[S × H]` row-major gene rows ρ.
    pub rho: Vec<f32>,
    /// `[S]` base biases b.
    pub b: Vec<f32>,
    pub h: usize,
}

impl Frozen {
    fn s(&self) -> usize {
        self.b.len()
    }

    fn rho_row(&self, g: usize) -> &[f32] {
        &self.rho[g * self.h..(g + 1) * self.h]
    }
}

/// Sum `rows`' counts per group: `groups[p]` lists the members of group `p`.
pub(crate) fn sum_groups(rows: &[UnitCounts], groups: &[Vec<usize>]) -> Vec<UnitCounts> {
    groups
        .par_iter()
        .map(|members| {
            let mut base: rustc_hash::FxHashMap<u32, f32> = Default::default();
            let mut disp: rustc_hash::FxHashMap<u32, f32> = Default::default();
            for &c in members {
                for &(g, v) in &rows[c].base {
                    *base.entry(g).or_default() += v;
                }
                for &(g, v) in &rows[c].displaced {
                    *disp.entry(g).or_default() += v;
                }
            }
            let mut base: Vec<(u32, f32)> = base.into_iter().collect();
            let mut displaced: Vec<(u32, f32)> = disp.into_iter().collect();
            base.sort_unstable_by_key(|&(g, _)| g);
            displaced.sort_unstable_by_key(|&(g, _)| g);
            UnitCounts { base, displaced }
        })
        .collect()
}

/// Dense `[n × s]` base and displaced blocks of `units`.
fn dense_block(units: &[&UnitCounts], s: usize) -> (Vec<f32>, Vec<f32>) {
    let n = units.len();
    let mut xb = vec![0f32; n * s];
    let mut xd = vec![0f32; n * s];
    xb.par_chunks_mut(s)
        .zip(xd.par_chunks_mut(s))
        .zip(units.par_iter())
        .for_each(|((b, d), u)| {
            for &(g, v) in &u.base {
                b[g as usize] = v;
            }
            for &(g, v) in &u.displaced {
                d[g as usize] = v;
            }
        });
    (xb, xd)
}

//////////////////////
// The likelihood   //
//////////////////////

/// The device-side frozen table: `[H × S]`, ρ transposed, for `d · ρᵀ`.
fn rho_t(frozen: &Frozen, dev: &Device) -> anyhow::Result<Tensor> {
    let rho = Tensor::from_slice(&frozen.rho, (frozen.s(), frozen.h), dev)?;
    Ok(rho.t()?.contiguous()?)
}

/// The per-unit negative log-likelihood `[n]` of the displaced reads among
/// all reads of each (unit, gene): `xd ~ Binom(xb + xd, σ(logit))` with
/// `logit = κ + δ + d·ρᵀ`. `d` `[n × H]`, `kappa` `[n]`, `delta` `[1 × S]`.
fn phase_loss(
    rho_t: &Tensor,
    d: &Tensor,
    kappa: &Tensor,
    delta: &Tensor,
    xb: &Tensor,
    xd: &Tensor,
) -> anyhow::Result<Tensor> {
    let logit = d
        .matmul(rho_t)?
        .broadcast_add(delta)?
        .broadcast_add(&kappa.unsqueeze(1)?)?;
    let pos = (xd * softplus(&logit.neg()?)?)?;
    let neg = (xb * softplus(&logit)?)?;
    Ok((pos + neg)?.sum(1)?)
}

/// `ln(1 + eˣ)`, stable: `max(x, 0) + ln(1 + e^{−|x|})`.
fn softplus(x: &Tensor) -> anyhow::Result<Tensor> {
    let tail = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    Ok((x.relu()? + tail)?)
}

/// Each gene's pooled log ratio over `units`, `ln((Σ x^u + ½)/(Σ x^s + ½))`:
/// the starting `δ`.
fn pooled_log_ratio(units: &[UnitCounts], s: usize) -> Vec<f32> {
    let mut su = vec![0f64; s];
    let mut ss = vec![0f64; s];
    for u in units {
        for &(g, v) in &u.displaced {
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

///////////////////////////////
// Stage 1: the pseudobulks  //
///////////////////////////////

/// The pseudobulks' fit.
pub(crate) struct PbFit {
    /// `[n_units × H]`.
    pub d: DMatrix<f32>,
    pub kappa: Vec<f32>,
    /// `[S]` the genes' average log ratio.
    pub delta: Vec<f32>,
}

/// Fit `d_p`, `κ_p` and the gene ratios `δ` on `units`.
pub(crate) fn fit_pseudobulks(
    frozen: &Frozen,
    units: &[UnitCounts],
    knobs: &DisplacedTrackConfig,
    seed: u64,
    dev: &Device,
) -> anyhow::Result<PbFit> {
    let (n, h, s) = (units.len(), frozen.h, frozen.s());
    let rho_t = rho_t(frozen, dev)?;
    let mut rng = StdRng::seed_from_u64(mix_seed(seed, SEED_PB));
    let d = Var::zeros((n, h), DType::F32, dev)?;
    let kappa = Var::zeros(n, DType::F32, dev)?;
    let delta = Var::from_tensor(&Tensor::from_vec(pooled_log_ratio(units, s), (1, s), dev)?)?;
    let mut adam = AdamW::new(
        vec![d.clone(), kappa.clone(), delta.clone()],
        ParamsAdamW {
            lr: knobs.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut order: Vec<usize> = (0..n).filter(|&u| !units[u].is_empty()).collect();
    let half_l2 = f64::from(knobs.l2_pb) / 2.0;
    info!(
        "Displaced track — fitting {} pseudobulks' displacements over {s} genes: {} epochs, \
         ridge {}",
        order.len(),
        knobs.pb_epochs,
        knobs.l2_pb
    );
    let bar = new_progress_bar(knobs.pb_epochs as u64);
    for _ in 0..knobs.pb_epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(UNITS_PER_STEP) {
            let rows: Vec<&UnitCounts> = chunk.iter().map(|&u| &units[u]).collect();
            let (xb, xd) = dense_block(&rows, s);
            let m = chunk.len();
            let idx = Tensor::from_vec(chunk.iter().map(|&u| u as u32).collect(), m, dev)?;
            let xb = Tensor::from_vec(xb, (m, s), dev)?;
            let xd = Tensor::from_vec(xd, (m, s), dev)?;
            let dd = d.as_tensor().index_select(&idx, 0)?;
            let kk = kappa.as_tensor().index_select(&idx, 0)?;
            let fit = phase_loss(&rho_t, &dd, &kk, delta.as_tensor(), &xb, &xd)?;
            let ridge = dd.sqr()?.sum(1)?.affine(half_l2, 0.0)?;
            let loss = (fit + ridge)?.mean_all()?;
            clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    Ok(PbFit {
        d: from_device(d.as_tensor())?,
        kappa: kappa.as_tensor().to_vec1()?,
        delta: delta.as_tensor().flatten_all()?.to_vec1()?,
    })
}

/// Fix the displacements' gauge. Adding one `c` to every unit's `d` and
/// `−⟨c, ρ_g⟩` to every `δ_g` leaves every score unchanged, so the likelihood
/// cannot tell where their mean sits (a weak ridge barely can). Move the
/// units' mean into `δ`: afterwards `d` is a unit's displacement relative to
/// the average unit.
fn center_units(fit: &mut PbFit, f: &Frozen) {
    let c = center_columns(&mut fit.d);
    for (g, delta) in fit.delta.iter_mut().enumerate() {
        *delta += dot(f.rho_row(g), &c);
    }
    log::info!(
        "Displaced track — moved the displacements' shared mean (|c| = {:.3}) into the gene ratios",
        dot(&c, &c).sqrt()
    );
}

/// Subtract each column's mean; return the means.
fn center_columns(m: &mut DMatrix<f32>) -> Vec<f32> {
    let n = m.nrows().max(1) as f32;
    let c: Vec<f32> = m.column_iter().map(|col| col.sum() / n).collect();
    for mut row in m.row_iter_mut() {
        for (x, mean) in row.iter_mut().zip(&c) {
            *x -= mean;
        }
    }
    c
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Per gene, `ā_g`: the log ratio of the displaced reads of the gene's
/// steady-state units to what `κ_p + δ_g` expects of them, with no
/// displacement. Steady-state units are the `STEADY_QUANTILE` of `units` at
/// each end of the gene's base score `⟨θ_p, ρ_g⟩ + b_g`, as the steady-state
/// fit of RNA velocity takes the cells at both extremes of a gene's spliced
/// expression. `theta` `[n × H]` places `units`.
pub(crate) fn steady_anchor(
    frozen: &Frozen,
    theta: &DMatrix<f32>,
    units: &[UnitCounts],
    fit: &PbFit,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    let (s, n) = (frozen.s(), units.len());
    let rho = DMatrix::from_row_slice(s, frozen.h, &frozen.rho);
    let score = theta * rho.transpose(); // [n × S], b_g added per column below
    let k = ((n as f64 * STEADY_QUANTILE).ceil() as usize).clamp(1, n.max(1));
    // Per gene, the k-th lowest and k-th highest score: the ends' thresholds.
    let bounds: Vec<(f32, f32)> = (0..s)
        .into_par_iter()
        .map(|g| {
            let mut col: Vec<f32> = score.column(g).iter().copied().collect();
            col.sort_unstable_by(f32::total_cmp);
            (col[k - 1], col[n - k])
        })
        .collect();
    let mut num = vec![0f64; s];
    let mut den = vec![0f64; s];
    for (p, unit) in units.iter().enumerate() {
        let row = rows.start + p;
        let at_end = |g: usize| {
            let x = score[(p, g)];
            x <= bounds[g].0 || x >= bounds[g].1
        };
        for &(g, v) in &unit.displaced {
            if at_end(g as usize) {
                num[g as usize] += f64::from(v);
            }
        }
        for &(g, v) in &unit.base {
            let g = g as usize;
            if at_end(g) {
                let logit = fit.kappa[row] + fit.delta[g];
                den[g] += f64::from(v) * f64::from(logit).exp();
            }
        }
    }
    num.iter()
        .zip(&den)
        .map(|(&a, &b)| ((a + 0.5) / (b + 0.5)).ln() as f32)
        .collect()
}

/////////////////////////////////
// Stage 2: the cells' encoder //
/////////////////////////////////

/// The trained cell encoder: reads displaced counts on the support, through
/// the frozen gene table, and gives `[d, κ]`.
pub struct DisplacementEncoder {
    encoder: PooledGeneEncoder,
    varmap: VarMap,
    mean_1d: Tensor,
    h: usize,
}

impl DisplacementEncoder {
    fn build(frozen: &Frozen, mean_1d: &[f32], dev: &Device) -> anyhow::Result<Self> {
        let rho = Tensor::from_slice(&frozen.rho, (frozen.s(), frozen.h), dev)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let encoder = PooledGeneEncoder::new(
            FeatureEmbedding::fixed(rho),
            PooledGeneEncoderArgs {
                layers: &[TRUNK_WIDTH],
                out_dim: frozen.h + 1,
                attn_pool: true,
                in_dim_extra: 0,
            },
            &varmap,
            vb.pp(VAR_PREFIX),
        )?;
        let mean_1d = Tensor::from_slice(mean_1d, (1, frozen.s()), dev)?;
        Ok(Self {
            encoder,
            varmap,
            mean_1d,
            h: frozen.h,
        })
    }

    /// `(d [n × H], κ [n])` for dense displaced counts `x [n × S]`.
    fn forward(&self, x: &Tensor, train: bool) -> anyhow::Result<(Tensor, Tensor)> {
        let z = self
            .encoder
            .forward(x, None, Some(&self.mean_1d), None, train)?;
        let d = z.narrow(1, 0, self.h)?.contiguous()?;
        let kappa = z.narrow(1, self.h, 1)?.squeeze(1)?.contiguous()?;
        Ok((d, kappa))
    }

    /// Save the trunk and the per-gene mean as safetensors at `path`.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        tensors.insert(MEAN_TENSOR.to_string(), self.mean_1d.flatten_all()?);
        legume_numeric::candle::candle_core::safetensors::save(&tensors, path)
            .with_context(|| format!("saving the displaced-track encoder to {path}"))?;
        Ok(())
    }
}

/// The cells' fit.
pub(crate) struct CellFit {
    pub d: DMatrix<f32>,
    pub kappa: Vec<f32>,
    pub encoder: DisplacementEncoder,
}

/// Train the cell encoder: distil onto each cell's pseudobulk's displacement
/// (`target` `[n × H]`), then refine on the cells' own counts against the
/// gene ratios `delta`, ridge toward the targets.
pub(crate) fn fit_cells(
    frozen: &Frozen,
    cells: &[UnitCounts],
    target: &DMatrix<f32>,
    delta: &[f32],
    knobs: &DisplacedTrackConfig,
    seed: u64,
    dev: &Device,
) -> anyhow::Result<CellFit> {
    let (n, s) = (cells.len(), frozen.s());
    anyhow::ensure!(
        target.nrows() == n,
        "the cells' targets are not [cells × H]"
    );
    let rho_t = rho_t(frozen, dev)?;
    let target_t = to_device(target, dev)?;
    let delta = Tensor::from_slice(delta, (1, s), dev)?;

    // The trunk's gate divides by each gene's mean displaced rate.
    let mut mean = vec![0f64; s];
    for c in cells {
        for &(g, v) in &c.displaced {
            mean[g as usize] += f64::from(v);
        }
    }
    let mean_1d: Vec<f32> = mean
        .iter()
        .map(|&m| ((m / n.max(1) as f64) as f32).max(1e-6))
        .collect();
    let encoder = DisplacementEncoder::build(frozen, &mean_1d, dev)?;

    let mut rng = StdRng::seed_from_u64(mix_seed(seed, SEED_CELL));
    let active: Vec<usize> = (0..n).filter(|&c| !cells[c].is_empty()).collect();
    let mut train = active.clone();

    let block = |ids: &[usize]| -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let rows: Vec<&UnitCounts> = ids.iter().map(|&c| &cells[c]).collect();
        let (xb, xd) = dense_block(&rows, s);
        let m = ids.len();
        let idx = Tensor::from_vec(ids.iter().map(|&c| c as u32).collect(), m, dev)?;
        Ok((
            idx,
            Tensor::from_vec(xb, (m, s), dev)?,
            Tensor::from_vec(xd, (m, s), dev)?,
        ))
    };

    let mut adam = AdamW::new(
        encoder.varmap.all_vars(),
        ParamsAdamW {
            lr: knobs.learning_rate / 10.0,
            weight_decay: 1e-4,
            ..Default::default()
        },
    )?;
    info!(
        "Displaced track — encoding {} cells: {} distil + {} refine epochs",
        train.len(),
        knobs.distill_epochs,
        knobs.refine_epochs
    );
    let half_l2 = f64::from(knobs.l2_cell) / 2.0;
    let bar = new_progress_bar((knobs.distill_epochs + knobs.refine_epochs) as u64);
    for epoch in 0..knobs.distill_epochs + knobs.refine_epochs {
        let distil = epoch < knobs.distill_epochs;
        train.shuffle(&mut rng);
        for chunk in train.chunks(UNITS_PER_STEP) {
            let (idx, xb, xd) = block(chunk)?;
            let (d, kappa) = encoder.forward(&xd, true)?;
            let toward = (&d - &target_t.index_select(&idx, 0)?)?.sqr()?.sum(1)?;
            let loss = if distil {
                toward.mean_all()?
            } else {
                let fit = phase_loss(&rho_t, &d, &kappa, &delta, &xb, &xd)?;
                (fit + toward.affine(half_l2, 0.0)?)?.mean_all()?
            };
            clipped_backward_step(&mut adam, &loss, GRAD_CLIP)?;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    // Every cell through the trained encoder; empty cells keep their targets.
    let mut d_all = target.clone();
    let mut kappa_all = vec![0f32; n];
    for chunk in active.chunks(UNITS_PER_STEP) {
        let (_, _, xd) = block(chunk)?;
        let (d, kappa) = encoder.forward(&xd, false)?;
        let d = from_device(&d)?;
        let kappa: Vec<f32> = kappa.to_vec1()?;
        for (k, &c) in chunk.iter().enumerate() {
            d_all.row_mut(c).copy_from(&d.row(k));
            kappa_all[c] = kappa[k];
        }
    }
    Ok(CellFit {
        d: d_all,
        kappa: kappa_all,
        encoder,
    })
}

/////////////////////////
// From the fit's data //
/////////////////////////

/// Every cell's base and displaced counts on the support, streamed from the
/// count backend (raw counts). `support` are the live rows; local gene `i` is
/// `support[i]`.
pub(crate) fn read_counts(
    unified: &UnifiedData,
    axis: &DisplacedAxis,
    support: &[u32],
) -> anyhow::Result<Vec<UnitCounts>> {
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    // backend row → (local gene, is displaced).
    let mut role = vec![(u32::MAX, false); data.num_rows()];
    for (i, &g) in support.iter().enumerate() {
        let g = g as usize;
        role[axis.base_backend_row[g] as usize] = (i as u32, false);
        role[axis.displaced_backend_row[g] as usize] = (i as u32, true);
    }
    let mut out: Vec<UnitCounts> = (0..n_cells).map(|_| UnitCounts::default()).collect();
    let chunk = (1usize << 14).min(n_cells.max(1));
    data.for_each_triplet(0..n_cells, chunk, |brow, col, v| {
        if v == 0.0 {
            return;
        }
        let (g, displaced) = role[brow as usize];
        if g == u32::MAX {
            return;
        }
        let cell = &mut out[col as usize];
        if displaced {
            cell.displaced.push((g, v));
        } else {
            cell.base.push((g, v));
        }
    })?;
    Ok(out)
}

/// Fit the displaced track after the base fit: the gene rows `e_feat`
/// `[G × H]` and biases `b_feat` `[G]` in the cells' frame, the pseudobulk
/// tables per level (same frame), and each level's cell membership.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_displaced(
    unified: &UnifiedData,
    knobs: &DisplacedTrackConfig,
    e_feat: &DMatrix<f32>,
    b_feat: &[f32],
    pb_tables: &[&DMatrix<f32>],
    cell_to_pb_per_level: &[Vec<usize>],
    seed: u64,
    dev: &Device,
) -> anyhow::Result<DisplacementOutput> {
    let h = e_feat.ncols();
    let support = knobs.axis.support();
    anyhow::ensure!(
        !support.is_empty(),
        "the displaced track has no rows on the base axis"
    );
    let mut rho = Vec::with_capacity(support.len() * h);
    for &g in &support {
        rho.extend(e_feat.row(g as usize).iter().copied());
    }
    let frozen = Frozen {
        rho,
        b: support.iter().map(|&g| b_feat[g as usize]).collect(),
        h,
    };
    let cells = read_counts(unified, &knobs.axis, &support)?;

    // Every level's pseudobulks, stacked.
    let mut pb_counts = Vec::new();
    let mut level_sizes = Vec::new();
    for (table, c2pb) in pb_tables.iter().zip(cell_to_pb_per_level) {
        let n_pb = table.nrows();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
        for (c, &p) in c2pb.iter().enumerate() {
            if p < n_pb {
                groups[p].push(c);
            }
        }
        pb_counts.extend(sum_groups(&cells, &groups));
        level_sizes.push(n_pb);
    }
    let mut pb = fit_pseudobulks(&frozen, &pb_counts, knobs, seed, dev)?;
    center_units(&mut pb, &frozen);

    // The steady state, on the finest level.
    let finest = cell_to_pb_per_level.last().context("no pseudobulk level")?;
    let finest_offset: usize = level_sizes[..level_sizes.len() - 1].iter().sum();
    let n_finest = *level_sizes.last().expect("a level");
    let finest_rows = finest_offset..finest_offset + n_finest;
    let steady = steady_anchor(
        &frozen,
        pb_tables.last().expect("a level"),
        &pb_counts[finest_rows.clone()],
        &pb,
        finest_rows,
    );

    // Each cell's finest pseudobulk's displacement is its target and centre.
    let mut target = DMatrix::<f32>::zeros(cells.len(), h);
    for (c, &p) in finest.iter().enumerate() {
        if p < n_finest {
            target.row_mut(c).copy_from(&pb.d.row(finest_offset + p));
        }
    }
    let fit = fit_cells(&frozen, &cells, &target, &pb.delta, knobs, seed, dev)?;

    // Split the stacked pseudobulk rows back per level.
    let mut d_pb = Vec::new();
    let mut kappa_pb = Vec::new();
    let mut at = 0usize;
    for &n in &level_sizes {
        d_pb.push(pb.d.rows(at, n).into_owned());
        kappa_pb.push(pb.kappa[at..at + n].to_vec());
        at += n;
    }
    Ok(DisplacementOutput {
        track_name: knobs.axis.track_name.clone(),
        b_displaced: frozen.b.iter().zip(&pb.delta).map(|(b, d)| b + d).collect(),
        b_base: frozen.b.clone(),
        genes: support,
        steady_anchor: steady,
        d_pb,
        kappa_pb,
        d_cell: fit.d,
        kappa_cell: fit.kappa,
        cell_pb: finest
            .iter()
            .map(|&p| if p < n_finest { p as u32 } else { u32::MAX })
            .collect(),
        encoder: fit.encoder,
    })
}

/////////////
// Helpers //
/////////////

fn to_device(m: &DMatrix<f32>, dev: &Device) -> anyhow::Result<Tensor> {
    let rows: Vec<f32> = m.transpose().as_slice().to_vec();
    Ok(Tensor::from_vec(rows, (m.nrows(), m.ncols()), dev)?)
}

fn from_device(t: &Tensor) -> anyhow::Result<DMatrix<f32>> {
    let (n, h) = t.dims2()?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(DMatrix::from_row_slice(n, h, &v))
}

#[cfg(test)]
#[path = "divergence_tests.rs"]
mod divergence_tests;
