//! **Marker scores in the decoder's latent space.** A cell type's marker score is the
//! weighted mean of its marker genes' standardised expression. Here it is computed in closed
//! form from the two embeddings, without materialising a gene × cell matrix.
//!
//! # Derivation
//!
//! The decoder reads gene `g` in cell `i` through the logit `x_gi = ρ_g · z_i + b_g`, with
//! `ρ` the `G × H` feature embedding, `z` the `N × H` cell embedding and `b_g` a per-gene
//! bias. Standardise each gene's logit over the cells, centring it at a reference point `m`
//! of the cell cloud:
//!
//! ```text
//! (x_gi − (ρ_g·m + b_g)) / sd_g  =  ρ_g · (z_i − m) / σ_g,      σ_g = √(ρ_gᵀ Σ_z ρ_g)
//! ```
//!
//! `Σ_z` is the sample covariance of the cells (ddof 1, [`cell_covariance`]), so `σ_g` is
//! exactly the sample standard deviation of gene `g`'s logit. The bias cancels. A type `T`'s
//! score is the weighted mean of its markers' standardised logits, and the sum moves inside
//! the dot product:
//!
//! ```text
//! s_T(i) = Σ_{g∈T} w_g ρ_g·(z_i − m)/σ_g / Σ_{g∈T} w_g  =  (z_i − m) · u_T,
//! u_T    = Σ_{g∈T} w_g ρ_g/σ_g / Σ_{g∈T} w_g
//! ```
//!
//! One direction per type ([`FeatureSpace::directions`]) and one `H`-dim dot product per
//! cell ([`scores`]) give the whole `N × C` table. Every term is a ratio of logits, so the
//! scores do not change under a joint rotation of `z` and `ρ`, or when any gene's `ρ_g` is
//! rescaled.
//!
//! # Relation to the module score
//!
//! Tirosh et al. (2016) score a gene set by the mean of its genes' relative expression
//! against expression-matched control sets; Scanpy's `score_genes` (Wolf et al., 2018)
//! implements it. **This variant scores in the decoder's latent space with the balanced
//! centre.** It reads the logits the fitted model implies rather than observed counts, and
//! standardises each gene by its logit's spread, so a high-variance marker does not dominate
//! its set. There is no control set: the reference is a point `m` of the cell cloud, the
//! **balanced centre** ([`balanced_centre`]). That is the mean of the cluster means, so a
//! large cluster does not define "average" and shift every other cluster's scores.
//!
//! # Markers and calls
//!
//! [`FeatureSpace::marker_sets`] matches a panel through the crate's shared marker reader
//! (the one behind [`parse_and_match_markers`](super::parse_and_match_markers)). Weights are
//! IDF, `ln(C / #types holding the gene)`, where `C` counts every panel type. IDF-0 genes are
//! dropped, and so is a type left with fewer than [`MIN_MARKERS`] markers.
//!
//! [`calls`] gives each cell its best **live** type, one whose direction is not the zero
//! row, and the margin over the runner-up. There is no null and no abstention: a call is the
//! argmax.
//!
//! A feature row whose logit does not vary over the cells (`σ_g = 0`, e.g. a row zeroed as
//! "no usable embedding") is missing data: it is left out of its set's mean rather than
//! averaged in as zero.
//!
//! # References
//!
//! - Tirosh, I. et al. (2016). Dissecting the multicellular ecosystem of metastatic melanoma
//!   by single-cell RNA-seq. *Science* 352:189–196. doi:10.1126/science.aad0501
//! - Wolf, F. A., Angerer, P. & Theis, F. J. (2018). SCANPY: large-scale single-cell gene
//!   expression data analysis. *Genome Biology* 19:15. doi:10.1186/s13059-017-1382-0

use super::markers::parse_and_match_idf_sets;
use super::UNASSIGNED;
use anyhow::{ensure, Result};
use legume_numeric::matrix::traits::MatOps;
use nalgebra::{DMatrix, DVector, RowDVector};
use std::collections::BTreeMap;

/// A type left with fewer markers than this (after matching and IDF) is dropped.
pub const MIN_MARKERS: usize = 3;

/// One type's markers: `(feature index, weight)`.
pub type WeightedSet = Vec<(u32, f64)>;

/// A matched marker panel: the surviving types, in sorted order, and their weighted sets.
pub struct PanelSets {
    pub type_names: Vec<Box<str>>,
    pub sets: Vec<WeightedSet>,
}

/// The feature embedding and each row's logit standard deviation over the cells.
pub struct FeatureSpace {
    /// `[G × H]` feature embedding `ρ`.
    rho: DMatrix<f64>,
    /// `[G]` `σ_g`; 0 marks a dead row.
    sigma: DVector<f64>,
    gene_names: Vec<Box<str>>,
}

impl FeatureSpace {
    /// `rho` is the `[G × H]` feature embedding, `cell_cov` the `[H × H]` sample covariance
    /// of the cells ([`cell_covariance`]), `gene_names` the `G` row names.
    pub fn new(
        rho: &DMatrix<f64>,
        cell_cov: &DMatrix<f64>,
        gene_names: &[Box<str>],
    ) -> Result<Self> {
        let (g, h) = rho.shape();
        ensure!(
            cell_cov.shape() == (h, h),
            "cell covariance is {:?}, but the feature embedding has H = {h}",
            cell_cov.shape()
        );
        ensure!(
            gene_names.len() == g,
            "{} gene names for {g} feature rows",
            gene_names.len()
        );
        let sigma = (rho * cell_cov)
            .component_mul(rho)
            .column_sum()
            .map(|v| v.max(0.0).sqrt());
        Ok(Self {
            rho: rho.clone(),
            sigma,
            gene_names: gene_names.to_vec(),
        })
    }

    /// `[G]` per-gene logit standard deviation `σ_g`.
    #[must_use]
    pub fn sigma(&self) -> &DVector<f64> {
        &self.sigma
    }

    #[must_use]
    pub fn gene_names(&self) -> &[Box<str>] {
        &self.gene_names
    }

    /// Read a `gene<TAB>celltype` panel and match it to this space's genes.
    pub fn marker_sets(&self, markers_path: &str, min_coverage: f32) -> Result<PanelSets> {
        let (type_names, sets) =
            parse_and_match_idf_sets(markers_path, &self.gene_names, min_coverage, MIN_MARKERS)?;
        Ok(PanelSets { type_names, sets })
    }

    /// `[C × H]` one direction `u_T` per set; an empty set, or one whose every row is dead,
    /// gives the zero row.
    #[must_use]
    pub fn directions(&self, sets: &[WeightedSet]) -> DMatrix<f64> {
        let mut dirs = DMatrix::zeros(sets.len(), self.rho.ncols());
        for (set, mut dir) in sets.iter().zip(dirs.row_iter_mut()) {
            let mut total = 0.0;
            for &(g, w) in set {
                let s = self.sigma[g as usize];
                if s > 0.0 {
                    dir += self.rho.row(g as usize) / s * w;
                    total += w;
                }
            }
            if total > 0.0 {
                dir /= total;
            }
        }
        dirs
    }
}

/// `[H × H]` sample covariance (ddof 1) of the `[N × H]` cells.
pub fn cell_covariance(cells: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = cells.nrows();
    ensure!(n >= 2, "a covariance needs at least 2 cells, got {n}");
    let c = cells.centre_columns();
    Ok(c.tr_mul(&c) / (n - 1) as f64)
}

/// The mean of the cluster means. `groups[i]` is cell `i`'s cluster; cells marked
/// [`UNASSIGNED`] are left out.
pub fn balanced_centre(cells: &DMatrix<f64>, groups: &[usize]) -> Result<DVector<f64>> {
    ensure!(
        groups.len() == cells.nrows(),
        "{} group labels for {} cells",
        groups.len(),
        cells.nrows()
    );
    let h = cells.ncols();
    let mut sums: BTreeMap<usize, (RowDVector<f64>, usize)> = BTreeMap::new();
    for (row, &k) in cells.row_iter().zip(groups) {
        if k != UNASSIGNED {
            let (sum, count) = sums.entry(k).or_insert_with(|| (RowDVector::zeros(h), 0));
            *sum += row;
            *count += 1;
        }
    }
    ensure!(!sums.is_empty(), "no cell is assigned to a group");
    let k = sums.len() as f64;
    let centre = sums
        .into_values()
        .fold(RowDVector::zeros(h), |acc, (sum, count)| {
            acc + sum / count as f64
        });
    Ok((centre / k).transpose())
}

/// `[N × C]` scores `(z_i − m) · u_T`.
#[must_use]
pub fn scores(cells: &DMatrix<f64>, centre: &DVector<f64>, dirs: &DMatrix<f64>) -> DMatrix<f64> {
    let mut centred = cells.clone();
    for (mut col, &m) in centred.column_iter_mut().zip(centre.iter()) {
        col.add_scalar_mut(-m);
    }
    centred * dirs.transpose()
}

/// One cell's call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellCall {
    /// The best live type, or `None` when no type is live.
    pub call: Option<usize>,
    /// Best minus runner-up among live types; NaN when fewer than two are live.
    pub margin: f64,
}

/// Per cell, the argmax of `scores` over the live types (rows of `dirs` that are not zero).
#[must_use]
pub fn calls(scores: &DMatrix<f64>, dirs: &DMatrix<f64>) -> Vec<CellCall> {
    let live: Vec<usize> = (0..dirs.nrows())
        .filter(|&t| dirs.row(t).iter().any(|&x| x != 0.0))
        .collect();
    scores
        .row_iter()
        .map(|row| {
            let mut best: Option<(usize, f64)> = None;
            let mut runner_up = f64::NEG_INFINITY;
            for &t in &live {
                let s = row[t];
                match best {
                    Some((_, b)) if s <= b => runner_up = runner_up.max(s),
                    prev => {
                        runner_up = prev.map_or(runner_up, |(_, b)| runner_up.max(b));
                        best = Some((t, s));
                    }
                }
            }
            CellCall {
                call: best.map(|(t, _)| t),
                margin: match best {
                    Some((_, b)) if live.len() >= 2 => b - runner_up,
                    _ => f64::NAN,
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
