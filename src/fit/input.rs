//! What [`super::fit`] trains on: the cells' counts, their batches, and the
//! pseudobulks a caller built from them (data-beans'
//! `unified::build_embedding_pseudobulks`, say). Plain matrices and a count
//! stream, so this crate depends on no storage crate.

use legume_numeric::matrix::cell_count_stream::CellCountStream;
use nalgebra::DMatrix;

/// One collapse level: its pseudobulk rates and each cell's pseudobulk.
pub struct PbLevel<'a> {
    /// `[n_features × n_pb]` on the fit's feature axis (batch-adjusted when
    /// the collapse fit a δ).
    pub rates: &'a DMatrix<f32>,
    /// `[n_cells]`: each cell's pseudobulk at this level.
    pub cell_to_pb: &'a [usize],
}

/// The finest collapse's per-batch gene fold δ.
pub struct BatchDelta<'a> {
    /// `[n_features × collapse batches]` on the fit's feature axis.
    pub delta: &'a DMatrix<f32>,
    /// The collapse's batch names, in `delta`'s column order.
    pub collapse_batch_names: &'a [Box<str>],
}

/// The finest level's observed counts `[n_features × n_pb]` and each
/// pseudobulk's size, built only when a module partition needs them.
pub type FinestCounts<'a> = &'a dyn Fn() -> anyhow::Result<(DMatrix<f32>, Vec<f32>)>;

/// Everything [`super::fit`] reads besides its [`super::FitConfig`].
pub struct FitInput<'a> {
    /// The cells' counts on the fit's feature axis: the samplers' only source.
    pub cells: &'a dyn CellCountStream,
    /// `[n_cells]`: each cell's batch id, indexing `batch_names`.
    pub batch_membership: &'a [u32],
    /// Batch labels in id order.
    pub batch_names: &'a [Box<str>],
    /// Modality id per feature on a multiome axis; `None` for one panel.
    pub feature_modality: Option<std::sync::Arc<[u32]>>,
    /// The collapse levels, coarsest → finest (at least one).
    pub levels: Vec<PbLevel<'a>>,
    pub finest_counts: FinestCounts<'a>,
    /// `None` when the collapse fit no δ.
    pub batch_delta: Option<BatchDelta<'a>>,
}

impl FitInput<'_> {
    /// Every level must map each cell, and the counts, batches and rates must
    /// share one cell axis and one feature axis.
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        let (n_cells, n_features) = (self.cells.n_cells(), self.cells.n_features());
        anyhow::ensure!(!self.levels.is_empty(), "fit: no pseudobulk level");
        anyhow::ensure!(
            self.batch_membership.len() == n_cells,
            "fit: {} batch ids for {n_cells} cells",
            self.batch_membership.len()
        );
        if let Some(&b) = self
            .batch_membership
            .iter()
            .find(|&&b| b as usize >= self.batch_names.len())
        {
            anyhow::bail!(
                "fit: batch id {b} past the {} batch names",
                self.batch_names.len()
            );
        }
        for (l, level) in self.levels.iter().enumerate() {
            anyhow::ensure!(
                level.rates.nrows() == n_features && level.cell_to_pb.len() == n_cells,
                "fit: level {l} has {} rate rows and {} cells; the counts have {n_features} \
                 features and {n_cells} cells",
                level.rates.nrows(),
                level.cell_to_pb.len()
            );
        }
        if let Some(m) = &self.feature_modality {
            anyhow::ensure!(
                m.len() == n_features,
                "fit: {} feature modalities for {n_features} features",
                m.len()
            );
        }
        Ok(())
    }
}
