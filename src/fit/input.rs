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
    /// The features the pseudobulks were sketched on (`None`: all). Must be
    /// [`super::TrackSpec::base_track_rows`] of the fit's tracks: on a
    /// multi-track axis the sketch reads the base track alone.
    pub sketched_rows: Option<&'a [bool]>,
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
        if let Some(d) = &self.batch_delta {
            anyhow::ensure!(
                d.delta.nrows() == n_features && d.delta.ncols() == d.collapse_batch_names.len(),
                "fit: the batch δ is {} × {}; the counts have {n_features} features and the \
                 collapse {} batch names",
                d.delta.nrows(),
                d.delta.ncols(),
                d.collapse_batch_names.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    /// A δ on another feature axis (the backend's rows, say) is refused, not
    /// read as if its rows were the fit's features.
    #[test]
    fn a_delta_off_the_feature_axis_is_refused() {
        let cells = CscMatrix::from(&CooMatrix::<f32>::new(3, 2));
        let rates = DMatrix::<f32>::zeros(3, 1);
        let cell_to_pb = [0usize, 0];
        let names: Vec<Box<str>> = vec!["a".into(), "b".into()];
        let finest = || Ok((DMatrix::<f32>::zeros(3, 1), vec![1.0]));
        let input = |delta| FitInput {
            cells: &cells,
            batch_membership: &[0, 1],
            batch_names: &names,
            feature_modality: None,
            levels: vec![PbLevel {
                rates: &rates,
                cell_to_pb: &cell_to_pb,
            }],
            finest_counts: &finest,
            batch_delta: Some(BatchDelta {
                delta,
                collapse_batch_names: &names,
            }),
            sketched_rows: None,
        };
        let on_axis = DMatrix::<f32>::from_element(3, 2, 1.0);
        let backend_rows = DMatrix::<f32>::from_element(5, 2, 1.0);
        assert!(input(&on_axis).validate().is_ok());
        assert!(input(&backend_rows).validate().is_err());
    }
}
