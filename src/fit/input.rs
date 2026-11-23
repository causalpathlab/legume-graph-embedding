//! What [`super::fit`] trains on: the cells' counts, their batches, and the
//! pseudobulks collapsed from them, read through the cell and pseudobulk
//! contracts of `legume-cell-pseudobulk` so this crate depends on no storage
//! crate (data-beans implements them).

use legume_cell_pseudobulk::{CellCountStream, PseudobulkLevels};

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
    /// The cells' pseudobulk levels on the same feature axis. On a multi-track
    /// axis they should be sketched on [`super::TrackSpec::base_track_rows`].
    pub pseudobulks: &'a dyn PseudobulkLevels,
}

impl FitInput<'_> {
    /// The counts, batches and pseudobulks must share one cell axis and one
    /// feature axis.
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        let (n_cells, n_features) = (self.cells.n_cells(), self.cells.n_features());
        let n_batches = self.batch_names.len();
        anyhow::ensure!(self.pseudobulks.n_levels() > 0, "fit: no pseudobulk level");
        anyhow::ensure!(
            self.batch_membership.len() == n_cells,
            "fit: {} batch ids for {n_cells} cells",
            self.batch_membership.len()
        );
        if let Some(&b) = self
            .batch_membership
            .iter()
            .find(|&&b| b as usize >= n_batches)
        {
            anyhow::bail!("fit: batch id {b} past the {n_batches} batch names");
        }
        for l in 0..self.pseudobulks.n_levels() {
            let cell_to_pb = self.pseudobulks.cell_to_pb(l);
            anyhow::ensure!(
                cell_to_pb.len() == n_cells,
                "fit: level {l} maps {} cells; the counts have {n_cells}",
                cell_to_pb.len()
            );
        }
        if let Some(d) = self.pseudobulks.finest_delta() {
            anyhow::ensure!(
                d.nrows() == n_features && d.ncols() == n_batches,
                "fit: the batch δ is {} × {}; expected {n_features} features × {n_batches} \
                 batches",
                d.nrows(),
                d.ncols()
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
    use nalgebra::DMatrix;
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    use std::borrow::Cow;

    /// One level of one pseudobulk over three features, with a δ of `rows`.
    struct OneLevel {
        delta_rows: usize,
    }

    impl PseudobulkLevels for OneLevel {
        fn n_levels(&self) -> usize {
            1
        }
        fn rates(&self, _: usize) -> Cow<'_, DMatrix<f32>> {
            Cow::Owned(DMatrix::zeros(3, 1))
        }
        fn cell_to_pb(&self, _: usize) -> &[usize] {
            &[0, 0]
        }
        fn finest_counts(&self) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)> {
            Ok((DMatrix::zeros(3, 1), vec![1.0]))
        }
        fn finest_delta(&self) -> Option<Cow<'_, DMatrix<f32>>> {
            Some(Cow::Owned(DMatrix::from_element(self.delta_rows, 2, 1.0)))
        }
    }

    /// A δ on another feature axis (the backend's rows, say) is refused, not
    /// read as if its rows were the fit's features.
    #[test]
    fn a_delta_off_the_feature_axis_is_refused() {
        let cells = CscMatrix::from(&CooMatrix::<f32>::new(3, 2));
        let names: Vec<Box<str>> = vec!["a".into(), "b".into()];
        let input = |pseudobulks| FitInput {
            cells: &cells,
            batch_membership: &[0, 1],
            batch_names: &names,
            feature_modality: None,
            pseudobulks,
        };
        assert!(input(&OneLevel { delta_rows: 3 }).validate().is_ok());
        assert!(input(&OneLevel { delta_rows: 5 }).validate().is_err());
    }
}
