use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-6, "entry {i}: got {g}, want {w}");
    }
}

/// Each batch's fold row is its δ column, in batch-id order.
#[test]
fn rows_follow_the_batch_ids() {
    let delta = DMatrix::from_row_slice(3, 2, &[1.0, 0.5, 2.0, 1.0, 4.0, 8.0]);
    let off = batch_gene_fold(&delta, &names(&["late", "early"]))
        .unwrap()
        .expect("two batches ⇒ a fold");
    assert_eq!(off.n_batches(), 2);
    assert_eq!(off.batch_names, names(&["late", "early"]));
    assert_close(off.row(0), &[1.0, 2.0, 4.0]);
    assert_close(off.row(1), &[0.5, 1.0, 8.0]);
}

#[test]
fn one_batch_yields_no_fold() {
    let delta = DMatrix::from_row_slice(2, 1, &[3.0, 0.2]);
    assert!(batch_gene_fold(&delta, &names(&["only"]))
        .unwrap()
        .is_none());
}

#[test]
fn a_delta_without_a_column_per_batch_is_an_error() {
    let delta = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(batch_gene_fold(&delta, &names(&["a", "b", "c"])).is_err());
}

#[test]
fn nonpositive_delta_is_floored() {
    let delta = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 2.0]);
    let off = batch_gene_fold(&delta, &names(&["a", "b"]))
        .unwrap()
        .unwrap();
    assert_close(off.row(0), &[DELTA_FLOOR, DELTA_FLOOR]);
    assert_close(off.row(1), &[1.0, 2.0]);
}
