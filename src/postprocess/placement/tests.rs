use super::*;

const H: usize = 4;

/// Cells on four spokes (`±x`, `±y`), cell `i` at radius `r(i)`, so distance-from-centre
/// percentiles are known and a cell's nearest neighbours lie on its own spoke.
fn spokes_with(n: usize, r: impl Fn(usize) -> f32) -> DMatrix<f32> {
    DMatrix::from_fn(n, H, |i, j| {
        let sign = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
        if j == i % 2 {
            sign * r(i)
        } else {
            0.0
        }
    })
}

/// Spokes at radii `1..=n`.
fn spokes(n: usize) -> DMatrix<f32> {
    spokes_with(n, |i| 1.0 + i as f32)
}

/// A diverged fit leaves NaN in the co-embedding. NaN rows must not read as placed.
#[test]
fn nan_features_are_not_on_the_manifold() {
    let cells = spokes(400);
    let feats = DMatrix::from_element(10, H, f32::NAN);
    let got = coembed_placement(&cells, &feats).unwrap();
    assert!(!got.on_manifold(), "an all-NaN co-embedding passed: {got}");

    let mut one_bad = DMatrix::from_fn(10, H, |i, j| cells[(i * 40, j)]);
    one_bad.row_mut(3).fill(f32::NAN);
    let got = coembed_placement(&cells, &one_bad).unwrap();
    assert!(
        got.beyond_p99 >= 0.1,
        "a NaN row must count as off the cloud, got {got}"
    );
    // One diverged row in ten is under the beyond_p99 bar, but a co-embedding with any
    // non-finite row is not placed — it is broken.
    assert!(!got.on_manifold(), "one NaN row in ten passed: {got}");
}

/// Identical cells have no cloud to measure against; zero-width rows have no space.
/// Both are errors, not a division by zero or a panic.
#[test]
fn degenerate_inputs_are_errors() {
    let same = DMatrix::from_element(5, H, 1.0f32);
    assert!(coembed_placement(&same, &DMatrix::from_element(2, H, 1.0)).is_err());
    let one = DMatrix::from_element(1, H, 1.0f32);
    assert!(coembed_placement(&one, &DMatrix::from_element(2, H, 1.0)).is_err());
    let empty = DMatrix::<f32>::zeros(5, 0);
    assert!(coembed_placement(&empty, &DMatrix::zeros(3, 0)).is_err());
}

/// Half the features parked at the hub, half beside the outermost cells (still inside the
/// 99th percentile): each half is misplaced, yet the mean nearest-cell percentile is ~50 and
/// nothing is beyond p99. The split itself must fail the contract.
#[test]
fn a_split_placement_is_not_on_the_manifold() {
    let n = 1000;
    let cells = spokes(n);
    let feats = DMatrix::from_fn(100, H, |g, j| {
        let src = if g % 2 == 0 {
            5 + g / 2
        } else {
            n - 60 + g / 2
        };
        cells[(src, j)]
    });
    let got = coembed_placement(&cells, &feats).unwrap();
    assert!(
        got.beyond_p99 < 0.15 && (got.nn_norm_pct - 50.0).abs() <= 10.0,
        "the fixture must hide the split from the mean, got {got}"
    );
    assert!(!got.on_manifold(), "a hub/edge split passed: {got}");
}

/// Features sitting among cells of ordinary norm pass.
#[test]
fn features_among_ordinary_cells_pass() {
    let n = 1000;
    let cells = spokes(n);
    let feats = DMatrix::from_fn(100, H, |g, j| cells[(g * 10 + 3, j)]);
    let got = coembed_placement(&cells, &feats).unwrap();
    assert!(got.on_manifold(), "{got}");
}

/// Cells at tied distances share a rank: a feature among a tied group reads the middle of
/// the group's percentile range, not its bottom. Here half the cells sit at radius 1 and half
/// at radius 3, and every feature sits among the inner half: its percentile is 25, not 0.
#[test]
fn tied_cells_rank_at_their_midpoint() {
    let n = 200;
    let cells = spokes_with(n, |i| if i < n / 2 { 1.0 } else { 3.0 });
    let feats = DMatrix::from_fn(10, H, |g, j| cells[(g * 4, j)]);
    let got = coembed_placement(&cells, &feats).unwrap();
    assert!(
        (got.nn_norm_pct - 25.0).abs() < 5.0,
        "features among the tied inner half should read ~25, got {got}"
    );
}

/// A finite row far out is off the cloud, not "non-finite": only a NaN or infinite
/// coordinate counts as a diverged row.
#[test]
fn a_huge_finite_row_is_not_counted_as_non_finite() {
    let cells = spokes(400);
    let mut feats = DMatrix::from_fn(10, H, |i, j| cells[(i * 40, j)]);
    feats.row_mut(0).fill(1e20);
    let got = coembed_placement(&cells, &feats).unwrap();
    assert_eq!(got.non_finite, 0, "{got}");
    assert!(got.beyond_p99 >= 0.1, "{got}");
}
