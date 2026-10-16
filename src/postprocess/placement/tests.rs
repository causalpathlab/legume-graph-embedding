use super::*;

const H: usize = 4;

/// Cells on four spokes (`±x`, `±y`) at radii `1..=n`, so distance-from-centre percentiles
/// are known and a cell's nearest neighbours lie on its own spoke, at similar norm.
fn spokes(n: usize) -> DMatrix<f32> {
    DMatrix::from_fn(n, H, |i, j| {
        let r = 1.0 + i as f32;
        let sign = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
        if j == i % 2 {
            sign * r
        } else {
            0.0
        }
    })
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
