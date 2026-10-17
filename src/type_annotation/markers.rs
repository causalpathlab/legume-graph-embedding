use super::*;
use log::warn;

/// Per-type `(feature_index, weight)` marker lists — one inner `Vec` per type.
pub type MarkerSets = Vec<Vec<(u32, f32)>>;

/// The kept type names and their weighted marker lists.
type TypedSets<W> = (Vec<Box<str>>, Vec<Vec<(u32, W)>>);

/// Parse a marker TSV and match its genes to `gene_names`, returning the
/// sorted type vocabulary and per-type `(gene_index, weight)` lists.
///
/// Matching is exact-first (lowercased full name or its last `_`-segment
/// symbol), falling back to the shared `flexible_name_match`. Weights are IDF
/// — `ln(C / df_gene)` — unless `use_idf` is false (then unit), so markers
/// shared across many types are down-weighted (a ubiquitous gene → IDF 0,
/// dropped so it can't anchor a type).
///
/// **A type that ends up with no markers is dropped, not kept.** Its signature would
/// be the zero vector, which is not merely uninformative but an outright **magnet**
/// in the nearest-centroid assignment — see `term_ora::assign_nearest` for the
/// geometry and what it cost on a real run. Dropping the type here also keeps
/// the reported counts honest: a type we cannot score should not appear at all.
///
/// **A type that keeps only a sliver of its panel is the quieter, more dangerous case**,
/// and [`report_panel_coverage`] is what surfaces it. The embedding is trained on a
/// feature axis narrowed by the HVG cut (`--n-hvg`) and the feature-null FDR, and a marker
/// off that axis is not *projected*, it is simply **absent** — so it silently leaves the
/// panel. A type that entered with 20 markers and scores on 1 still produces a confident-
/// looking call, indistinguishable in the output from one that kept all 20. `min_coverage`
/// is the floor below which that stops being a warning and becomes an error.
pub fn parse_and_match_markers(
    markers_path: &str,
    gene_names: &[Box<str>],
    use_idf: bool,
    min_coverage: f32,
) -> Result<(Vec<Box<str>>, MarkerSets)> {
    let panel = match_panel(markers_path, gene_names, min_coverage, |_| true)?;
    let c = panel.type_names.len();
    let use_idf = use_idf && c >= 2;
    let type_markers = panel.weigh(|df| {
        if use_idf {
            let w = idf_weight(c, df);
            (w > 0.0).then_some(w) // drop ubiquitous (IDF 0)
        } else {
            Some(1.0)
        }
    });
    drop_thin_types(&panel.type_names, type_markers, 1, 1)
}

/// [`parse_and_match_markers`]'s matching and IDF, weighted in `f64` and with each set sorted
/// by gene index: `ln(C / df_gene)` over every panel type `C`, IDF-0 genes dropped. Genes
/// that are not `live` leave after weighting, so they still count in `df` and `C`, and do not
/// count toward `min_coverage`. A type left with fewer than `min_markers` live markers is then
/// dropped, and fewer than `min_types` surviving types is an error.
pub(super) fn parse_and_match_idf_sets(
    markers_path: &str,
    gene_names: &[Box<str>],
    min_coverage: f32,
    min_markers: usize,
    min_types: usize,
    live: impl Fn(u32) -> bool,
) -> Result<TypedSets<f64>> {
    let panel = match_panel(markers_path, gene_names, min_coverage, &live)?;
    let c = panel.type_names.len() as f64;
    let mut sets = panel.weigh(|df| Some((c / df as f64).ln()).filter(|&w| w > 0.0));
    for set in &mut sets {
        set.retain(|&(gi, _)| live(gi));
        set.sort_unstable_by_key(|&(gi, _)| gi);
    }
    drop_thin_types(&panel.type_names, sets, min_markers, min_types)
}

/// A marker panel matched to a feature axis, before weighting.
struct MatchedPanel {
    /// Every panel type, sorted — including those that matched nothing.
    type_names: Vec<Box<str>>,
    /// Per type, the matched gene indices.
    membership: Vec<FxHashSet<u32>>,
    /// Per matched gene, the number of types holding it.
    df: FxHashMap<u32, usize>,
}

impl MatchedPanel {
    /// Each type's genes, weighted by `weight(df_gene)`; a `None` weight drops the gene.
    fn weigh<W>(&self, weight: impl Fn(usize) -> Option<W>) -> Vec<Vec<(u32, W)>> {
        self.membership
            .iter()
            .map(|genes| {
                genes
                    .iter()
                    .filter_map(|&gi| Some((gi, weight(self.df[&gi])?)))
                    .collect()
            })
            .collect()
    }
}

/// Match the panel and report its coverage, counting only the `live` matches as covered.
fn match_panel(
    markers_path: &str,
    gene_names: &[Box<str>],
    min_coverage: f32,
    live: impl Fn(u32) -> bool,
) -> Result<MatchedPanel> {
    let pairs = read_marker_tsv(markers_path)?;

    // Type vocabulary (sorted, stable).
    let mut type_names: Vec<Box<str>> = pairs.iter().map(|(_, t)| t.clone()).collect();
    type_names.sort();
    type_names.dedup();
    let type_idx: FxHashMap<&str, usize> = type_names
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_ref(), i))
        .collect();

    // Shared three-tier gene matcher (exact → symbol → flexible fallback).
    let index = GeneIndex::build(gene_names);

    let c = type_names.len();
    let mut membership: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); c];
    // Distinct genes the panel *asked* for, per type — the denominator of the coverage
    // report below. Kept separate from `membership` (the ones we could find) because the
    // gap between the two is the whole point.
    let mut requested: Vec<FxHashSet<&str>> = vec![FxHashSet::default(); c];
    // Memoize matches by gene string: the same gene recurs across types, and
    // the flexible fallback is an O(genes) scan — resolve each distinct gene
    // at most once.
    let mut match_cache: FxHashMap<&str, Option<usize>> = FxHashMap::default();
    let mut unmatched = 0usize;
    for (gene, ct) in &pairs {
        let Some(&ti) = type_idx.get(ct.as_ref()) else {
            continue;
        };
        requested[ti].insert(gene.as_ref());
        match *match_cache
            .entry(gene.as_ref())
            .or_insert_with(|| index.match_gene(gene))
        {
            Some(gi) => {
                membership[ti].insert(gi as u32);
            }
            None => unmatched += 1,
        }
    }
    if unmatched > 0 {
        info!("{unmatched} marker pairs had no matching gene (dropped)");
    }
    let hits: Vec<usize> = membership
        .iter()
        .map(|genes| genes.iter().filter(|&&gi| live(gi)).count())
        .collect();
    report_panel_coverage(&type_names, &requested, &hits, min_coverage)?;

    // IDF: df_gene = #types containing the gene.
    let mut df: FxHashMap<u32, usize> = FxHashMap::default();
    for genes in &membership {
        for &gi in genes {
            *df.entry(gi).or_insert(0) += 1;
        }
    }
    Ok(MatchedPanel {
        type_names,
        membership,
        df,
    })
}

/// Drop the types left with fewer than `min_markers` markers — none of their genes are in the
/// embedding, or IDF zeroed them. See [`parse_and_match_markers`]: an empty type's zero
/// signature out-competes every real one for any cell whose best cosine is below 0.5, so
/// keeping it silently mislabels the majority of the data. Fewer than `min_types` survivors
/// is an error naming the types lost. Both bounds are at least 1.
fn drop_thin_types<W>(
    type_names: &[Box<str>],
    sets: Vec<Vec<(u32, W)>>,
    min_markers: usize,
    min_types: usize,
) -> Result<TypedSets<W>> {
    debug_assert!(min_markers >= 1 && min_types >= 1);
    let c = type_names.len();
    let (kept, lost): (Vec<_>, Vec<_>) = type_names
        .iter()
        .zip(sets)
        .partition(|(_, m)| m.len() >= min_markers);
    let lost: Vec<&str> = lost.iter().map(|(n, _)| n.as_ref()).collect();
    anyhow::ensure!(
        kept.len() >= min_types,
        "{} marker type(s) kept {min_markers} or more usable marker gene(s) in the embedding; \
         {min_types} needed. Lost: {}. A marker is lost when no row name matches it, when every \
         type holds it (IDF 0), or when its row is dead — check the marker file's gene naming \
         against the dictionary's row names",
        kept.len(),
        listed(&lost)
    );
    if !lost.is_empty() {
        warn!(
            "{} of {c} marker type(s) kept fewer than {min_markers} usable marker gene(s) in the \
             embedding and were DROPPED: {}. Their cells will be assigned among the remaining \
             types. If you need them, widen the feature axis (raise `--n-hvg`) or force the \
             panel into training with `--must-train-features <the marker file>`.",
            lost.len(),
            listed(&lost)
        );
    }

    Ok(kept.into_iter().map(|(n, m)| (n.clone(), m)).unzip())
}

/// Coverage below which a panel is reported as degraded rather than merely noted. Shared with
/// `data_beans::aux`'s gene-set reconciliation, which asks the same question of GAF/GMT term
/// sets — a marker panel is just another term→genes map, and a thin overlap means the same
/// thing in both.
use data_beans::aux::gene_sets::COVERAGE_WARN_FRAC as WARN_COVERAGE;

/// How many under-covered types to name before truncating the warning.
const MAX_LISTED: usize = 10;

/// `names` joined, truncated after [`MAX_LISTED`].
fn listed(names: &[&str]) -> String {
    let shown = names[..names.len().min(MAX_LISTED)].join(", ");
    match names.len().saturating_sub(MAX_LISTED) {
        0 => shown,
        more => format!("{shown}, … and {more} more"),
    }
}

/// Report what fraction of the marker panel actually reached the embedding, overall and per
/// type — and fail when it falls under `min_coverage`.
///
/// This is the guard on a failure mode that otherwise leaves no trace. Marker calls are only
/// as good as the genes they are computed on, but the embedding's feature axis is narrowed
/// long before the panel is read (`--n-hvg`, then the feature-null FDR), and a marker that
/// falls off that axis just quietly stops contributing. The result is a call that *looks*
/// identical to a well-supported one: same shape of output, same confidence, computed on a
/// handful of surviving genes. Without this, the only signal is a count of dropped pairs,
/// which says nothing about *which* types were gutted.
///
/// `min_coverage == 0` (the default) never errors — it reports and warns, so existing
/// pipelines keep running while the degradation stops being invisible. Set it to make a
/// thin panel fatal.
fn report_panel_coverage(
    type_names: &[Box<str>],
    requested: &[FxHashSet<&str>],
    hits: &[usize],
    min_coverage: f32,
) -> Result<()> {
    let (n_req, n_hit): (usize, usize) = (
        requested.iter().map(FxHashSet::len).sum(),
        hits.iter().sum(),
    );
    if n_req == 0 {
        return Ok(());
    }
    let overall = n_hit as f32 / n_req as f32;

    // Per-type coverage, worst first — the overall number can look healthy while one type
    // has been reduced to a single gene.
    let mut thin: Vec<(f32, &str, usize, usize)> = (0..type_names.len())
        .filter(|&i| !requested[i].is_empty())
        .map(|i| {
            let (hit, req) = (hits[i], requested[i].len());
            (hit as f32 / req as f32, type_names[i].as_ref(), hit, req)
        })
        .filter(|&(cov, ..)| cov < WARN_COVERAGE)
        .collect();
    thin.sort_by(|a, b| a.0.total_cmp(&b.0));

    info!(
        "marker panel: {n_hit}/{n_req} genes ({:.0}%) are usable on the embedding's feature axis, \
         across {} type(s)",
        overall * 100.0,
        type_names.len()
    );

    if !thin.is_empty() {
        let listed: Vec<String> = thin
            .iter()
            .take(MAX_LISTED)
            .map(|(cov, name, hit, req)| format!("{name} {hit}/{req} ({:.0}%)", cov * 100.0))
            .collect();
        // Name `--must-train-features`, not `senna gem --markers`: this guard fires in senna and
        // pinto too, and they have the former but not the latter. A warning whose remedy is a
        // flag your binary does not have is a warning you learn to ignore.
        warn!(
            "{} marker type(s) kept under {:.0}% of their panel — their calls rest on the few \
             genes that survived the feature axis, and will still look confident: {}{}. \
             Widen the axis (raise `--n-hvg`) or force the panel into training with \
             `--must-train-features <the marker file>` (in faba, `gem --markers` does this).",
            thin.len(),
            WARN_COVERAGE * 100.0,
            listed.join("; "),
            if thin.len() > MAX_LISTED {
                format!(", … and {} more", thin.len() - MAX_LISTED)
            } else {
                String::new()
            }
        );
    }

    anyhow::ensure!(
        overall >= min_coverage,
        "marker panel coverage {:.0}% is below the required {:.0}% ({n_hit}/{n_req} genes usable \
         on the embedding's feature axis). The panel the calls would be built on is mostly \
         missing. Widen the feature axis (raise `--n-hvg`) or force the panel into training \
         with `--must-train-features <the marker file>` (in faba, `gem --markers` does this).",
        overall * 100.0,
        min_coverage * 100.0
    );
    Ok(())
}

// The panel and the label→CL map are read by one row reader with one line rule and one
// label key, so a label written the same way in both files meets.
//
// TODO: move both files to upstream readers together (data-beans' `read_membership_pairs`,
// enrichment's `parse_label_map`) once those follow `split_tab_first` (tab-first, cells kept
// in their columns) and key labels like `type_key`; moving one alone would read the two files
// differently again.

/// Split one line: on the tab only if it has one before any trailing whitespace, so a label
/// keeps its commas (a Cell Ontology name such as `CD4-positive, alpha-beta T cell`); else on
/// commas. Cells keep their columns (an empty one stays empty) and are trimmed and unquoted.
fn split_tab_first(line: &str) -> Vec<&str> {
    use legume_numeric::matrix::table::Delimiter;
    let line = line.trim_end();
    let delim = if line.contains('\t') {
        Delimiter::Tab
    } else {
        Delimiter::Comma
    };
    delim.split(line)
}

/// A label as a type name: spaces → `_`.
fn type_key(label: &str) -> Box<str> {
    Box::from(label.replace(' ', "_"))
}

/// The first two cells of every row of a two-column label file (gz-aware), split by
/// [`split_tab_first`]. Comment rows (a first cell starting with `#` or `%`, indented or
/// quoted) are skipped; so are rows with an empty or missing first or second cell, with a
/// warning counting them. A line that cannot be read (not UTF-8, a corrupt gz) is an error
/// naming the line.
fn read_two_field_rows(path: &str) -> Result<Vec<(Box<str>, Box<str>)>> {
    use std::io::BufRead;
    let mut rows = Vec::new();
    let mut gapped = 0usize;
    let reader = legume_numeric::matrix::common_io::open_buf_reader(path)?;
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("line {}", i + 1))?;
        let cells = split_tab_first(&line);
        match cells.as_slice() {
            [] | [""] => {}
            [first, ..] if first.starts_with(['#', '%']) => {}
            [first, second, ..] if !first.is_empty() && !second.is_empty() => {
                rows.push((Box::from(*first), Box::from(*second)));
            }
            _ => gapped += 1,
        }
    }
    if gapped > 0 {
        log::warn!("{path}: skipped {gapped} row(s) with an empty or missing first or second cell");
    }
    Ok(rows)
}

/// Parse a marker TSV/CSV into `(gene, celltype)` pairs ([`read_two_field_rows`], a
/// `gene`/`symbol` header row skipped); type names are [`type_key`]s.
pub(super) fn read_marker_tsv(path: &str) -> Result<Vec<(Box<str>, Box<str>)>> {
    let out: Vec<(Box<str>, Box<str>)> = read_two_field_rows(path)
        .with_context(|| format!("reading markers {path}"))?
        .into_iter()
        .filter(|(gene, _)| {
            !["gene", "symbol"]
                .iter()
                .any(|h| gene.eq_ignore_ascii_case(h))
        })
        .map(|(gene, label)| (gene, type_key(&label)))
        .collect();
    anyhow::ensure!(!out.is_empty(), "no marker pairs parsed from {path}");
    Ok(out)
}

/// Read the curated `label<TAB>CL:id` map ([`read_two_field_rows`], rows whose id is not a
/// `CL:` id skipped as a header), keyed by [`type_key`] like the panel's type names. Two ids
/// for one key is an error.
pub(super) fn read_label_map(path: &str) -> Result<FxHashMap<Box<str>, Box<str>>> {
    use std::collections::hash_map::Entry;
    let mut map: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for (label, id) in read_two_field_rows(path)
        .with_context(|| format!("reading label→id map {path}"))?
        .into_iter()
        .filter(|(_, id)| id.starts_with("CL:"))
    {
        match map.entry(type_key(&label)) {
            Entry::Vacant(e) => {
                e.insert(id);
            }
            Entry::Occupied(e) => anyhow::ensure!(
                *e.get() == id,
                "label {label:?} (as {:?}) has two ids in {path}: {} and {id}",
                e.key(),
                e.get()
            ),
        }
    }
    anyhow::ensure!(!map.is_empty(), "no `label<TAB>id` rows parsed from {path}");
    Ok(map)
}

/// Union of every type's marker gene indices (sorted, unique, in range) — the
/// universe the label-shuffle null samples from.
pub(super) fn marker_gene_pool(type_markers: &[Vec<(u32, f32)>], n_features: usize) -> Vec<u32> {
    let mut set: FxHashSet<u32> = FxHashSet::default();
    for markers in type_markers {
        for &(g, _) in markers {
            if (g as usize) < n_features {
                set.insert(g);
            }
        }
    }
    let mut pool: Vec<u32> = set.into_iter().collect();
    pool.sort_unstable();
    pool
}
