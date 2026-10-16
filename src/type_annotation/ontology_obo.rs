//! Bridge from the concrete Cell Ontology loader to the generic TreeBH core.
//!
//! The calling/tree/TreeBH math is generic and lives in `enrichment`
//! ([`enrichment::annotate_ontology_core`]) with **no** `data_beans::aux`
//! dependency — ontology access is injected as closures. This module is the one
//! place that binds that core to the concrete `data_beans::aux::ontology::Ontology`
//! OBO loader: load the OBO + the curated `label→CL` map, build the closures,
//! and run the core. Shared by the term-ORA projection path here and by
//! `senna annotate-ontology` / `-by-enrichment` (which delegate to it), so the
//! OBO glue is written once.

use super::markers::{split_tab_first, type_key};
use anyhow::{ensure, Context, Result};
use data_beans::aux::ontology::Ontology;
use enrichment::{annotate_ontology_core, Mat, OntologyAccess, OntologyParams, OntologyScore};
use log::info;
use rustc_hash::FxHashMap;

/// Read the curated `label<TAB>CL:id` map, keyed the way panel type names are: lines split by
/// [`split_tab_first`], labels mapped by [`type_key`]. Blank lines, `#` comments and rows whose
/// id is not a `CL:` id (a header) are skipped; one label with two ids is an error.
pub(super) fn read_label_map(path: &str) -> Result<FxHashMap<Box<str>, Box<str>>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading label→id map {path}"))?;
    let mut map: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = split_tab_first(line);
        let (Some(&label), Some(&id)) = (fields.first(), fields.get(1)) else {
            continue;
        };
        if label.is_empty() || !id.starts_with("CL:") {
            continue;
        }
        let key = type_key(label);
        if let Some(prev) = map.insert(key.clone(), Box::from(id)) {
            ensure!(
                prev.as_ref() == id,
                "label {key:?} mapped to two different ids in {path}: {prev} vs {id}"
            );
        }
    }
    ensure!(!map.is_empty(), "no `label<TAB>id` rows parsed from {path}");
    Ok(map)
}

/// Load the Cell Ontology + curated `label<TAB>CL:id` map, inject them into the
/// generic TreeBH core as closures, and run it. Returns the
/// `(ontology_assignment.tsv, ontology_node_mass.parquet)` paths.
#[allow(clippy::too_many_arguments)]
pub fn annotate_ontology_from_obo(
    out: &str,
    label_cl: &str,
    obo: &str,
    fdr_q: f64,
    by: bool,
    score: OntologyScore<'_>,
    q: Option<&Mat>,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<(String, String)> {
    let label_to_id = read_label_map(label_cl)?;
    let onto = Ontology::load_obo(obo)?;
    info!("loaded Cell Ontology: {} terms from {obo}", onto.len());

    // Bind the closures to locals so the borrows in `OntologyAccess` outlive the
    // call.
    let ancestors_or_self = |id: &str| onto.ancestors_or_self(id);
    let name_of = |id: &str| onto.name(id).map(Box::from);
    let contains = |id: &str| onto.contains(id);
    let access = OntologyAccess {
        ancestors_or_self: &ancestors_or_self,
        name_of: &name_of,
        contains: &contains,
    };

    annotate_ontology_core(
        &OntologyParams { out, fdr_q, by },
        score,
        q,
        &label_to_id,
        &access,
        cluster_names,
        celltype_names,
    )
}
