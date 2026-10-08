//! Hard gene→module partition and each unit's view through it.

use super::units::UnitTable;

/// One module per gene (feature row). `module_of` is indexed by gene id,
/// `members[m]` lists that module's gene ids in ascending order.
pub struct Partition {
    pub module_of: Vec<u32>,
    pub members: Vec<Vec<u32>>,
}

/// Hard labels from a soft membership `[D × M]`: the argmax column per row
/// (ties → the lowest index), `0` for an all-zero row.
#[must_use]
pub fn labels_from_membership(pi: &nalgebra::DMatrix<f32>) -> Vec<u32> {
    pi.row_iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..row.ncols() {
                let v = row[j];
                if v > best_val {
                    best_val = v;
                    best = j;
                }
            }
            best as u32
        })
        .collect()
}

impl Partition {
    /// `labels` is one module per GENE (`labels.len() == n_genes`).
    pub fn from_labels(labels: &[u32], n_modules: usize) -> Self {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_modules];
        for (g, &m) in labels.iter().enumerate() {
            members[m as usize].push(g as u32);
        }
        for m in &mut members {
            m.sort_unstable();
        }
        Self {
            module_of: labels.to_vec(),
            members,
        }
    }

    pub fn n_modules(&self) -> usize {
        self.members.len()
    }

    pub fn slot_of(&self) -> Vec<u32> {
        let mut slot = vec![0u32; self.module_of.len()];
        for m in &self.members {
            for (s, &g) in m.iter().enumerate() {
                slot[g as usize] = s as u32;
            }
        }
        slot
    }
}

/// One unit's buckets: per module, that unit's `(slot, count)` pairs in it.
pub type UnitBuckets = Vec<(u32, Vec<(u32, f32)>)>;

/// Each unit's view through the gene partition.
///
/// Invariants: `q` and `n_um` are `[n_units × n_modules]`, indexed by
/// [`UnitModules::idx`]; `q[idx(u, ·)]` sums to 1 when unit `u` has any counts
/// and is all-zero otherwise; `by_module[u]` is sorted by module and holds only
/// the modules the unit has counts in.
pub struct UnitModules {
    pub n_modules: usize,
    /// `[n_units × n_modules]`, index `u*M + m`.
    pub q: Vec<f32>,
    pub n_um: Vec<f32>,
    /// Per unit, sorted by module: that unit's (slot, count) pairs in the
    /// module. `slot` is the gene's position in `Partition::members[m]`.
    pub by_module: Vec<UnitBuckets>,
}

impl UnitModules {
    /// Flat index of `(unit, module)` into [`Self::q`] / [`Self::n_um`].
    #[must_use]
    pub fn idx(&self, u: usize, m: usize) -> usize {
        u * self.n_modules + m
    }

    /// The `(slot, count)` pairs of unit `u` in module `m`; empty when the
    /// unit has no counts there. `by_module[u]` is sorted by module, so this
    /// is a binary search.
    #[must_use]
    pub fn counts_of(&self, u: usize, m: usize) -> &[(u32, f32)] {
        match self.by_module[u].binary_search_by_key(&(m as u32), |(k, _)| *k) {
            Ok(i) => self.by_module[u][i].1.as_slice(),
            Err(_) => &[],
        }
    }

    pub fn new(units: &UnitTable, part: &Partition) -> Self {
        let (n_u, m) = (units.n_units(), part.n_modules());
        let slot = part.slot_of();
        let mut n_um = vec![0f32; n_u * m];
        let mut by_module: Vec<UnitBuckets> = Vec::with_capacity(n_u);
        // One bucket per module, indexed directly, so the kept entries come out
        // sorted by module. Within a bucket `feats` is ascending, so the slots
        // come out ascending too.
        let mut buckets: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
        for u in 0..n_u {
            for (&g, &c) in units.feats[u].iter().zip(&units.counts[u]) {
                let g = g as usize;
                let mm = part.module_of[g] as usize;
                n_um[u * m + mm] += c;
                buckets[mm].push((slot[g], c));
            }
            by_module.push(
                buckets
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(k, v)| (k as u32, std::mem::take(v)))
                    .collect(),
            );
        }
        let mut q = vec![0f32; n_u * m];
        for u in 0..n_u {
            let tot = units.total[u];
            for k in 0..m {
                let i = u * m + k;
                q[i] = if tot > 0.0 { n_um[i] / tot } else { 0.0 };
            }
        }
        Self {
            n_modules: m,
            q,
            n_um,
            by_module,
        }
    }
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod partition_tests;
