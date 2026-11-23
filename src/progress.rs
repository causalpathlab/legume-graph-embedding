//! Senna-style progress bar for rayon-parallel builds.
//!
//! The implementation lives in [`legume_numeric::matrix::progress`] so the whole
//! workspace shares one style and one `MULTI_PROGRESS` (previously this file
//! held a byte-for-byte copy that was *not* registered with senna's bars). This
//! module is kept as `crate::progress` so existing call sites need no change.

pub use legume_numeric::matrix::progress::new_progress_bar;
