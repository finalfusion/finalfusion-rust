//! A library for reading, writing, and using word embeddings.
//!
//! rust2vec allows you to read, write, and use word2vec and GloVe
//! embeddings.  rust2vec also provides its own data format that has
//! various benefits over existing formats.

mod embeddings;
pub use crate::embeddings::{Embeddings, Iter};

pub mod io;

pub mod similarity;

pub mod storage;

pub(crate) mod subword;

pub mod text;

pub(crate) mod util;

pub mod vocab;

pub mod word2vec;

#[cfg(test)]
mod tests;
