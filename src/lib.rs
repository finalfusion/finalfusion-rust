mod embeddings;
pub use crate::embeddings::{Embeddings, Iter};

pub mod io;

pub mod similarity;

pub mod storage;

pub mod text;

pub mod vocab;

pub mod word2vec;

#[cfg(test)]
mod tests;
