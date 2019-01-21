mod embeddings;
pub use crate::embeddings::{Embeddings, Iter};

pub mod similarity;

pub mod storage;

pub mod text;

pub mod word2vec;

#[cfg(test)]
mod tests;
