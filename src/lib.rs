mod embeddings;
pub use crate::embeddings::{Builder, BuilderError, Embeddings, Iter};

pub mod similarity;

pub mod text;

pub mod word2vec;

#[cfg(test)]
mod tests;
