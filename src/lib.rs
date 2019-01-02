mod embeddings;
pub use crate::embeddings::{Builder, BuilderError, Embeddings, Iter};

pub mod similarity;

mod text;
pub use crate::text::{ReadText, WriteText};

mod word2vec;
pub use crate::word2vec::{ReadWord2Vec, WriteWord2Vec};

#[cfg(test)]
mod tests;
