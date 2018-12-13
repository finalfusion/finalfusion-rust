extern crate byteorder;

#[macro_use]
extern crate failure;

extern crate itertools;

extern crate ndarray;

mod embeddings;
pub use embeddings::{Builder, BuilderError, Embeddings, Iter, WordSimilarity};

mod text;
pub use text::{ReadText, WriteText};

mod word2vec;
pub use word2vec::{ReadWord2Vec, WriteWord2Vec};

#[cfg(test)]
mod tests;
