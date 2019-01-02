extern crate byteorder;

#[macro_use]
extern crate failure;

extern crate itertools;

#[macro_use]
extern crate ndarray;

extern crate ordered_float;

mod embeddings;
pub use embeddings::{Builder, BuilderError, Embeddings, Iter};

pub mod similarity;

mod text;
pub use text::{ReadText, WriteText};

mod word2vec;
pub use word2vec::{ReadWord2Vec, WriteWord2Vec};

#[cfg(test)]
mod tests;
