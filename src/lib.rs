extern crate byteorder;

#[macro_use]
extern crate error_chain;

#[macro_use]
extern crate itertools;

#[macro_use(s)]
extern crate ndarray;

macro_rules! try_opt {
    ($expr:expr) => (match $expr {
        Some(val) => val,
        None      => return None
    })
}

mod embeddings;
pub use embeddings::{Embeddings, WordSimilarity};

mod error;
pub use error::*;

mod text;
pub use text::{ReadText, WriteText};

mod word2vec;
pub use word2vec::{ReadWord2Vec, WriteWord2Vec};

#[cfg(test)]
mod tests;
