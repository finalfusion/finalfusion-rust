//! A library for reading, writing, and using word embeddings.
//!
//! finalfusion allows you to read, write, and use word2vec and GloVe
//! embeddings. finalfusion uses *finalfusion* as its native data
//! format, which has several benefits over the word2vec and GloVe
//! formats.

pub mod embeddings;

pub mod io;

pub mod metadata;

pub mod prelude;

pub mod similarity;

pub mod storage;

pub(crate) mod subword;

pub mod text;

pub(crate) mod util;

pub mod vocab;

pub mod word2vec;

#[cfg(test)]
mod tests;
