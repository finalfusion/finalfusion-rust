//! A library for reading, writing, and using word embeddings.
//!
//! finalfusion allows you to read, write, and use
//! word2vec/[GloVe](https://nlp.stanford.edu/projects/glove/)
//! embeddings and read [fastText](https://fasttext.cc/) embeddings.
//! finalfusion uses *finalfusion* as its native data format, which
//! has several benefits over the word2vec, GloVe, and fastText
//! formats.
//!
//! ## Reading finalfusion embeddings
//!
//! finalfusion embeddings can be read with the `read_embeddings`
//! method, which expects a reader that implements the `BufRead`
//! trait.
//!
//! Since finalfusion supports various types of vocabularies and
//! embedding matrix (storage) formats, these should be specified
//! as type parameters of the `Embeddings` type. However, typically
//! one would want to read finalfusion embeddings with any type of
//! vocabulary or embedding matrix. For this purpose, the `VocabWrap`
//! and `StorageWrap` types are provided, which wrap any type of
//! vocabulary and embeddung matrix.
//!
//! We can thus load a finalfusion format and retrieve an embedding
//! as follows:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use finalfusion::prelude::*;
//!
//! let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
//!
//! // Read the embeddings.
//! let embeddings: Embeddings<VocabWrap, StorageWrap> =
//!     Embeddings::read_embeddings(&mut reader)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("Berlin");
//! ```
//!
//! For performing analogy/similarity queries on the embedding
//! matrix, we need an embedding matrix which can act as a view.
//! In that case one should use `StorageViewWrap` in place of
//! `StorageWrap`. `StorageViewWrap` is only supported for a
//! subset of embedding matrix types -- in particular, quantized
//! matrices cannot be used as a view.
//!
//! ## Reading other embedding formats
//!
//! Consult the documentation of the `fasttext`, `text` and
//! `word2vec` modules for information on how to read fastText,
//! GloVe, and word2vec embeddings.

mod chunks;
pub use chunks::{metadata, norms, storage, vocab};

pub mod compat;

pub mod embeddings;

pub mod error;

pub mod io;

pub mod prelude;

pub mod similarity;

pub mod subword;

pub(crate) mod util;
