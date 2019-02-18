//! Prelude exports the most commonly-used types and traits.

pub use crate::embeddings::Embeddings;

pub use crate::io::{MmapEmbeddings, ReadEmbeddings, WriteEmbeddings};

pub use crate::storage::{
    MmapArray, NdArray, Quantize, QuantizedArray, Storage, StorageView, StorageViewWrap,
    StorageWrap,
};

pub use crate::text::{ReadText, ReadTextDims, WriteText, WriteTextDims};

pub use crate::word2vec::{ReadWord2Vec, WriteWord2Vec};

pub use crate::vocab::{SimpleVocab, SubwordVocab, Vocab, VocabWrap};
