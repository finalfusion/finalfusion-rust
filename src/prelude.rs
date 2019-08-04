//! Prelude exports the most commonly-used types and traits.

pub use crate::chunks::metadata::Metadata;

pub use crate::chunks::storage::{
    MmapArray, NdArray, QuantizedArray, Storage, StorageView, StorageViewWrap, StorageWrap,
};

pub use crate::chunks::vocab::{SimpleVocab, SubwordVocab, Vocab, VocabWrap};

pub use crate::compat::fasttext::ReadFastText;

pub use crate::compat::text::{ReadText, ReadTextDims, WriteText, WriteTextDims};

pub use crate::compat::word2vec::{ReadWord2Vec, WriteWord2Vec};

pub use crate::embeddings::{Embeddings, Quantize};

pub use crate::io::{MmapEmbeddings, ReadEmbeddings, ReadMetadata, WriteEmbeddings};
