use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::vocab::subword::{
    BucketSubwordVocab, ExplicitSubwordVocab, FastTextSubwordVocab,
};
use crate::chunks::vocab::{SimpleVocab, SubwordVocab, Vocab, WordIndex};
use crate::error::{Error, Result};
use crate::vocab::FloretSubwordVocab;

/// Vocabulary types wrapper.
///
/// This crate makes it possible to create fine-grained embedding
/// types, such as `Embeddings<SimpleVocab, NdArray>` or
/// `Embeddings<SubwordVocab, QuantizedArray>`. However, in some cases
/// it is more pleasant to have a single type that covers all
/// vocabulary and storage types. `VocabWrap` and `StorageWrap` wrap
/// all the vocabularies and storage types known to this crate such
/// that the type `Embeddings<VocabWrap, StorageWrap>` covers all
/// variations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VocabWrap {
    SimpleVocab(SimpleVocab),
    ExplicitSubwordVocab(ExplicitSubwordVocab),
    FastTextSubwordVocab(FastTextSubwordVocab),
    FloretSubwordVocab(FloretSubwordVocab),
    BucketSubwordVocab(BucketSubwordVocab),
}

impl Vocab for VocabWrap {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.idx(word),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.idx(word),
            VocabWrap::FastTextSubwordVocab(inner) => inner.idx(word),
            VocabWrap::FloretSubwordVocab(inner) => inner.idx(word),
            VocabWrap::BucketSubwordVocab(inner) => inner.idx(word),
        }
    }

    /// Get the vocabulary size.
    fn words_len(&self) -> usize {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.words_len(),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.words_len(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.words_len(),
            VocabWrap::FloretSubwordVocab(inner) => inner.words_len(),
            VocabWrap::BucketSubwordVocab(inner) => inner.words_len(),
        }
    }

    fn vocab_len(&self) -> usize {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.vocab_len(),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.vocab_len(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.vocab_len(),
            VocabWrap::FloretSubwordVocab(inner) => inner.vocab_len(),
            VocabWrap::BucketSubwordVocab(inner) => inner.vocab_len(),
        }
    }

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String] {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.words(),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.words(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.words(),
            VocabWrap::FloretSubwordVocab(inner) => inner.words(),
            VocabWrap::BucketSubwordVocab(inner) => inner.words(),
        }
    }
}

impl From<SimpleVocab> for VocabWrap {
    fn from(v: SimpleVocab) -> Self {
        VocabWrap::SimpleVocab(v)
    }
}

impl From<FastTextSubwordVocab> for VocabWrap {
    fn from(v: FastTextSubwordVocab) -> Self {
        VocabWrap::FastTextSubwordVocab(v)
    }
}

impl From<FloretSubwordVocab> for VocabWrap {
    fn from(v: FloretSubwordVocab) -> Self {
        VocabWrap::FloretSubwordVocab(v)
    }
}

impl From<BucketSubwordVocab> for VocabWrap {
    fn from(v: BucketSubwordVocab) -> Self {
        VocabWrap::BucketSubwordVocab(v)
    }
}

impl From<ExplicitSubwordVocab> for VocabWrap {
    fn from(v: ExplicitSubwordVocab) -> Self {
        VocabWrap::ExplicitSubwordVocab(v)
    }
}

impl ReadChunk for VocabWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read
            .stream_position()
            .map_err(|e| Error::read_error("Cannot get vocabulary chunk start position", e))?;
        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read vocabulary chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| Error::read_error("Cannot seek to vocabulary chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::SimpleVocab => {
                SimpleVocab::read_chunk(read).map(VocabWrap::SimpleVocab)
            }
            ChunkIdentifier::FastTextSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::FastTextSubwordVocab)
            }
            ChunkIdentifier::BucketSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::BucketSubwordVocab)
            }
            ChunkIdentifier::ExplicitSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::ExplicitSubwordVocab)
            }
            ChunkIdentifier::FloretSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::FloretSubwordVocab)
            }
            _ => Err(Error::Format(format!(
                "Invalid chunk identifier, expected one of: {}, {}, {}, {} or {}, got: {}",
                ChunkIdentifier::SimpleVocab,
                ChunkIdentifier::ExplicitSubwordVocab,
                ChunkIdentifier::FastTextSubwordVocab,
                ChunkIdentifier::FloretSubwordVocab,
                ChunkIdentifier::BucketSubwordVocab,
                chunk_id
            ))),
        }
    }
}

impl WriteChunk for VocabWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.chunk_identifier(),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.chunk_identifier(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.chunk_identifier(),
            VocabWrap::FloretSubwordVocab(inner) => inner.chunk_identifier(),
            VocabWrap::BucketSubwordVocab(inner) => inner.chunk_identifier(),
        }
    }

    fn chunk_len(&self, offset: u64) -> u64 {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.chunk_len(offset),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.chunk_len(offset),
            VocabWrap::FastTextSubwordVocab(inner) => inner.chunk_len(offset),
            VocabWrap::FloretSubwordVocab(inner) => inner.chunk_len(offset),
            VocabWrap::BucketSubwordVocab(inner) => inner.chunk_len(offset),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.write_chunk(write),
            VocabWrap::ExplicitSubwordVocab(inner) => inner.write_chunk(write),
            VocabWrap::FastTextSubwordVocab(inner) => inner.write_chunk(write),
            VocabWrap::FloretSubwordVocab(inner) => inner.write_chunk(write),
            VocabWrap::BucketSubwordVocab(inner) => inner.write_chunk(write),
        }
    }
}
