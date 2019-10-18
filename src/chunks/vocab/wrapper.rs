use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::vocab::subword::{
    BucketSubwordVocab, ExplicitSubwordVocab, FastTextSubwordVocab,
};
use crate::chunks::vocab::{SimpleVocab, SubwordVocab, Vocab, WordIndex};
use crate::io::{Error, ErrorKind, Result};

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
#[derive(Clone, Debug)]
pub enum VocabWrap {
    SimpleVocab(SimpleVocab),
    FinalfusionNGramVocab(ExplicitSubwordVocab),
    FastTextSubwordVocab(FastTextSubwordVocab),
    FinalfusionSubwordVocab(BucketSubwordVocab),
}

impl Vocab for VocabWrap {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.idx(word),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.idx(word),
            VocabWrap::FastTextSubwordVocab(inner) => inner.idx(word),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.idx(word),
        }
    }

    /// Get the vocabulary size.
    fn words_len(&self) -> usize {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.words_len(),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.words_len(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.words_len(),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.words_len(),
        }
    }

    fn vocab_len(&self) -> usize {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.vocab_len(),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.vocab_len(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.vocab_len(),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.vocab_len(),
        }
    }

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String] {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.words(),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.words(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.words(),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.words(),
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

impl From<BucketSubwordVocab> for VocabWrap {
    fn from(v: BucketSubwordVocab) -> Self {
        VocabWrap::FinalfusionSubwordVocab(v)
    }
}

impl From<ExplicitSubwordVocab> for VocabWrap {
    fn from(v: ExplicitSubwordVocab) -> Self {
        VocabWrap::FinalfusionNGramVocab(v)
    }
}

impl ReadChunk for VocabWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read
            .seek(SeekFrom::Current(0))
            .map_err(|e| ErrorKind::io_error("Cannot get vocabulary chunk start position", e))?;
        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
            .map_err(Error::from)?;

        read.seek(SeekFrom::Start(chunk_start_pos)).map_err(|e| {
            ErrorKind::io_error("Cannot seek to vocabulary chunk start position", e)
        })?;

        match chunk_id {
            ChunkIdentifier::SimpleVocab => {
                SimpleVocab::read_chunk(read).map(VocabWrap::SimpleVocab)
            }
            ChunkIdentifier::FastTextSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::FastTextSubwordVocab)
            }
            ChunkIdentifier::FinalfusionSubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::FinalfusionSubwordVocab)
            }
            ChunkIdentifier::FinalfusionNGramVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::FinalfusionNGramVocab)
            }
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected one of: {}, {}, {} or {}, got: {}",
                ChunkIdentifier::SimpleVocab,
                ChunkIdentifier::FinalfusionNGramVocab,
                ChunkIdentifier::FastTextSubwordVocab,
                ChunkIdentifier::FinalfusionSubwordVocab,
                chunk_id
            ))
            .into()),
        }
    }
}

impl WriteChunk for VocabWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.chunk_identifier(),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.chunk_identifier(),
            VocabWrap::FastTextSubwordVocab(inner) => inner.chunk_identifier(),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.write_chunk(write),
            VocabWrap::FinalfusionNGramVocab(inner) => inner.write_chunk(write),
            VocabWrap::FastTextSubwordVocab(inner) => inner.write_chunk(write),
            VocabWrap::FinalfusionSubwordVocab(inner) => inner.write_chunk(write),
        }
    }
}
