//! Embedding vocabularies

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::io::private::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::io::{Error, ErrorKind, Result};
use crate::subword::{FinalfusionHashIndexer, SubwordIndices};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Index of a vocabulary word.
pub enum WordIndex {
    /// The index of an in-vocabulary word.
    Word(usize),

    /// The subword indices of out-of-vocabulary words.
    Subword(Vec<usize>),
}

/// Vocabulary without subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl SimpleVocab {
    pub fn new(words: impl Into<Vec<String>>) -> Self {
        let words = words.into();
        let indices = create_indices(&words);
        SimpleVocab { words, indices }
    }
}

impl ReadChunk for SimpleVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::SimpleVocab)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary chunk length", e))?;

        let vocab_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary length", e))?
            as usize;
        let mut words = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let word_len = read
                .read_u32::<LittleEndian>()
                .map_err(|e| ErrorKind::io_error("Cannot read token length", e))?
                as usize;
            let mut bytes = vec![0; word_len];
            read.read_exact(&mut bytes)
                .map_err(|e| ErrorKind::io_error("Cannot read token", e))?;
            let word = String::from_utf8(bytes)
                .map_err(|e| ErrorKind::Format(format!("Token contains invalid UTF-8: {}", e)))
                .map_err(Error::from)?;
            words.push(word);
        }

        Ok(SimpleVocab::new(words))
    }
}

impl WriteChunk for SimpleVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::SimpleVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        // Chunk size: vocabulary size (u64), for each word:
        // word length in bytes (4 bytes), word bytes (variable-length).
        let chunk_len = size_of::<u64>()
            + self
                .words
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(ChunkIdentifier::SimpleVocab as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary chunk identifier", e))?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary length", e))?;

        for word in &self.words {
            write
                .write_u32::<LittleEndian>(word.len() as u32)
                .map_err(|e| ErrorKind::io_error("Cannot write token length", e))?;
            write
                .write_all(word.as_bytes())
                .map_err(|e| ErrorKind::io_error("Cannot write token", e))?;
        }

        Ok(())
    }
}

/// Vocabulary with subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubwordVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
    min_n: u32,
    max_n: u32,
    indexer: FinalfusionHashIndexer,
    buckets_exp: u32,
}

impl SubwordVocab {
    const BOW: char = '<';
    const EOW: char = '>';

    pub fn new(words: impl Into<Vec<String>>, min_n: u32, max_n: u32, buckets_exp: u32) -> Self {
        let words = words.into();
        let indices = create_indices(&words);

        SubwordVocab {
            indices,
            words,
            min_n,
            max_n,
            indexer: FinalfusionHashIndexer::new(buckets_exp as usize),
            buckets_exp,
        }
    }

    fn bracket(word: impl AsRef<str>) -> String {
        let mut bracketed = String::new();
        bracketed.push(Self::BOW);
        bracketed.push_str(word.as_ref());
        bracketed.push(Self::EOW);

        bracketed
    }

    /// Get the subword indices of a token.
    ///
    /// Returns `None` when the model does not support subwords or
    /// when no subwords could be extracted.
    fn subword_indices(&self, word: &str) -> Option<Vec<usize>> {
        let indices = Self::bracket(word)
            .as_str()
            .subword_indices(self.min_n as usize, self.max_n as usize, &self.indexer)
            .into_iter()
            .map(|idx| idx as usize + self.len())
            .collect::<Vec<_>>();
        if indices.is_empty() {
            None
        } else {
            Some(indices)
        }
    }
}

impl ReadChunk for SubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::SubwordVocab)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary chunk length", e))?;

        let vocab_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary length", e))?
            as usize;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read maximum n-gram length", e))?;
        let buckets_exp = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot bucket exponent", e))?;

        let mut words = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let word_len = read
                .read_u32::<LittleEndian>()
                .map_err(|e| ErrorKind::io_error("Cannot read token length", e))?
                as usize;
            let mut bytes = vec![0; word_len];
            read.read_exact(&mut bytes)
                .map_err(|e| ErrorKind::io_error("Cannot read token", e))?;
            let word = String::from_utf8(bytes)
                .map_err(|e| ErrorKind::Format(format!("Token contains invalid UTF-8: {}", e)))
                .map_err(Error::from)?;
            words.push(word);
        }

        Ok(SubwordVocab::new(words, min_n, max_n, buckets_exp))
    }
}

impl WriteChunk for SubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::SubwordVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        // Chunk size: vocab size (u64), minimum n-gram length (u32),
        // maximum n-gram length (u32), bucket exponent (u32), for
        // each word: word length in bytes (u32), word bytes
        // (variable-length).
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + self
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(ChunkIdentifier::SubwordVocab as u32)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write subword vocabulary chunk identifier", e)
            })?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| ErrorKind::io_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| ErrorKind::io_error("Cannot write maximum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.buckets_exp)
            .map_err(|e| ErrorKind::io_error("Cannot write bucket exponent", e))?;

        for word in self.words() {
            write
                .write_u32::<LittleEndian>(word.len() as u32)
                .map_err(|e| ErrorKind::io_error("Cannot write token length", e))?;
            write
                .write_all(word.as_bytes())
                .map_err(|e| ErrorKind::io_error("Cannot write token", e))?;
        }

        Ok(())
    }
}

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
    SubwordVocab(SubwordVocab),
}

impl From<SimpleVocab> for VocabWrap {
    fn from(v: SimpleVocab) -> Self {
        VocabWrap::SimpleVocab(v)
    }
}

impl From<SubwordVocab> for VocabWrap {
    fn from(v: SubwordVocab) -> Self {
        VocabWrap::SubwordVocab(v)
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
            ChunkIdentifier::SubwordVocab => {
                SubwordVocab::read_chunk(read).map(VocabWrap::SubwordVocab)
            }
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected one of: {} or {}, got: {}",
                ChunkIdentifier::SimpleVocab,
                ChunkIdentifier::SubwordVocab,
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
            VocabWrap::SubwordVocab(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.write_chunk(write),
            VocabWrap::SubwordVocab(inner) => inner.write_chunk(write),
        }
    }
}

/// Embedding vocabularies.
#[allow(clippy::len_without_is_empty)]
pub trait Vocab: Clone {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<WordIndex>;

    /// Get the vocabulary size.
    fn len(&self) -> usize;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

impl Vocab for SimpleVocab {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        self.indices.get(word).cloned().map(WordIndex::Word)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl Vocab for SubwordVocab {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        // If the word is known, return its index.
        if let Some(idx) = self.indices.get(word).cloned() {
            return Some(WordIndex::Word(idx));
        }

        // Otherwise, return the subword indices.
        self.subword_indices(word).map(WordIndex::Subword)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl Vocab for VocabWrap {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.idx(word),
            VocabWrap::SubwordVocab(inner) => inner.idx(word),
        }
    }

    /// Get the vocabulary size.
    fn len(&self) -> usize {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.len(),
            VocabWrap::SubwordVocab(inner) => inner.len(),
        }
    }

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String] {
        match self {
            VocabWrap::SimpleVocab(inner) => inner.words(),
            VocabWrap::SubwordVocab(inner) => inner.words(),
        }
    }
}

fn create_indices(words: &[String]) -> HashMap<String, usize> {
    let mut indices = HashMap::new();

    for (idx, word) in words.iter().enumerate() {
        indices.insert(word.to_owned(), idx);
    }

    indices
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};

    use super::{SimpleVocab, SubwordVocab};
    use crate::io::private::{ReadChunk, WriteChunk};

    fn test_simple_vocab() -> SimpleVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];

        SimpleVocab::new(words)
    }

    fn test_subword_vocab() -> SubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        SubwordVocab::new(words, 3, 6, 20)
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[test]
    fn simple_vocab_write_read_roundtrip() {
        let check_vocab = test_simple_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SimpleVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn simple_vocab_correct_chunk_size() {
        let check_vocab = test_simple_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }

    #[test]
    fn subword_vocab_write_read_roundtrip() {
        let check_vocab = test_subword_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn subword_vocab_correct_chunk_size() {
        let check_vocab = test_subword_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }
}
