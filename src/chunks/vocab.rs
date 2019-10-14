//! Embedding vocabularies

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use super::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::compat::fasttext::FastTextIndexer;
use crate::io::{Error, ErrorKind, Result};
use crate::subword::{
    BucketIndexer, FinalfusionHashIndexer, Indexer, NGramIndexer,
    SubwordIndices as StrSubwordIndices,
};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Index of a vocabulary word.
pub enum WordIndex {
    /// The index of an in-vocabulary word.
    Word(usize),

    /// The subword indices of out-of-vocabulary words.
    Subword(Vec<usize>),
}

impl WordIndex {
    pub fn word(&self) -> Option<usize> {
        use WordIndex::*;

        match self {
            Word(idx) => Some(*idx),
            Subword(_) => None,
        }
    }

    pub fn subword(&self) -> Option<&[usize]> {
        use WordIndex::*;

        match self {
            Word(_) => None,
            Subword(indices) => Some(indices),
        }
    }
}

/// Vocabulary without subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl SimpleVocab {
    /// Construct a new simple vocabulary.
    ///
    /// Words are assigned indices in the given order.
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

        let words = read_vocab_items(read, vocab_len)?;

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

        write_vocab_items(write, self.words())?;

        Ok(())
    }
}

/// fastText subword vocabulary.
pub type FastTextSubwordVocab = SubwordVocab<FastTextIndexer>;

/// Native finalfusion subword vocabulary.
pub type FinalfusionSubwordVocab = SubwordVocab<FinalfusionHashIndexer>;

/// finalfusion NGram vocabulary.
pub type FinalfusionNGramVocab = SubwordVocab<NGramIndexer>;

/// Vocabulary with subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubwordVocab<I> {
    indexer: I,
    indices: HashMap<String, usize>,
    words: Vec<String>,
    min_n: u32,
    max_n: u32,
}

impl<I> SubwordVocab<I>
where
    I: Clone + Indexer,
{
    const BOW: char = '<';
    const EOW: char = '>';

    /// Construct a new `SubwordVocab`.
    ///
    /// Words are assigned indices in the given order. NGrams in range `(min_n..max_n)` are
    /// considered. The `indexer` is used to look up indices for the NGrams produced by this
    /// `SubwordVocab`.
    pub fn new(words: impl Into<Vec<String>>, min_n: u32, max_n: u32, indexer: I) -> Self {
        let words = words.into();
        let indices = create_indices(&words);

        SubwordVocab {
            indices,
            words,
            min_n,
            max_n,
            indexer,
        }
    }

    /// Get the vocab's indexer.
    pub fn indexer(&self) -> &I {
        &self.indexer
    }

    /// Get the lower bound of the generated ngram lengths.
    pub fn min_n(&self) -> u32 {
        self.min_n
    }

    /// Get the upper bound of the generated ngram lengths.
    pub fn max_n(&self) -> u32 {
        self.max_n
    }

    fn bracket(word: impl AsRef<str>) -> String {
        let mut bracketed = String::new();
        bracketed.push(Self::BOW);
        bracketed.push_str(word.as_ref());
        bracketed.push(Self::EOW);

        bracketed
    }
}

impl ReadChunk for FastTextSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_bucketed_chunk(read, ChunkIdentifier::FastTextSubwordVocab)
    }
}

impl ReadChunk for FinalfusionSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_bucketed_chunk(read, ChunkIdentifier::FinalfusionSubwordVocab)
    }
}

impl ReadChunk for FinalfusionNGramVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_ngram_chunk(read, ChunkIdentifier::FinalfusionNGramVocab)
    }
}

impl WriteChunk for FastTextSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::FastTextSubwordVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_bucketed_chunk(write, self.chunk_identifier())
    }
}

impl WriteChunk for FinalfusionSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::FinalfusionSubwordVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_bucketed_chunk(write, self.chunk_identifier())
    }
}

impl WriteChunk for FinalfusionNGramVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::FinalfusionNGramVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_ngram_chunk(write, self.chunk_identifier())
    }
}

impl<I> SubwordVocab<I>
where
    I: BucketIndexer + Clone,
{
    fn read_bucketed_chunk<R>(
        read: &mut R,
        chunk_identifier: ChunkIdentifier,
    ) -> Result<SubwordVocab<I>>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, chunk_identifier)?;

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
        let buckets = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of buckets", e))?;

        let words = read_vocab_items(read, vocab_len as usize)?;

        Ok(SubwordVocab::new(
            words,
            min_n,
            max_n,
            I::new(buckets as usize),
        ))
    }

    fn write_bucketed_chunk<W>(
        &self,
        write: &mut W,
        chunk_identifier: ChunkIdentifier,
    ) -> Result<()>
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
            .write_u32::<LittleEndian>(chunk_identifier as u32)
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
            .write_u32::<LittleEndian>(self.indexer.buckets() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write number of buckets", e))?;

        write_vocab_items(write, self.words())?;

        Ok(())
    }
}

impl SubwordVocab<NGramIndexer> {
    fn read_ngram_chunk<R>(
        read: &mut R,
        chunk_identifier: ChunkIdentifier,
    ) -> Result<SubwordVocab<NGramIndexer>>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, chunk_identifier)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary chunk length", e))?;
        let words_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of words", e))?;
        let ngrams_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of ngrams", e))?;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read maximum n-gram length", e))?;

        let words = read_vocab_items(read, words_len as usize)?;
        let ngrams = read_vocab_items(read, ngrams_len as usize)?;
        let indexer = NGramIndexer::new(ngrams);
        Ok(SubwordVocab::new(words, min_n, max_n, indexer))
    }

    fn write_ngram_chunk<W>(&self, write: &mut W, chunk_identifier: ChunkIdentifier) -> Result<()>
    where
        W: Write + Seek,
    {
        // Chunk size: word vocab size (u64), ngram vocab size (u64)
        // minimum n-gram length (u32), maximum n-gram length (u32),
        // for each word and ngram:
        // length in bytes (u32), number of bytes (variable-length).
        let chunk_len = size_of::<u64>()
            + size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + self
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>()
            + self
                .indexer
                .ngrams()
                .iter()
                .map(|ngram| ngram.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(chunk_identifier as u32)
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
            .write_u64::<LittleEndian>(self.indexer.ngrams().len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write ngram length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| ErrorKind::io_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| ErrorKind::io_error("Cannot write maximum n-gram length", e))?;

        write_vocab_items(write, self.words())?;
        write_vocab_items(write, self.indexer.ngrams())?;

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
    FinalfusionNGramVocab(FinalfusionNGramVocab),
    FastTextSubwordVocab(FastTextSubwordVocab),
    FinalfusionSubwordVocab(FinalfusionSubwordVocab),
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

impl From<FinalfusionSubwordVocab> for VocabWrap {
    fn from(v: FinalfusionSubwordVocab) -> Self {
        VocabWrap::FinalfusionSubwordVocab(v)
    }
}

impl From<FinalfusionNGramVocab> for VocabWrap {
    fn from(v: FinalfusionNGramVocab) -> Self {
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

/// Embedding vocabularies.
#[allow(clippy::len_without_is_empty)]
pub trait Vocab: Clone {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<WordIndex>;

    /// Get the number of words in the vocabulary.
    fn words_len(&self) -> usize;

    /// Get the total length of this vocabulary, including possible subword indices.
    fn vocab_len(&self) -> usize;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

impl Vocab for SimpleVocab {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        self.indices.get(word).cloned().map(WordIndex::Word)
    }

    fn words_len(&self) -> usize {
        self.indices.len()
    }

    fn vocab_len(&self) -> usize {
        self.words_len()
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl<I> Vocab for SubwordVocab<I>
where
    I: Clone + Indexer,
{
    fn idx(&self, word: &str) -> Option<WordIndex> {
        // If the word is known, return its index.
        if let Some(idx) = self.indices.get(word).cloned() {
            return Some(WordIndex::Word(idx));
        }

        // Otherwise, return the subword indices.
        self.subword_indices(word).map(WordIndex::Subword)
    }

    fn words_len(&self) -> usize {
        self.indices.len()
    }

    fn vocab_len(&self) -> usize {
        self.words_len() + self.indexer.upper_bound() as usize
    }

    fn words(&self) -> &[String] {
        &self.words
    }
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

/// Get subword indices.
///
/// Get the subword ngrams and their indices of a word in the
/// subword vocabulary.
pub trait NGramIndices {
    /// Return the subword ngrams and their indices of a word,
    /// in the subword vocabulary.
    fn ngram_indices(&self, word: &str) -> Option<Vec<(String, Option<usize>)>>;
}

impl<I> NGramIndices for SubwordVocab<I>
where
    I: Clone + Indexer,
{
    fn ngram_indices(&self, word: &str) -> Option<Vec<(String, Option<usize>)>> {
        let indices = Self::bracket(word)
            .as_str()
            .subword_indices_with_ngrams(self.min_n as usize, self.max_n as usize, &self.indexer)
            .map(|(ngram, idx)| {
                (
                    ngram.to_owned(),
                    idx.map(|idx| idx as usize + self.words_len()),
                )
            })
            .collect::<Vec<_>>();
        if indices.is_empty() {
            None
        } else {
            Some(indices)
        }
    }
}

/// Get subword indices.
///
/// Get the subword indices of a token in the subword vocabulary.
pub trait SubwordIndices {
    /// Return the subword indices of the subwords of a string,
    /// according to the subword vocabulary.
    fn subword_indices(&self, word: &str) -> Option<Vec<usize>>;
}

impl<I> SubwordIndices for SubwordVocab<I>
where
    I: Clone + Indexer,
{
    fn subword_indices(&self, word: &str) -> Option<Vec<usize>> {
        let indices = Self::bracket(word)
            .as_str()
            .subword_indices(self.min_n as usize, self.max_n as usize, &self.indexer)
            .map(|idx| idx as usize + self.words_len())
            .collect::<Vec<_>>();
        if indices.is_empty() {
            None
        } else {
            Some(indices)
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

fn read_vocab_items<R>(read: &mut R, len: usize) -> Result<Vec<String>>
where
    R: Read + Seek,
{
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        let item_len = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read item length", e))?
            as usize;
        let mut bytes = vec![0; item_len];
        read.read_exact(&mut bytes)
            .map_err(|e| ErrorKind::io_error("Cannot read item", e))?;
        let item = String::from_utf8(bytes)
            .map_err(|e| ErrorKind::Format(format!("Item contains invalid UTF-8: {}", e)))
            .map_err(Error::from)?;
        items.push(item);
    }
    Ok(items)
}

fn write_vocab_items<W>(write: &mut W, items: &[String]) -> Result<()>
where
    W: Write + Seek,
{
    for word in items {
        write
            .write_u32::<LittleEndian>(word.len() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write token length", e))?;
        write
            .write_all(word.as_bytes())
            .map_err(|e| ErrorKind::io_error("Cannot write token", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};

    use super::{FastTextSubwordVocab, FinalfusionSubwordVocab, SimpleVocab, SubwordVocab};
    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::vocab::FinalfusionNGramVocab;
    use crate::compat::fasttext::FastTextIndexer;
    use crate::subword::{BucketIndexer, FinalfusionHashIndexer, NGramIndexer};

    fn test_fasttext_subword_vocab() -> FastTextSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let indexer = FastTextIndexer::new(20);
        SubwordVocab::new(words, 3, 6, indexer)
    }

    fn test_simple_vocab() -> SimpleVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];

        SimpleVocab::new(words)
    }

    fn test_subword_vocab() -> FinalfusionSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let indexer = FinalfusionHashIndexer::new(20);
        SubwordVocab::new(words, 3, 6, indexer)
    }

    fn test_ngram_vocab() -> FinalfusionNGramVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];

        let ngrams = vec!["is>".to_owned(), "is".to_owned(), "<t".to_owned()];

        FinalfusionNGramVocab::new(words, 2, 3, NGramIndexer::new(ngrams))
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[test]
    fn fasttext_subword_vocab_write_read_roundtrip() {
        let check_vocab = test_fasttext_subword_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
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

    #[test]
    fn ngram_vocab_write_read_roundtrip() {
        let check_vocab = test_ngram_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = FinalfusionNGramVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn ngram_vocab_correct_chunk_size() {
        let check_vocab = test_ngram_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }
}
