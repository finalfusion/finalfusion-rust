use std::collections::HashMap;
use std::convert::TryFrom;
use std::io;
use std::io::{ErrorKind, Read, Seek, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use smallvec::SmallVec;

use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::vocab::{create_indices, read_vocab_items, write_vocab_items, Vocab, WordIndex};
use crate::compat::fasttext::FastTextIndexer;
use crate::compat::floret::FloretIndexer;
use crate::error::{Error, Result};
use crate::subword::{
    BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer,
    SubwordIndices as StrSubwordIndices,
};
use crate::vocab::{read_string, write_string};

/// fastText vocabulary with hashed n-grams.
pub type FastTextSubwordVocab = SubwordVocab<FastTextIndexer>;

/// floret vocabulary with hashed n-grams.
pub type FloretSubwordVocab = SubwordVocab<FloretIndexer>;

/// finalfusion vocabulary with hashed n-grams.
pub type BucketSubwordVocab = SubwordVocab<FinalfusionHashIndexer>;

/// finalfusion vocabulary with explicit n-grams.
pub type ExplicitSubwordVocab = SubwordVocab<ExplicitIndexer>;

/// Vocabulary with subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubwordVocab<I> {
    bow: String,
    eow: String,
    indexer: I,
    indices: HashMap<String, usize>,
    words: Vec<String>,
    min_n: u32,
    max_n: u32,
}

impl<I> SubwordVocab<I>
where
    I: Indexer,
{
    const DEFAULT_BOW: &'static str = "<";
    const DEFAULT_EOW: &'static str = ">";

    /// Construct a new `SubwordVocab` with default word boundaries.
    ///
    /// Words are assigned indices in the given order. NGrams in range `(min_n..max_n)` are
    /// considered. The `indexer` is used to look up indices for the NGrams produced by this
    /// `SubwordVocab`.
    ///
    /// Panics when there are duplicate words.
    pub fn new(words: impl Into<Vec<String>>, min_n: u32, max_n: u32, indexer: I) -> Self {
        Self::new_with_boundaries(
            words,
            min_n,
            max_n,
            indexer,
            Self::DEFAULT_BOW,
            Self::DEFAULT_EOW,
        )
    }

    /// Construct a new `SubwordVocab`.
    ///
    /// Words are assigned indices in the given order. NGrams in range `(min_n..max_n)` are
    /// considered. The `indexer` is used to look up indices for the NGrams produced by this
    /// `SubwordVocab`. N-grams are extracted from words with the `bow` and `eow` boundaries
    /// added.
    ///
    /// Panics when there are duplicate words.
    pub fn new_with_boundaries(
        words: impl Into<Vec<String>>,
        min_n: u32,
        max_n: u32,
        indexer: I,
        bow: impl Into<String>,
        eow: impl Into<String>,
    ) -> Self {
        let words = words.into();
        let indices = create_indices(&words);
        assert_eq!(
            words.len(),
            indices.len(),
            "words contained duplicate entries."
        );

        // Check that usize can represent the indexer's upper bound.
        assert!(
            usize::try_from(indexer.upper_bound()).is_ok(),
            "The upper bound of the indexer cannot be represented by the native word size."
        );

        // Check that usize can represent the combined vocabulary sizes.
        assert!(
            words
                .len()
                .checked_add(indexer.upper_bound() as usize)
                .is_some(),
            "The vocab + subword vocab size cannot be represented by the native word size"
        );

        SubwordVocab {
            bow: bow.into(),
            eow: eow.into(),
            indexer,
            indices,
            words,
            min_n,
            max_n,
        }
    }

    /// Get begin-of-word marker.
    pub fn bow(&self) -> &str {
        &self.bow
    }

    /// Get end-of-word marker.
    pub fn eow(&self) -> &str {
        &self.eow
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

    fn bracket(&self, word: impl AsRef<str>) -> String {
        let mut bracketed =
            String::with_capacity(word.as_ref().len() + self.bow.len() + self.eow.len());
        bracketed.push_str(&self.bow);
        bracketed.push_str(word.as_ref());
        bracketed.push_str(&self.eow);

        bracketed
    }
}

impl<I> Vocab for SubwordVocab<I>
where
    I: Indexer,
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

/// Get subword indices.
///
/// Get the subword ngrams and their indices of a word in the
/// subword vocabulary.
pub trait NGramIndices {
    /// Return the subword ngrams and their indices of a word,
    /// in the subword vocabulary.
    fn ngram_indices(&self, word: &str) -> Option<Vec<(String, SmallVec<[usize; 4]>)>>;
}

impl<I> NGramIndices for SubwordVocab<I>
where
    I: Indexer,
{
    fn ngram_indices(&self, word: &str) -> Option<Vec<(String, SmallVec<[usize; 4]>)>> {
        let indices = self
            .bracket(word)
            .as_str()
            .subword_indices_with_ngrams(self.min_n as usize, self.max_n as usize, &self.indexer)
            .map(|(ngram, indices)| {
                (
                    ngram.to_owned(),
                    indices
                        .into_iter()
                        .map(|idx| idx as usize + self.words_len())
                        .collect(),
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
    I: Indexer,
{
    fn subword_indices(&self, word: &str) -> Option<Vec<usize>> {
        let word = self.bracket(word);
        let indices = word
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

impl ReadChunk for FastTextSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_bucketed_chunk(read, ChunkIdentifier::FastTextSubwordVocab)
    }
}

impl ReadChunk for BucketSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_bucketed_chunk(read, ChunkIdentifier::BucketSubwordVocab)
    }
}

impl ReadChunk for ExplicitSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_ngram_chunk(read, ChunkIdentifier::ExplicitSubwordVocab)
    }
}

impl ReadChunk for FloretSubwordVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        Self::read_floret_chunk(read, ChunkIdentifier::FloretSubwordVocab)
    }
}

impl WriteChunk for FastTextSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::FastTextSubwordVocab
    }

    fn chunk_len(&self, _offset: u64) -> u64 {
        self.chunk_len_()
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_bucketed_chunk(write, self.chunk_identifier())
    }
}

impl WriteChunk for BucketSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::BucketSubwordVocab
    }

    fn chunk_len(&self, _offset: u64) -> u64 {
        self.chunk_len_()
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_bucketed_chunk(write, self.chunk_identifier())
    }
}

impl WriteChunk for ExplicitSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::ExplicitSubwordVocab
    }

    fn chunk_len(&self, _offset: u64) -> u64 {
        self.chunk_len_()
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_ngram_chunk(write, self.chunk_identifier())
    }
}

impl WriteChunk for FloretSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::FloretSubwordVocab
    }

    fn chunk_len(&self, _offset: u64) -> u64 {
        self.chunk_len_()
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_floret_chunk(write, self.chunk_identifier())
    }
}

impl<I> SubwordVocab<I>
where
    I: BucketIndexer,
{
    fn chunk_len_(&self) -> u64 {
        // Chunk size: chunk identifier (u32) + chunk len (u64) +
        // vocab size (u64) + minimum n-gram length (u32) +
        // maximum n-gram length (u32) + bucket exponent (u32) +
        // for each word: word length in bytes (u32) +
        // word bytes (variable-length).
        (size_of::<u32>()
            + size_of::<u64>()
            + size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + self
                .words()
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>()) as u64
    }

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
            .map_err(|e| Error::read_error("Cannot read vocabulary chunk length", e))?;

        let vocab_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read vocabulary length", e))?
            as usize;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read maximum n-gram length", e))?;
        let buckets = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of buckets", e))?;

        let words = read_vocab_items(read, vocab_len)?;

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
        write
            .write_u32::<LittleEndian>(chunk_identifier as u32)
            .map_err(|e| {
                Error::write_error("Cannot write subword vocabulary chunk identifier", e)
            })?;

        let remaining_chunk_len = self.chunk_len_() - (size_of::<u32>() + size_of::<u64>()) as u64;

        write
            .write_u64::<LittleEndian>(remaining_chunk_len)
            .map_err(|e| Error::write_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| Error::write_error("Cannot write vocabulary length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::write_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::write_error("Cannot write maximum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.indexer.buckets() as u32)
            .map_err(|e| Error::write_error("Cannot write number of buckets", e))?;

        write_vocab_items(write, self.words())?;

        Ok(())
    }

    /// Convert the hash-based vocabulary to an Explicit Vocabulary.
    ///
    /// N-grams in the range `(self.min_n..self.max_n)` are extracted from the words in the
    /// vocabulary, each of these gets assigned an index from the `BucketIndexer` which is used to
    /// determine the index in the explicit subword vocab.
    ///
    /// The second item in the returned tuple holds the `bucket -> explicit_index` mapping for
    /// all buckets hit by `self`.
    pub fn to_explicit(&self) -> Result<(ExplicitSubwordVocab, HashMap<u64, usize>)> {
        let mut ngram_index = HashMap::new();
        let SubwordVocab {
            bow: _,
            eow: _,
            words,
            indices: _,
            indexer,
            min_n,
            max_n,
        } = &self;

        for word in words.iter().map(|word| self.bracket(word)) {
            for (ngram, indices) in
                word.subword_indices_with_ngrams(*min_n as usize, *max_n as usize, indexer)
            {
                if indices.is_empty() {
                    continue;
                }

                if indices.len() > 1 {
                    return Err(Error::ngram_conversion_error("Vocab maps n-gram to multiple indices, cannot be converted to explicit n-gram vocab."));
                }

                ngram_index.entry(ngram.into()).or_insert(indices[0]);
            }
        }

        let (indexer, mapping) = ExplicitIndexer::new_with_indices(ngram_index);
        Ok((
            ExplicitSubwordVocab::new(words.to_owned(), *min_n, *max_n, indexer),
            mapping,
        ))
    }
}

impl ExplicitSubwordVocab {
    fn chunk_len_(&self) -> u64 {
        // Chunk size: chunk identifier (u32) + chunk len (u64) +
        // word vocab size (u64) + ngram vocab size (u64) +
        // minimum n-gram length (u32) + maximum n-gram length (u32) +
        // for each word and ngram: length in bytes (u32) +
        // number of bytes (variable-length) +
        // each ngram is followed by its index (u64)
        (size_of::<u32>()
            + size_of::<u64>()
            + size_of::<u64>()
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
                .map(|ngram| ngram.len() + size_of::<u32>() + size_of::<u64>())
                .sum::<usize>()) as u64
    }

    fn read_ngram_chunk<R>(
        read: &mut R,
        chunk_identifier: ChunkIdentifier,
    ) -> Result<SubwordVocab<ExplicitIndexer>>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, chunk_identifier)?;
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read vocabulary chunk length", e))?;
        let words_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of words", e))?;
        let ngrams_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of ngrams", e))?;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read maximum n-gram length", e))?;

        let words = read_vocab_items(read, words_len as usize)?;
        let ngrams = read_ngrams_with_indices(read, ngrams_len as usize)?;
        let (indexer, _) = ExplicitIndexer::new_with_indices(ngrams);
        Ok(SubwordVocab::new(words, min_n, max_n, indexer))
    }

    fn write_ngram_chunk<W>(&self, write: &mut W, chunk_identifier: ChunkIdentifier) -> Result<()>
    where
        W: Write + Seek,
    {
        let remaining_chunk_len = self.chunk_len_() - (size_of::<u32>() + size_of::<u64>()) as u64;

        write
            .write_u32::<LittleEndian>(chunk_identifier as u32)
            .map_err(|e| {
                Error::write_error("Cannot write subword vocabulary chunk identifier", e)
            })?;
        write
            .write_u64::<LittleEndian>(remaining_chunk_len)
            .map_err(|e| Error::write_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| Error::write_error("Cannot write vocabulary length", e))?;
        write
            .write_u64::<LittleEndian>(self.indexer.ngrams().len() as u64)
            .map_err(|e| Error::write_error("Cannot write ngram length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::write_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::write_error("Cannot write maximum n-gram length", e))?;

        write_vocab_items(write, self.words())?;
        write_ngrams_with_indices(write, self.indexer())?;

        Ok(())
    }
}

impl FloretSubwordVocab {
    fn chunk_len_(&self) -> u64 {
        // Chunk size: chunk identifier (u32) + chunk len (u64) +
        // minimum n-gram length (u32) + maximum n-gram length (u32) +
        // number of buckets (u64) + number of hashes (u32) + hash seed (u32) +
        // bow and row (variable length).
        (size_of::<u32>()
            + size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + self.bow.len()
            + size_of::<u32>()
            + self.eow.len()
            + size_of::<u32>()) as u64
    }

    fn read_floret_chunk<R>(
        read: &mut R,
        chunk_identifier: ChunkIdentifier,
    ) -> Result<SubwordVocab<FloretIndexer>>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, chunk_identifier)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read vocabulary chunk length", e))?;

        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read maximum n-gram length", e))?;
        let n_buckets = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of buckets", e))?;
        let n_hashes: u32 = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of hashes", e))?;
        if !(1..=4).contains(&n_hashes) {
            return Err(Error::read_error(
                format!(
                    "Number of hashes should be in be more than 0 and less than 5, was: {}",
                    n_hashes
                ),
                ErrorKind::InvalidData.into(),
            ));
        }
        let seed: u32 = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read hasher seed", e))?;
        let bow = read_string(read).map_err(|e| e.context("Cannot read begin of word marker"))?;
        let eow = read_string(read).map_err(|e| e.context("Cannot read end of word marker"))?;

        Ok(SubwordVocab::new_with_boundaries(
            Vec::new(),
            min_n,
            max_n,
            FloretIndexer::new(n_buckets, n_hashes, seed),
            bow,
            eow,
        ))
    }

    fn write_floret_chunk<W>(&self, write: &mut W, chunk_identifier: ChunkIdentifier) -> Result<()>
    where
        W: Write + Seek,
    {
        let remaining_chunk_len = self.chunk_len_() - (size_of::<u32>() + size_of::<u64>()) as u64;

        write
            .write_u32::<LittleEndian>(chunk_identifier as u32)
            .map_err(|e| {
                Error::write_error("Cannot write subword vocabulary chunk identifier", e)
            })?;
        write
            .write_u64::<LittleEndian>(remaining_chunk_len)
            .map_err(|e| Error::write_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::write_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::write_error("Cannot write maximum n-gram length", e))?;
        write
            .write_u64::<LittleEndian>(self.indexer.n_buckets())
            .map_err(|e| Error::write_error("Cannot write number of buckets", e))?;
        write
            .write_u32::<LittleEndian>(self.indexer.n_hashes())
            .map_err(|e| Error::write_error("Cannot write number of hashes", e))?;
        write
            .write_u32::<LittleEndian>(self.indexer().seed())
            .map_err(|e| Error::write_error("Cannot write hasher seed", e))?;
        write_string(write, self.bow())
            .map_err(|e| e.context("Cannot write begin of word marker"))?;
        write_string(write, self.eow())
            .map_err(|e| e.context("Cannot write end of word marker"))?;

        Ok(())
    }
}

fn read_ngrams_with_indices<R>(read: &mut R, len: usize) -> Result<Vec<(String, u64)>>
where
    R: Read + Seek,
{
    let mut ngrams = Vec::with_capacity(len);
    for _ in 0..len {
        let ngram_len =
            read.read_u32::<LittleEndian>()
                .map_err(|e| Error::read_error("Cannot read item length", e))? as usize;
        let mut bytes = vec![0; ngram_len];
        read.read_exact(&mut bytes)
            .map_err(|e| Error::read_error("Cannot read item", e))?;
        let item = String::from_utf8(bytes)
            .map_err(|e| Error::Format(format!("Item contains invalid UTF-8: {}", e)))
            .map_err(Error::from)?;
        let idx = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read ngram index.", e))?;
        ngrams.push((item, idx));
    }
    Ok(ngrams)
}

fn write_ngrams_with_indices<W>(write: &mut W, indexer: &ExplicitIndexer) -> Result<()>
where
    W: Write + Seek,
{
    for ngram in indexer.ngrams() {
        let indices = indexer.index_ngram(&ngram.as_str().into());
        if indices.is_empty() {
            return Err(Error::write_error(
                format!(
                    "Indexer could not index n-gram during serialization: {}",
                    ngram
                ),
                io::ErrorKind::Other.into(),
            ));
        }

        if indices.len() > 1 {
            return Err(Error::write_error(
                format!(
                    "Indexer maps n-gram to multiple indices during serialization: {}",
                    ngram
                ),
                io::ErrorKind::Other.into(),
            ));
        }

        let idx = indices[0];

        write
            .write_u32::<LittleEndian>(ngram.len() as u32)
            .map_err(|e| Error::write_error("Cannot write ngram length", e))?;
        write
            .write_all(ngram.as_bytes())
            .map_err(|e| Error::write_error("Cannot write ngram", e))?;
        write
            .write_u64::<LittleEndian>(idx)
            .map_err(|e| Error::write_error("Cannot write ngram idx", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use super::{BucketSubwordVocab, FastTextSubwordVocab, SubwordVocab};
    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::vocab::{ExplicitSubwordVocab, Vocab};
    use crate::compat::fasttext::FastTextIndexer;
    use crate::compat::floret::FloretIndexer;
    use crate::subword::{
        BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer, StrWithCharLen,
    };
    use crate::vocab::tests::test_vocab_chunk_len;
    use crate::vocab::FloretSubwordVocab;

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

    fn test_floret_subword_vocab() -> FloretSubwordVocab {
        let words = vec![];
        let indexer = FloretIndexer::new(65535, 3, 42);
        SubwordVocab::new_with_boundaries(words, 3, 6, indexer, "[", "]")
    }

    fn test_subword_vocab() -> BucketSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let indexer = FinalfusionHashIndexer::new(20);
        SubwordVocab::new(words, 3, 6, indexer)
    }

    fn test_ngram_vocab() -> ExplicitSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];

        let ngrams = vec![
            ("is>".to_owned(), 0),
            ("is".to_owned(), 1),
            ("<t".to_owned(), 2),
        ];

        ExplicitSubwordVocab::new(words, 2, 3, ExplicitIndexer::new_with_indices(ngrams).0)
    }

    #[test]
    fn test_conversion() {
        let words = vec!["groß".to_owned(), "allerdings".to_owned()];
        let indexer = FinalfusionHashIndexer::new(21);
        let bucket_vocab = SubwordVocab::new(words, 3, 6, indexer);
        let (explicit, _) = bucket_vocab.to_explicit().unwrap();
        let dings = StrWithCharLen::new("dings");
        let gro = StrWithCharLen::new("<gro");
        let dings_expl_idx = explicit.indexer().index_ngram(&dings);
        let gro_expl_idx = explicit.indexer().index_ngram(&gro);
        assert_eq!(dings_expl_idx, gro_expl_idx);
        let dings_buck_idx = bucket_vocab.indexer().index_ngram(&dings);
        let gro_buck_idx = bucket_vocab.indexer().index_ngram(&gro);
        assert_eq!(gro_buck_idx, dings_buck_idx);
        assert_eq!(explicit.vocab_len(), explicit.words_len() + 43);
        assert_eq!(explicit.indexer().upper_bound(), 43);
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
    fn fasttext_subword_vocab_correct_chunk_size() {
        test_vocab_chunk_len(test_fasttext_subword_vocab().into());
    }

    #[test]
    fn floret_subword_vocab_write_read_roundtrip() {
        let check_vocab = test_floret_subword_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn floret_subword_vocab_correct_chunk_size() {
        test_vocab_chunk_len(test_floret_subword_vocab().into());
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
        test_vocab_chunk_len(test_subword_vocab().into());
    }

    #[test]
    fn ngram_vocab_write_read_roundtrip() {
        let check_vocab = test_ngram_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = ExplicitSubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn ngram_vocab_correct_chunk_size() {
        test_vocab_chunk_len(test_ngram_vocab().into());
    }

    #[test]
    fn bucket_vocabs_no_indices_are_none() {
        let check_vocab = test_subword_vocab();
        assert!(check_vocab.idx("").is_none());
        let check_vocab = test_ngram_vocab();
        assert!(check_vocab.idx("").is_none());
    }
}
