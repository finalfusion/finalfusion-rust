use std::collections::HashMap;
use std::convert::TryFrom;
use std::io;
use std::io::{Read, Seek, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use smallvec::SmallVec;

use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::vocab::{create_indices, read_vocab_items, write_vocab_items, Vocab, WordIndex};
use crate::compat::fasttext::FastTextIndexer;
use crate::compat::floret::FloretIndexer;
use crate::error::{Error, Result};
use crate::subword::{
    BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer, IndicesScope,
    SubwordIndices as StrSubwordIndices,
};

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
    indices_scope: IndicesScope,
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
    pub fn new(
        words: impl Into<Vec<String>>,
        min_n: u32,
        max_n: u32,
        indexer: I,
        indices_scope: IndicesScope,
    ) -> Self {
        Self::new_with_boundaries(
            words,
            min_n,
            max_n,
            indexer,
            indices_scope,
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
        indices_scope: IndicesScope,
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
            indices_scope,
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
        self.subword_indices(word, self.indices_scope)
            .map(WordIndex::Subword)
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
            .subword_indices_with_ngrams(
                self.min_n as usize,
                self.max_n as usize,
                &self.indexer,
                self.indices_scope,
            )
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
    /// according to the subword vocabulary. `scope` specifies
    /// whether the word itself should also be included in indexing.
    fn subword_indices(&self, word: &str, scope: IndicesScope) -> Option<Vec<usize>>;
}

impl<I> SubwordIndices for SubwordVocab<I>
where
    I: Indexer,
{
    fn subword_indices(&self, word: &str, indices_scope: IndicesScope) -> Option<Vec<usize>> {
        let word = self.bracket(word);
        let indices = word
            .as_str()
            .subword_indices(
                self.min_n as usize,
                self.max_n as usize,
                &self.indexer,
                indices_scope,
            )
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

impl WriteChunk for BucketSubwordVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::BucketSubwordVocab
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

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        self.write_ngram_chunk(write, self.chunk_identifier())
    }
}

impl<I> SubwordVocab<I>
where
    I: BucketIndexer,
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
            .map_err(|e| Error::io_error("Cannot read vocabulary chunk length", e))?;

        let vocab_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read vocabulary length", e))?
            as usize;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read maximum n-gram length", e))?;
        let buckets = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read number of buckets", e))?;

        let words = read_vocab_items(read, vocab_len as usize)?;

        Ok(SubwordVocab::new(
            words,
            min_n,
            max_n,
            I::new(buckets as usize),
            IndicesScope::Substrings,
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
            .map_err(|e| Error::io_error("Cannot write subword vocabulary chunk identifier", e))?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| Error::io_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| Error::io_error("Cannot write vocabulary length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::io_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::io_error("Cannot write maximum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.indexer.buckets() as u32)
            .map_err(|e| Error::io_error("Cannot write number of buckets", e))?;

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
            indices_scope: _,
            indices: _,
            indexer,
            min_n,
            max_n,
        } = &self;

        for word in words.iter().map(|word| self.bracket(word)) {
            for (ngram, indices) in word.subword_indices_with_ngrams(
                *min_n as usize,
                *max_n as usize,
                indexer,
                self.indices_scope,
            ) {
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
            ExplicitSubwordVocab::new(
                words.to_owned(),
                *min_n,
                *max_n,
                indexer,
                IndicesScope::Substrings,
            ),
            mapping,
        ))
    }
}

impl SubwordVocab<ExplicitIndexer> {
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
            .map_err(|e| Error::io_error("Cannot read vocabulary chunk length", e))?;
        let words_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read number of words", e))?;
        let ngrams_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read number of ngrams", e))?;
        let min_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read minimum n-gram length", e))?;
        let max_n = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read maximum n-gram length", e))?;

        let words = read_vocab_items(read, words_len as usize)?;
        let ngrams = read_ngrams_with_indices(read, ngrams_len as usize)?;
        let (indexer, _) = ExplicitIndexer::new_with_indices(ngrams);
        Ok(SubwordVocab::new(
            words,
            min_n,
            max_n,
            indexer,
            IndicesScope::Substrings,
        ))
    }

    fn write_ngram_chunk<W>(&self, write: &mut W, chunk_identifier: ChunkIdentifier) -> Result<()>
    where
        W: Write + Seek,
    {
        // Chunk size: word vocab size (u64), ngram vocab size (u64)
        // minimum n-gram length (u32), maximum n-gram length (u32),
        // for each word and ngram:
        // length in bytes (u32), number of bytes (variable-length).
        // each ngram is followed by its index (u64)
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
                .map(|ngram| ngram.len() + size_of::<u32>() + size_of::<u64>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(chunk_identifier as u32)
            .map_err(|e| Error::io_error("Cannot write subword vocabulary chunk identifier", e))?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| Error::io_error("Cannot write subword vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| Error::io_error("Cannot write vocabulary length", e))?;
        write
            .write_u64::<LittleEndian>(self.indexer.ngrams().len() as u64)
            .map_err(|e| Error::io_error("Cannot write ngram length", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::io_error("Cannot write minimum n-gram length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::io_error("Cannot write maximum n-gram length", e))?;

        write_vocab_items(write, self.words())?;
        write_ngrams_with_indices(write, self.indexer())?;

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
                .map_err(|e| Error::io_error("Cannot read item length", e))? as usize;
        let mut bytes = vec![0; ngram_len];
        read.read_exact(&mut bytes)
            .map_err(|e| Error::io_error("Cannot read item", e))?;
        let item = String::from_utf8(bytes)
            .map_err(|e| Error::Format(format!("Item contains invalid UTF-8: {}", e)))
            .map_err(Error::from)?;
        let idx = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read ngram index.", e))?;
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
            return Err(Error::io_error(
                format!(
                    "Indexer could not index n-gram during serialization: {}",
                    ngram
                ),
                io::ErrorKind::Other.into(),
            ));
        }

        if indices.len() > 1 {
            return Err(Error::io_error(
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
            .map_err(|e| Error::io_error("Cannot write ngram length", e))?;
        write
            .write_all(ngram.as_bytes())
            .map_err(|e| Error::io_error("Cannot write ngram", e))?;
        write
            .write_u64::<LittleEndian>(idx)
            .map_err(|e| Error::io_error("Cannot write ngram idx", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use super::{BucketSubwordVocab, FastTextSubwordVocab, SubwordVocab};
    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::vocab::{read_chunk_size, ExplicitSubwordVocab, Vocab};
    use crate::compat::fasttext::FastTextIndexer;
    use crate::subword::{
        BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer, IndicesScope,
        StrWithCharLen,
    };

    fn test_fasttext_subword_vocab() -> FastTextSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let indexer = FastTextIndexer::new(20);
        SubwordVocab::new(words, 3, 6, indexer, IndicesScope::Substrings)
    }

    fn test_subword_vocab() -> BucketSubwordVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let indexer = FinalfusionHashIndexer::new(20);
        SubwordVocab::new(words, 3, 6, indexer, IndicesScope::Substrings)
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

        ExplicitSubwordVocab::new(
            words,
            2,
            3,
            ExplicitIndexer::new_with_indices(ngrams).0,
            IndicesScope::Substrings,
        )
    }

    #[test]
    fn test_conversion() {
        let words = vec!["groß".to_owned(), "allerdings".to_owned()];
        let indexer = FinalfusionHashIndexer::new(21);
        let bucket_vocab = SubwordVocab::new(words, 3, 6, indexer, IndicesScope::Substrings);
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
        let vocab = ExplicitSubwordVocab::read_chunk(&mut cursor).unwrap();
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

    #[test]
    fn bucket_vocabs_no_indices_are_none() {
        let check_vocab = test_subword_vocab();
        assert!(check_vocab.idx("").is_none());
        let check_vocab = test_ngram_vocab();
        assert!(check_vocab.idx("").is_none());
    }
}
