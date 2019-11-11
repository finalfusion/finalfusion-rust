//! Embedding vocabularies

use std::collections::HashMap;
use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array1;

use crate::io::{Error, ErrorKind, Result};

mod subword;
pub use subword::{
    BucketSubwordVocab, ExplicitSubwordVocab, FastTextSubwordVocab, NGramIndices, SubwordIndices,
    SubwordVocab,
};

mod simple;
pub use simple::SimpleVocab;

mod wrappers;
pub use wrappers::VocabWrap;

/// Embedding vocabularies.
#[allow(clippy::len_without_is_empty)]
pub trait Vocab {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<WordIndex>;

    /// Get the number of words in the vocabulary.
    fn words_len(&self) -> usize;

    /// Get the total length of this vocabulary, including possible subword indices.
    fn vocab_len(&self) -> usize;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

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

/// Prune the embedding vocabularies.
pub trait VocabPrune: Vocab {
    /// Prune the vocabulary and get a new one.
    fn prune_vocab(&self, remapped_indices: HashMap<String, usize>) -> VocabWrap;
}

/// Handle the indices changes during pruning.
pub trait VocabPruneIndices: Vocab {
    /// Seperate the indices of the words whose original vectors need to be tossed from the
    /// ones whose vectors need to be kept.
    fn part_indices(&self, n_keep: usize) -> (Vec<usize>, Vec<usize>);

    /// Remap the indices of the words whose original vectors need to be tossed to their
    /// closest remainings' indices.
    fn create_remapped_indices(
        &self,
        most_similar_indices: &Array1<usize>,
    ) -> HashMap<String, usize>;
}

pub(crate) fn create_indices(words: &[String]) -> HashMap<String, usize> {
    let mut indices = HashMap::new();

    for (idx, word) in words.iter().enumerate() {
        indices.insert(word.to_owned(), idx);
    }

    indices
}

pub(crate) fn read_vocab_items<R>(read: &mut R, len: usize) -> Result<Vec<String>>
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

pub(crate) fn write_vocab_items<W>(write: &mut W, items: &[String]) -> Result<()>
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
pub(crate) fn read_chunk_size(read: &mut impl Read) -> u64 {
    // Skip identifier.
    read.read_u32::<LittleEndian>().unwrap();

    // Return chunk length.
    read.read_u64::<LittleEndian>().unwrap()
}
