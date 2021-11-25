//! Embedding vocabularies

use std::collections::HashMap;
use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Error, Result};

mod subword;
pub use subword::{
    BucketSubwordVocab, ExplicitSubwordVocab, FastTextSubwordVocab, FloretSubwordVocab,
    NGramIndices, SubwordIndices, SubwordVocab,
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

pub(crate) fn create_indices(words: &[String]) -> HashMap<String, usize> {
    let mut indices = HashMap::new();

    for (idx, word) in words.iter().enumerate() {
        indices.insert(word.to_owned(), idx);
    }

    indices
}

pub(crate) fn read_string<R>(read: &mut R) -> Result<String>
where
    R: Read,
{
    let string_len =
        read.read_u32::<LittleEndian>()
            .map_err(|e| Error::io_error("Cannot read string length", e))? as usize;
    let mut bytes = vec![0; string_len];
    read.read_exact(&mut bytes)
        .map_err(|e| Error::io_error("Cannot read item", e))?;
    String::from_utf8(bytes)
        .map_err(|e| Error::Format(format!("Item contains invalid UTF-8: {}", e)))
        .map_err(Error::from)
}

pub(crate) fn read_vocab_items<R>(read: &mut R, len: usize) -> Result<Vec<String>>
where
    R: Read,
{
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        let item = read_string(read)?;
        items.push(item);
    }
    Ok(items)
}

pub(crate) fn write_string<W>(write: &mut W, s: &str) -> Result<()>
where
    W: Write,
{
    write
        .write_u32::<LittleEndian>(s.len() as u32)
        .map_err(|e| Error::io_error("Cannot write string length", e))?;
    write
        .write_all(s.as_bytes())
        .map_err(|e| Error::io_error("Cannot write string", e))
}

pub(crate) fn write_vocab_items<W>(write: &mut W, items: &[String]) -> Result<()>
where
    W: Write + Seek,
{
    for word in items {
        write_string(write, word)?;
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
