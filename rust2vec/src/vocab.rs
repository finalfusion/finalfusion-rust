use std::collections::HashMap;
use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, Error};

use crate::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::subword::SubwordIndices;

pub trait Vocab {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<usize>;

    /// Get the vocabulary size.
    fn len(&self) -> usize;

    /// Get the subword indices of a token.
    ///
    /// Returns `None` when the model does not support subwords or
    /// when no subwords could be extracted.
    fn subword_indices(&self, word: &str) -> Option<Vec<usize>>;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

/// Vocabulary that only consists of words.
#[derive(Debug, Eq, PartialEq)]
pub struct SimpleVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl SimpleVocab {
    pub fn new(words: impl Into<Vec<String>>) -> Self {
        let words = words.into();

        let mut indices = HashMap::new();
        for (idx, word) in words.iter().enumerate() {
            indices.insert(word.to_owned(), idx);
        }

        SimpleVocab { words, indices }
    }
}

impl Vocab for SimpleVocab {
    fn idx(&self, word: &str) -> Option<usize> {
        self.indices.get(word).cloned()
    }

    fn len(&self) -> usize {
        self.words.len()
    }

    fn subword_indices(&self, _word: &str) -> Option<Vec<usize>> {
        None
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl ReadChunk for SimpleVocab {
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error> {
        ensure!(
            read.read_u32::<LittleEndian>()? == ChunkIdentifier::SimpleVocab as u32,
            "invalid chunk identifier for NdArray"
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let vocab_len = read.read_u64::<LittleEndian>()? as usize;
        let mut words = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let word_len = read.read_u32::<LittleEndian>()? as usize;
            let mut bytes = vec![0; word_len];
            read.read_exact(&mut bytes)?;
            let word = String::from_utf8(bytes)?;
            words.push(word);
        }

        Ok(SimpleVocab::new(words))
    }
}

impl WriteChunk for SimpleVocab {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error> {
        let chunk_len = self.words().iter().map(|w| w.len() as u64 + 4).sum::<u64>() + 8;

        write.write_u32::<LittleEndian>(ChunkIdentifier::SimpleVocab as u32)?;
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.words().len() as u64)?;

        for word in self.words() {
            write.write_u32::<LittleEndian>(word.len() as u32)?;
            write.write_all(word.as_bytes())?;
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct SubwordVocab {
    inner: SimpleVocab,
    min_n: u32,
    max_n: u32,
    buckets_exp: u32,
}

impl SubwordVocab {
    const BOW: char = '<';
    const EOW: char = '>';

    pub fn new(words: impl Into<Vec<String>>, min_n: u32, max_n: u32, buckets_exp: u32) -> Self {
        let words = words.into();

        let mut indices = HashMap::new();
        for (idx, word) in words.iter().enumerate() {
            indices.insert(word.to_owned(), idx);
        }

        SubwordVocab {
            inner: SimpleVocab { words, indices },
            min_n,
            max_n,
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
}

impl Vocab for SubwordVocab {
    fn idx(&self, word: &str) -> Option<usize> {
        self.inner.idx(word)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn subword_indices(&self, word: &str) -> Option<Vec<usize>> {
        let indices = Self::bracket(word)
            .as_str()
            .subword_indices(
                self.min_n as usize,
                self.max_n as usize,
                self.buckets_exp as usize,
            )
            .into_iter()
            .map(|idx| idx as usize + self.len())
            .collect::<Vec<_>>();
        if indices.len() == 0 {
            None
        } else {
            Some(indices)
        }
    }

    fn words(&self) -> &[String] {
        &self.inner.words()
    }
}

impl ReadChunk for SubwordVocab {
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error> {
        ensure!(
            read.read_u32::<LittleEndian>()? == ChunkIdentifier::SubwordVocab as u32,
            "invalid chunk identifier for NdArray"
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let vocab_len = read.read_u64::<LittleEndian>()? as usize;
        let min_n = read.read_u32::<LittleEndian>()?;
        let max_n = read.read_u32::<LittleEndian>()?;
        let buckets_exp = read.read_u32::<LittleEndian>()?;

        let mut words = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let word_len = read.read_u32::<LittleEndian>()? as usize;
            let mut bytes = vec![0; word_len];
            read.read_exact(&mut bytes)?;
            let word = String::from_utf8(bytes)?;
            words.push(word);
        }

        Ok(SubwordVocab::new(words, min_n, max_n, buckets_exp))
    }
}

impl WriteChunk for SubwordVocab {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error> {
        let chunk_len = self.words().iter().map(|w| w.len() as u64 + 4).sum::<u64>() + 20;

        write.write_u32::<LittleEndian>(ChunkIdentifier::SubwordVocab as u32)?;
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.words().len() as u64)?;
        write.write_u32::<LittleEndian>(self.min_n)?;
        write.write_u32::<LittleEndian>(self.max_n)?;
        write.write_u32::<LittleEndian>(self.buckets_exp)?;

        for word in self.words() {
            write.write_u32::<LittleEndian>(word.len() as u32)?;
            write.write_all(word.as_bytes())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{SimpleVocab, SubwordVocab};
    use crate::io::{ReadChunk, WriteChunk};

    #[test]
    fn simple_vocab_write_read_roundtrip() {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let check_vocab = SimpleVocab::new(words);
        let mut serialized = Vec::new();
        check_vocab.write_chunk(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let vocab = SimpleVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn subword_vocab_write_read_roundtrip() {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let check_vocab = SubwordVocab::new(words, 3, 6, 20);
        let mut serialized = Vec::new();
        check_vocab.write_chunk(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let vocab = SubwordVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }
}
