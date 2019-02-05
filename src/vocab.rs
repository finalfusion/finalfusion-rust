use std::collections::HashMap;
use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, Error};

use crate::io::{ReadChunk, WriteChunk};

pub trait Vocab {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<usize>;

    /// Get the vocabulary size.
    fn len(&self) -> usize;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

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

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl ReadChunk for SimpleVocab {
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error> {
        ensure!(
            read.read_u32::<LittleEndian>()? == 0,
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

        write.write_u32::<LittleEndian>(0)?;
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.words().len() as u64)?;

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

    use super::SimpleVocab;
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
}
