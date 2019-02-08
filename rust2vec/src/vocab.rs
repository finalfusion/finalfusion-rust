use std::collections::HashMap;
use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{err_msg, format_err, Error};

use crate::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::subword::SubwordIndices;

#[derive(Debug, Eq, PartialEq)]
pub enum Vocab {
    /// Vocabulary that only consists of words.
    SimpleVocab {
        indices: HashMap<String, usize>,
        words: Vec<String>,
    },
    SubwordVocab {
        indices: HashMap<String, usize>,
        words: Vec<String>,
        min_n: u32,
        max_n: u32,
        buckets_exp: u32,
    },
}

impl Vocab {
    const BOW: char = '<';
    const EOW: char = '>';

    pub fn new_simple_vocab(words: impl Into<Vec<String>>) -> Self {
        let words = words.into();
        let indices = Self::create_indices(&words);
        Vocab::SimpleVocab { words, indices }
    }

    pub fn new_subword_vocab(
        words: impl Into<Vec<String>>,
        min_n: u32,
        max_n: u32,
        buckets_exp: u32,
    ) -> Self {
        let words = words.into();
        let indices = Self::create_indices(&words);

        Vocab::SubwordVocab {
            indices,
            words,
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

    fn create_indices(words: &[String]) -> HashMap<String, usize> {
        let mut indices = HashMap::new();

        for (idx, word) in words.iter().enumerate() {
            indices.insert(word.to_owned(), idx);
        }

        indices
    }

    pub(crate) fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            Vocab::SimpleVocab { .. } => ChunkIdentifier::SimpleVocab,
            Vocab::SubwordVocab { .. } => ChunkIdentifier::SubwordVocab,
        }
    }

    /// Get the index of a token.
    pub fn idx(&self, word: &str) -> Option<usize> {
        self.indices().get(word).cloned()
    }

    fn indices(&self) -> &HashMap<String, usize> {
        match self {
            Vocab::SimpleVocab { ref indices, .. } => indices,
            Vocab::SubwordVocab { ref indices, .. } => indices,
        }
    }

    /// Get the vocabulary size.
    pub fn len(&self) -> usize {
        self.indices().len()
    }

    /// Get the subword indices of a token.
    ///
    /// Returns `None` when the model does not support subwords or
    /// when no subwords could be extracted.
    pub fn subword_indices(&self, word: &str) -> Option<Vec<usize>> {
        match self {
            Vocab::SimpleVocab { .. } => None,
            Vocab::SubwordVocab {
                min_n,
                max_n,
                buckets_exp,
                ..
            } => {
                let indices = Self::bracket(word)
                    .as_str()
                    .subword_indices(*min_n as usize, *max_n as usize, *buckets_exp as usize)
                    .into_iter()
                    .map(|idx| idx as usize + self.len())
                    .collect::<Vec<_>>();
                if indices.len() == 0 {
                    None
                } else {
                    Some(indices)
                }
            }
        }
    }

    /// Get the words in the vocabulary.
    pub fn words(&self) -> &[String] {
        match self {
            Vocab::SimpleVocab { words, .. } => words,
            Vocab::SubwordVocab { words, .. } => words,
        }
    }
}

impl ReadChunk for Vocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = ChunkIdentifier::try_from(read.read_u32::<LittleEndian>()?)
            .ok_or(err_msg("Unknown chunk identifier"))?;
        match chunk_id {
            ChunkIdentifier::SimpleVocab => {
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

                Ok(Vocab::new_simple_vocab(words))
            }
            ChunkIdentifier::SubwordVocab => {
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

                Ok(Vocab::new_subword_vocab(words, min_n, max_n, buckets_exp))
            }
            unknown => Err(format_err!("Not a vocabulary chunk: {:?}", unknown)),
        }
    }
}

impl WriteChunk for Vocab {
    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        match self {
            Vocab::SimpleVocab { words, .. } => {
                let chunk_len = words.iter().map(|w| w.len() as u64 + 4).sum::<u64>() + 8;

                write.write_u32::<LittleEndian>(ChunkIdentifier::SimpleVocab as u32)?;
                write.write_u64::<LittleEndian>(chunk_len as u64)?;
                write.write_u64::<LittleEndian>(words.len() as u64)?;

                for word in self.words() {
                    write.write_u32::<LittleEndian>(word.len() as u32)?;
                    write.write_all(word.as_bytes())?;
                }

                Ok(())
            }
            Vocab::SubwordVocab {
                min_n,
                max_n,
                buckets_exp,
                words,
                ..
            } => {
                let chunk_len = self.words().iter().map(|w| w.len() as u64 + 4).sum::<u64>() + 20;

                write.write_u32::<LittleEndian>(ChunkIdentifier::SubwordVocab as u32)?;
                write.write_u64::<LittleEndian>(chunk_len as u64)?;
                write.write_u64::<LittleEndian>(words.len() as u64)?;
                write.write_u32::<LittleEndian>(*min_n)?;
                write.write_u32::<LittleEndian>(*max_n)?;
                write.write_u32::<LittleEndian>(*buckets_exp)?;

                for word in self.words() {
                    write.write_u32::<LittleEndian>(word.len() as u32)?;
                    write.write_all(word.as_bytes())?;
                }

                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use super::Vocab;
    use crate::io::{ReadChunk, WriteChunk};

    #[test]
    fn simple_vocab_write_read_roundtrip() {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];
        let check_vocab = Vocab::new_simple_vocab(words);
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = Vocab::read_chunk(&mut cursor).unwrap();
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
        let check_vocab = Vocab::new_subword_vocab(words, 3, 6, 20);
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = Vocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }
}
