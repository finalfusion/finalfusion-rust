use std::cmp::Ordering;
use std::io::{Read, Write};
use std::iter::Enumerate;
use std::slice;

use failure::Error;
use ndarray::Array1;

use crate::io::{Header, ReadChunk, ReadEmbeddings, WriteChunk, WriteEmbeddings};
use crate::storage::{CowArray, CowArray1, Storage};
use crate::util::l2_normalize;
use crate::vocab::Vocab;

/// A word similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug)]
pub struct WordSimilarity<'a> {
    pub word: &'a str,
    pub similarity: f32,
}

impl<'a> Ord for WordSimilarity<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.similarity > other.similarity {
            Ordering::Less
        } else if self.similarity < other.similarity {
            Ordering::Greater
        } else {
            self.word.cmp(other.word)
        }
    }
}

impl<'a> PartialOrd for WordSimilarity<'a> {
    fn partial_cmp(&self, other: &WordSimilarity) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Eq for WordSimilarity<'a> {}

impl<'a> PartialEq for WordSimilarity<'a> {
    fn eq(&self, other: &WordSimilarity) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

/// Word embeddings.
///
/// This data structure stores word embeddings (also known as *word vectors*)
/// and provides some useful methods on the embeddings, such as similarity
/// and analogy queries.
#[derive(Debug, PartialEq)]
pub struct Embeddings {
    storage: Storage,
    vocab: Vocab,
}

impl Embeddings {
    pub fn new(vocab: Vocab, storage: Storage) -> Embeddings {
        Embeddings { vocab, storage }
    }

    /// Get the embedding storage.
    pub fn data(&self) -> &Storage {
        &self.storage
    }

    /// Return the length (in vector components) of the word embeddings.
    pub fn embed_len(&self) -> usize {
        self.storage.dims()
    }

    /// Get the embedding of a word.
    pub fn embedding(&self, word: &str) -> Option<CowArray1<f32>> {
        // For known words, we can just return the embedding.
        if let Some(idx) = self.vocab.idx(word) {
            return Some(self.storage.embedding(idx));
        }

        // For unknown words, return the l2-normalized sum of subword
        // embeddings (when available).
        self.vocab.subword_indices(word).map(|indices| {
            let mut embed = Array1::zeros((self.storage.dims(),));
            for idx in indices {
                embed += &self.storage.embedding(idx).as_view();
            }

            l2_normalize(embed.view_mut());

            CowArray::Owned(embed)
        })
    }

    /// Get an iterator over pairs of words and the corresponding embeddings.
    pub fn iter(&self) -> Iter {
        Iter {
            storage: &self.storage,
            inner: self.vocab.words().iter().enumerate(),
        }
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

impl<'a> IntoIterator for &'a Embeddings {
    type Item = (&'a str, CowArray1<'a, f32>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl ReadEmbeddings for Embeddings {
    fn read_embeddings(read: &mut impl Read) -> Result<Self, Error> {
        Header::read_chunk(read)?;
        let vocab = Vocab::read_chunk(read)?;
        let storage = Storage::read_chunk(read)?;

        Ok(Embeddings { vocab, storage })
    }
}

impl WriteEmbeddings for Embeddings {
    fn write_embeddings(&self, write: &mut impl Write) -> Result<(), Error> {
        let header = Header::new(vec![
            self.vocab.chunk_identifier(),
            self.storage.chunk_identifier(),
        ]);
        header.write_chunk(write)?;
        self.vocab.write_chunk(write)?;
        self.storage.write_chunk(write)?;
        Ok(())
    }
}

/// Iterator over embeddings.
pub struct Iter<'a> {
    storage: &'a Storage,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, CowArray1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(idx, word)| (word.as_str(), self.storage.embedding(idx)))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor};

    use super::Embeddings;
    use crate::io::{ReadEmbeddings, WriteEmbeddings};
    use crate::word2vec::ReadWord2Vec;

    fn test_embeddings() -> Embeddings {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary(&mut reader, false).unwrap()
    }

    #[test]
    fn write_read_simple_roundtrip() {
        let check_embeds = test_embeddings();
        let mut serialized = Vec::new();
        check_embeds.write_embeddings(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let embeds = Embeddings::read_embeddings(&mut cursor).unwrap();
        assert_eq!(embeds, check_embeds);
    }
}
