use std::cmp::Ordering;
use std::io::{Read, Write};
use std::iter::Enumerate;
use std::slice;

use failure::{ensure, Error};
use ndarray::Array1;

use crate::io::{
    ChunkIdentifier, Header, ReadChunk, ReadModelBinary, WriteChunk, WriteModelBinary,
};
use crate::storage::{CowArray, CowArray1, NdArray, Normalize, Storage};
use crate::util::l2_normalize;
use crate::vocab::{SimpleVocab, SubwordVocab, Vocab};

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
pub struct Embeddings<V, S> {
    storage: S,
    vocab: V,
}

impl<V, S> Embeddings<V, S>
where
    S: Storage,
{
    /// Get the embedding storage.
    pub fn data(&self) -> &S {
        &self.storage
    }

    /// Return the length (in vector components) of the word embeddings.
    pub fn embed_len(&self) -> usize {
        self.storage.dims()
    }
}

impl<V, S> Embeddings<V, S>
where
    V: Vocab,
{
    pub fn vocab(&self) -> &V {
        &self.vocab
    }
}

impl<V, S> Embeddings<V, S>
where
    V: Vocab,
    S: Storage,
{
    pub fn new(vocab: V, storage: S) -> Embeddings<V, S> {
        Embeddings { vocab, storage }
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
    pub fn iter(&self) -> Iter<S> {
        Iter {
            storage: &self.storage,
            inner: self.vocab.words().iter().enumerate(),
        }
    }
}

impl<V, S> Embeddings<V, S>
where
    V: Vocab,
    S: Normalize,
{
    pub fn normalize(&mut self) {
        self.storage.normalize();
    }
}

impl<'a, V, S> IntoIterator for &'a Embeddings<V, S>
where
    V: Vocab,
    S: Storage,
{
    type Item = (&'a str, CowArray1<'a, f32>);
    type IntoIter = Iter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

macro_rules! read_impl {
    ($vocab:tt, $storage:tt, $vocab_id:expr, $storage_id:expr) => {
        impl ReadModelBinary for Embeddings<$vocab, $storage> {
            fn read_model_binary(read: &mut impl Read) -> Result<Self, Error> {
                let header = Header::read_chunk(read)?;
                ensure!(
                    header.chunk_identifiers() == [$vocab_id, $storage_id],
                    "Data does not contain correct chunks."
                );
                let vocab = $vocab::read_chunk(read)?;
                let storage = $storage::read_chunk(read)?;
                Ok(Embeddings { vocab, storage })
            }
        }
    };
}

read_impl!(
    SimpleVocab,
    NdArray,
    ChunkIdentifier::SimpleVocab,
    ChunkIdentifier::NdArray
);
read_impl!(
    SubwordVocab,
    NdArray,
    ChunkIdentifier::SubwordVocab,
    ChunkIdentifier::NdArray
);

macro_rules! write_impl {
    ($vocab:tt, $storage:tt, $vocab_id:expr, $storage_id:expr) => {
        impl WriteModelBinary for Embeddings<$vocab, $storage> {
            fn write_model_binary(&self, write: &mut impl Write) -> Result<(), Error> {
                let header = Header::new(vec![$vocab_id, $storage_id]);
                header.write_chunk(write)?;
                self.vocab().write_chunk(write)?;
                self.data().write_chunk(write)
            }
        }
    };
}

write_impl!(
    SimpleVocab,
    NdArray,
    ChunkIdentifier::SimpleVocab,
    ChunkIdentifier::NdArray
);
write_impl!(
    SubwordVocab,
    NdArray,
    ChunkIdentifier::SubwordVocab,
    ChunkIdentifier::NdArray
);

/// Iterator over embeddings.
pub struct Iter<'a, S> {
    storage: &'a S,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a, S> Iterator for Iter<'a, S>
where
    S: Storage,
{
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
    use crate::io::{ReadModelBinary, WriteModelBinary};
    use crate::storage::NdArray;
    use crate::vocab::SimpleVocab;
    use crate::word2vec::ReadWord2Vec;

    fn test_embeddings() -> Embeddings<SimpleVocab, NdArray> {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary(&mut reader).unwrap()
    }

    #[test]
    fn write_read_simple_roundtrip() {
        let check_embeds = test_embeddings();
        let mut serialized = Vec::new();
        check_embeds.write_model_binary(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let embeds = Embeddings::read_model_binary(&mut cursor).unwrap();
        assert_eq!(embeds, check_embeds);
    }
}
