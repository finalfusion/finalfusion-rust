use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::Enumerate;
use std::slice;

use crate::storage::{CowArray1, Normalize, Storage};

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
pub struct Embeddings<S> {
    storage: S,
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl<S> Embeddings<S>
where
    S: Storage,
{
    pub(crate) fn new(
        storage: S,
        indices: HashMap<String, usize>,
        words: Vec<String>,
    ) -> Embeddings<S> {
        Embeddings {
            storage: storage,
            indices: indices,
            words: words,
        }
    }

    /// Get the embedding storage.
    pub fn data(&self) -> &S {
        &self.storage
    }

    /// Return the length (in vector components) of the word embeddings.
    pub fn embed_len(&self) -> usize {
        self.storage.dims()
    }

    /// Get the embedding of a word.
    pub fn embedding(&self, word: &str) -> Option<CowArray1<f32>> {
        self.indices
            .get(word)
            .map(|idx| self.storage.embedding(*idx))
    }

    /// Get the mapping from words to row indices of the embedding matrix.
    pub fn indices(&self) -> &HashMap<String, usize> {
        &self.indices
    }

    /// Get an iterator over pairs of words and the corresponding embeddings.
    pub fn iter(&self) -> Iter<S> {
        Iter {
            storage: &self.storage,
            inner: self.words.iter().enumerate(),
        }
    }

    /// Get the number of words for which embeddings are stored.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Get the words for which embeddings are stored. The words line up with
    /// the rows in the matrix returned by `data`.
    pub fn words(&self) -> &[String] {
        &self.words
    }
}

impl<S> Embeddings<S>
where
    S: Normalize,
{
    pub fn normalize(&mut self) {
        self.storage.normalize();
    }
}

impl<'a, S> IntoIterator for &'a Embeddings<S>
where
    S: Storage,
{
    type Item = (&'a str, CowArray1<'a, f32>);
    type IntoIter = Iter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

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
