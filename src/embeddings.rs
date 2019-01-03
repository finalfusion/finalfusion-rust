use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::Enumerate;
use std::slice;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

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
pub struct Embeddings {
    matrix: Array2<f32>,
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl Embeddings {
    pub(crate) fn new(
        matrix: Array2<f32>,
        indices: HashMap<String, usize>,
        words: Vec<String>,
    ) -> Embeddings {
        Embeddings {
            matrix: matrix,
            indices: indices,
            words: words,
        }
    }

    /// Get (a view of) the raw embedding matrix.
    pub fn data(&self) -> ArrayView2<f32> {
        self.matrix.view()
    }

    /// Return the length (in vector components) of the word embeddings.
    pub fn embed_len(&self) -> usize {
        self.matrix.cols()
    }

    /// Get the embedding of a word.
    pub fn embedding(&self, word: &str) -> Option<ArrayView1<f32>> {
        self.indices
            .get(word)
            .map(|idx| self.matrix.index_axis(Axis(0), *idx))
    }

    /// Get the mapping from words to row indices of the embedding matrix.
    pub fn indices(&self) -> &HashMap<String, usize> {
        &self.indices
    }

    /// Get an iterator over pairs of words and the corresponding embeddings.
    pub fn iter(&self) -> Iter {
        Iter {
            embeddings: self,
            inner: self.words.iter().enumerate(),
        }
    }

    /// Normalize the embeddings using their L2 (euclidean) norms.
    ///
    /// **Note:** when you are using the output of e.g. word2vec, you should
    /// normalize the embeddings to get good query results.
    pub fn normalize(&mut self) {
        for mut embedding in self.matrix.outer_iter_mut() {
            let l2norm = embedding.dot(&embedding).sqrt();
            if l2norm != 0f32 {
                embedding /= l2norm;
            }
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

impl<'a> IntoIterator for &'a Embeddings {
    type Item = (&'a str, ArrayView1<'a, f32>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over embeddings.
pub struct Iter<'a> {
    embeddings: &'a Embeddings,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, ArrayView1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(idx, word)| {
            (
                word.as_str(),
                self.embeddings.matrix.index_axis(Axis(0), idx),
            )
        })
    }
}
