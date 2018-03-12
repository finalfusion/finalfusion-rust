use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::iter::Enumerate;
use std::slice;

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1};

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
    embed_len: usize,
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl Embeddings {
    /// Perform an analogy query.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    pub fn analogy(&self,
                   word1: &str,
                   word2: &str,
                   word3: &str,
                   limit: usize)
                   -> Option<Vec<WordSimilarity>> {
        self.analogy_by(word1,
                        word2,
                        word3,
                        limit,
                        |embeds, embed| embeds.dot(&embed))
    }

    /// Perform an analogy query using the given similarity function.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    pub fn analogy_by<F>(&self,
                         word1: &str,
                         word2: &str,
                         word3: &str,
                         limit: usize,
                         similarity: F)
                         -> Option<Vec<WordSimilarity>>
        where F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>
    {
        let embedding1 = try_opt!(self.indices
            .get(word1)
            .map(|idx| self.matrix.subview(Axis(0), *idx).to_owned()));
        let embedding2 = try_opt!(self.indices
            .get(word2)
            .map(|idx| self.matrix.subview(Axis(0), *idx).to_owned()));
        let embedding3 = try_opt!(self.indices
            .get(word3)
            .map(|idx| self.matrix.subview(Axis(0), *idx).to_owned()));

        let embedding = (embedding2 - embedding1) + embedding3;

        let skip = [word1, word2, word3].iter().cloned().collect();

        Some(self.similarity_(embedding.view(), &skip, limit, similarity))
    }

    /// Get (a view of) the raw embedding matrix.
    pub fn data(&self) -> ArrayView2<f32> {
        self.matrix.view()
    }

    /// Return the length (in vector components) of the word embeddings.
    pub fn embed_len(&self) -> usize {
        self.embed_len
    }

    /// Get the embedding of a word.
    pub fn embedding(&self, word: &str) -> Option<ArrayView1<f32>> {
        self.indices.get(word).map(|idx| self.matrix.subview(Axis(0), *idx))
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

    /// Find words that are similar to the query word.
    ///
    /// The similarity between two words is defined by the dot product of
    /// the embeddings. If the vectors are unit vectors (e.g. by virtue of
    /// calling `normalize`), this is the cosine similarity. At most, `limit`
    /// results are returned.
    pub fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>> {
        self.similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }

    /// Find words that are similar to the query word using the given similarity
    /// function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the word vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    pub fn similarity_by<F>(&self,
                            word: &str,
                            limit: usize,
                            similarity: F)
                            -> Option<Vec<WordSimilarity>>
        where F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>
    {
        self.indices.get(word).map(|idx| {
            let embedding = self.matrix.subview(Axis(0), *idx);
            let mut skip = HashSet::new();
            skip.insert(word);
            self.similarity_(embedding, &skip, limit, similarity)
        })
    }

    fn similarity_<F, S>(&self,
                         embed: ArrayBase<S, Ix1>,
                         skip: &HashSet<&str>,
                         limit: usize,
                         mut similarity: F)
                         -> Vec<WordSimilarity>
        where F: FnMut(ArrayView2<f32>, ArrayBase<S, Ix1>) -> Array1<f32>,
              S: Data<Elem = f32>
    {
        let sims = similarity(self.matrix.view(), embed);

        let mut results: BinaryHeap<WordSimilarity> = BinaryHeap::new();
        for (idx, sim) in sims.iter().enumerate() {
            let word = self.words[idx].as_ref();

            // Don't add words that we are explicitly asked to skip.
            if skip.contains(word) {
                continue;
            }

            let word_distance = WordSimilarity {
                word: word,
                similarity: *sim,
            };

            if results.len() == limit {
                if let Some(mut min_distance) = results.peek_mut() {
                    if word_distance.similarity > min_distance.similarity {
                        *min_distance = word_distance
                    }
                }
            } else {
                results.push(word_distance);
            }
        }

        results.into_sorted_vec()
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

/// Iterator over embeddings.
pub struct Iter<'a> {
    embeddings: &'a Embeddings,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, ArrayView1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(idx, word)| (word.as_str(), self.embeddings.matrix.subview(Axis(0), idx)))
    }
}

pub fn new_embeddings(matrix: Array2<f32>,
                      embed_len: usize,
                      indices: HashMap<String, usize>,
                      words: Vec<String>)
                      -> Embeddings {
    Embeddings {
        matrix: matrix,
        embed_len: embed_len,
        indices: indices,
        words: words,
    }
}
