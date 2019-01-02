use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::iter::Enumerate;
use std::slice;

use failure::{bail, ensure, Error, Fail};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

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

#[derive(Debug, Fail)]
pub enum BuilderError {
    #[fail(
        display = "invalid embedding shape, expected: {}, got: {}",
        expected_len, len
    )]
    InvalidEmbeddingLength { expected_len: usize, len: usize },
    #[fail(display = "word not unique: {}", word)]
    DuplicateWord { word: String },
}

/// Builder for word embedding matrices.
///
/// This builder can be used to construct an embedding matrix. The builder
/// does not assume that the number of embeddings is known ahead of time.
/// The embedding size is determined by the size of the first embedding that
/// is added. All subsequently added embeddings should have the same size.
pub struct Builder {
    words: Vec<String>,
    indices: HashMap<String, usize>,
    embeddings: Vec<Array1<f32>>,
}

impl Builder {
    /// Create a builder.
    pub fn new() -> Self {
        Builder {
            words: Vec::new(),
            indices: HashMap::new(),
            embeddings: Vec::new(),
        }
    }

    /// Construct the final embeddin matrix.
    ///
    /// The `None` is returned when no embedding was added to the builder.
    pub fn build(self) -> Option<Embeddings> {
        let embed_len = self.embeddings.first()?.shape()[0];
        let mut matrix = Array2::zeros((self.embeddings.len(), embed_len));
        for (idx, embed) in self.embeddings.into_iter().enumerate() {
            matrix.index_axis_mut(Axis(0), idx).assign(&embed);
        }

        Some(Embeddings {
            embed_len,
            indices: self.indices,
            words: self.words,
            matrix,
        })
    }

    /// Add a new embedding to the builder.
    ///
    /// An `Err` value is returned when the word has been inserted in the
    /// builder before or when the embedding has a different size than
    /// previously inserted embeddings.
    pub fn push<S, E>(&mut self, word: S, embedding: E) -> Result<(), Error>
    where
        S: Into<String>,
        E: Into<Array1<f32>>,
    {
        let word = word.into();
        let embedding = embedding.into();

        // Check that the embedding has the same length as embeddings that
        // were inserted before.
        if let Some(first) = self.embeddings.first() {
            ensure!(
                embedding.shape() == first.shape(),
                BuilderError::InvalidEmbeddingLength {
                    expected_len: first.shape()[0],
                    len: embedding.shape()[0],
                }
            );
        }

        // Insert the word if it was not inserted before.
        match self.indices.entry(word.to_owned()) {
            Entry::Vacant(vacant) => vacant.insert(self.words.len()),
            Entry::Occupied(_) => bail!(BuilderError::DuplicateWord { word: word }),
        };

        self.words.push(word.clone());
        self.embeddings.push(embedding);

        Ok(())
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

pub fn new_embeddings(
    matrix: Array2<f32>,
    embed_len: usize,
    indices: HashMap<String, usize>,
    words: Vec<String>,
) -> Embeddings {
    Embeddings {
        matrix: matrix,
        embed_len: embed_len,
        indices: indices,
        words: words,
    }
}
