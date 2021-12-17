//! Traits and trait implementations for similarity queries.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::f32;

use ndarray::{s, ArrayView1, Axis, CowArray, Ix1};
use ordered_float::NotNan;

use crate::chunks::storage::{Storage, StorageView};
use crate::chunks::vocab::Vocab;
use crate::embeddings::Embeddings;
use crate::util::l2_normalize;

/// A word with its similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug, Eq, PartialEq)]
pub struct WordSimilarityResult<'a> {
    similarity: NotNan<f32>,
    word: &'a str,
}

impl<'a> WordSimilarityResult<'a> {
    /// Get the word's similarity in angular similarity.
    pub fn angular_similarity(&self) -> f32 {
        1f32 - (self.similarity.acos() / f32::consts::PI)
    }

    /// Get the word's similarity in cosine similarity.
    pub fn cosine_similarity(&self) -> f32 {
        *self.similarity
    }

    /// Get the euclidean distance between the vectors.
    pub fn euclidean_distance(&self) -> f32 {
        // Trivially derived from the law of cosines
        (2f32 - 2f32 * self.cosine_similarity()).sqrt()
    }

    /// Returns the euclidean similarity.
    ///
    /// This method returns a similarity in *[0,1]*, where *0*
    /// corresponds to a euclidean distance of *2* (the maximum
    /// distance between two unit vectors) and *1* to a euclidean
    /// distance of *0*.
    pub fn euclidean_similarity(&self) -> f32 {
        1f32 - (self.euclidean_distance() / 2f32)
    }

    pub fn word(&self) -> &str {
        self.word
    }
}

impl<'a> Ord for WordSimilarityResult<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.similarity.cmp(&self.similarity) {
            Ordering::Equal => self.word.cmp(other.word),
            ordering => ordering,
        }
    }
}

impl<'a> PartialOrd for WordSimilarityResult<'a> {
    fn partial_cmp(&self, other: &WordSimilarityResult) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Trait for analogy queries.
pub trait Analogy {
    /// Perform an analogy query.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned. `Result::Err` is returned
    /// when no embedding could be computed for one or more of the tokens,
    /// indicating which of the tokens were present.
    ///
    /// If `batch_size` is `None`, the query will be performed on all
    /// word embeddings at once. This is typically the most efficient, but
    /// can require a large amount of memory. The query is performed on batches
    /// of size `n` when `batch_size` is `Some(n)`. Setting this to a smaller
    /// value than the number of word embeddings reduces memory use at the
    /// cost of computational efficiency.
    fn analogy(
        &self,
        query: [&str; 3],
        limit: usize,
        batch_size: Option<usize>,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]> {
        self.analogy_masked(query, [true, true, true], limit, batch_size)
    }

    /// Perform an analogy query.
    ///
    /// This method returns words that are close in vector space for the
    /// analogy query `word1` is to `word2` as `word3` is to `?`. More
    /// concretely, it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    ///
    /// `remove` specifies which parts of the queries are excluded from the
    /// output candidates. If `remove[0]` is `true`, `word1` cannot be
    /// returned as an answer to the query.
    ///
    /// If `batch_size` is `None`, the query will be performed on all
    /// word embeddings at once. This is typically the most efficient, but
    /// can require a large amount of memory. The query is performed on batches
    /// of size `n` when `batch_size` is `Some(n)`. Setting this to a smaller
    /// value than the number of word embeddings reduces memory use at the
    /// cost of computational efficiency.
    ///
    ///`Result::Err` is returned when no embedding could be computed
    /// for one or more of the tokens, indicating which of the tokens
    /// were present.
    fn analogy_masked(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
        batch_size: Option<usize>,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]>;
}

impl<V, S> Analogy for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn analogy_masked(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
        batch_size: Option<usize>,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]> {
        {
            let [embedding1, embedding2, embedding3] = lookup_words3(self, query)?;

            let mut embedding = (&embedding2.view() - &embedding1.view()) + embedding3.view();
            l2_normalize(embedding.view_mut());

            let skip = query
                .iter()
                .zip(remove.iter())
                .filter(|(_, &exclude)| exclude)
                .map(|(word, _)| word.to_owned())
                .collect();

            Ok(self.similarity_(embedding.view(), &skip, limit, batch_size))
        }
    }
}

/// Trait for word similarity queries.
pub trait WordSimilarity {
    /// Find words that are similar to the query word.
    ///
    /// The similarity between two words is defined by the dot product of
    /// the embeddings. If the vectors are unit vectors (e.g. by virtue of
    /// calling `normalize`), this is the cosine similarity. At most, `limit`
    /// results are returned.
    ///
    /// If `batch_size` is `None`, the query will be performed on all
    /// word embeddings at once. This is typically the most efficient, but
    /// can require a large amount of memory. The query is performed on batches
    /// of size `n` when `batch_size` is `Some(n)`. Setting this to a smaller
    /// value than the number of word embeddings reduces memory use at the
    /// cost of computational efficiency.
    fn word_similarity(
        &self,
        word: &str,
        limit: usize,
        batch_size: Option<usize>,
    ) -> Option<Vec<WordSimilarityResult>>;
}

impl<V, S> WordSimilarity for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn word_similarity(
        &self,
        word: &str,
        limit: usize,
        batch_size: Option<usize>,
    ) -> Option<Vec<WordSimilarityResult>> {
        let embed = self.embedding(word)?;
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed.view(), &skip, limit, batch_size))
    }
}

/// Trait for embedding similarity queries.
pub trait EmbeddingSimilarity {
    /// Find words that are similar to the query embedding.
    ///
    /// The similarity between the query embedding and other embeddings is
    /// defined by the dot product of the embeddings. The embeddings in the
    /// storage are l2-normalized, this method l2-normalizes the input query,
    /// therefore the dot product is equivalent to the cosine similarity.
    ///
    /// If `batch_size` is `None`, the query will be performed on all
    /// word embeddings at once. This is typically the most efficient, but
    /// can require a large amount of memory. The query is performed on batches
    /// of size `n` when `batch_size` is `Some(n)`. Setting this to a smaller
    /// value than the number of word embeddings reduces memory use at the
    /// cost of computational efficiency.
    fn embedding_similarity(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        batch_size: Option<usize>,
    ) -> Option<Vec<WordSimilarityResult>> {
        self.embedding_similarity_masked(query, limit, &HashSet::new(), batch_size)
    }

    /// Find words that are similar to the query embedding while skipping
    /// certain words.
    ///
    /// The similarity between the query embedding and other embeddings is
    /// defined by the dot product of the embeddings. The embeddings in the
    /// storage are l2-normalized, this method l2-normalizes the input query,
    /// therefore the dot product is equivalent to the cosine similarity.
    ///
    /// If `batch_size` is `None`, the query will be performed on all
    /// word embeddings at once. This is typically the most efficient, but
    /// can require a large amount of memory. The query is performed on batches
    /// of size `n` when `batch_size` is `Some(n)`. Setting this to a smaller
    /// value than the number of word embeddings reduces memory use at the
    /// cost of computational efficiency.
    fn embedding_similarity_masked(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skips: &HashSet<&str>,
        batch_size: Option<usize>,
    ) -> Option<Vec<WordSimilarityResult>>;
}

impl<V, S> EmbeddingSimilarity for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn embedding_similarity_masked(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skip: &HashSet<&str>,
        batch_size: Option<usize>,
    ) -> Option<Vec<WordSimilarityResult>> {
        let mut query = query.to_owned();
        l2_normalize(query.view_mut());
        Some(self.similarity_(query.view(), skip, limit, batch_size))
    }
}

trait SimilarityPrivate {
    fn similarity_(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        batch_size: Option<usize>,
    ) -> Vec<WordSimilarityResult>;
}

impl<V, S> SimilarityPrivate for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn similarity_(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        batch_size: Option<usize>,
    ) -> Vec<WordSimilarityResult> {
        let batch_size = batch_size.unwrap_or_else(|| self.vocab().words_len());

        let mut results = BinaryHeap::with_capacity(limit);

        for (batch_idx, batch) in self
            .storage()
            .view()
            .slice(s![0..self.vocab().words_len(), ..])
            .axis_chunks_iter(Axis(0), batch_size)
            .enumerate()
        {
            let sims = batch.dot(&embed.view());

            for (idx, &sim) in sims.iter().enumerate() {
                let word = &self.vocab().words()[(batch_idx * batch_size) + idx];

                // Don't add words that we are explicitly asked to skip.
                if skip.contains(word.as_str()) {
                    continue;
                }

                let word_similarity = WordSimilarityResult {
                    word,
                    similarity: NotNan::new(sim).expect("Encountered NaN"),
                };

                if results.len() < limit {
                    results.push(word_similarity);
                } else {
                    let mut peek = results.peek_mut().expect("Cannot peek non-empty heap");
                    if word_similarity < *peek {
                        *peek = word_similarity
                    }
                }
            }
        }

        results.into_sorted_vec()
    }
}

fn lookup_words3<'a, V, S>(
    embeddings: &'a Embeddings<V, S>,
    query: [&str; 3],
) -> Result<[CowArray<'a, f32, Ix1>; 3], [bool; 3]>
where
    V: Vocab,
    S: Storage,
{
    let embedding1 = embeddings.embedding(query[0]);
    let embedding2 = embeddings.embedding(query[1]);
    let embedding3 = embeddings.embedding(query[2]);

    let present = [
        embedding1.is_some(),
        embedding2.is_some(),
        embedding3.is_some(),
    ];

    if !present.iter().all(|&present| present) {
        return Err(present);
    }

    Ok([
        embedding1.unwrap(),
        embedding2.unwrap(),
        embedding3.unwrap(),
    ])
}

#[cfg(test)]
mod tests {

    use std::fs::File;
    use std::io::BufReader;

    use approx::AbsDiffEq;
    use ordered_float::NotNan;

    use crate::compat::word2vec::ReadWord2Vec;
    use crate::embeddings::Embeddings;
    use crate::similarity::{Analogy, EmbeddingSimilarity, WordSimilarity, WordSimilarityResult};

    static SIMILARITY_ORDER_STUTTGART_10: &[&str] = &[
        "Karlsruhe",
        "Mannheim",
        "München",
        "Darmstadt",
        "Heidelberg",
        "Wiesbaden",
        "Kassel",
        "Düsseldorf",
        "Leipzig",
        "Berlin",
    ];

    static SIMILARITY_ORDER: &[&str] = &[
        "Potsdam",
        "Hamburg",
        "Leipzig",
        "Dresden",
        "München",
        "Düsseldorf",
        "Bonn",
        "Stuttgart",
        "Weimar",
        "Berlin-Charlottenburg",
        "Rostock",
        "Karlsruhe",
        "Chemnitz",
        "Breslau",
        "Wiesbaden",
        "Hannover",
        "Mannheim",
        "Kassel",
        "Köln",
        "Danzig",
        "Erfurt",
        "Dessau",
        "Bremen",
        "Charlottenburg",
        "Magdeburg",
        "Neuruppin",
        "Darmstadt",
        "Jena",
        "Wien",
        "Heidelberg",
        "Dortmund",
        "Stettin",
        "Schwerin",
        "Neubrandenburg",
        "Greifswald",
        "Göttingen",
        "Braunschweig",
        "Berliner",
        "Warschau",
        "Berlin-Spandau",
    ];

    static ANALOGY_ORDER: &[&str] = &[
        "Deutschland",
        "Westdeutschland",
        "Sachsen",
        "Mitteldeutschland",
        "Brandenburg",
        "Polen",
        "Norddeutschland",
        "Dänemark",
        "Schleswig-Holstein",
        "Österreich",
        "Bayern",
        "Thüringen",
        "Bundesrepublik",
        "Ostdeutschland",
        "Preußen",
        "Deutschen",
        "Hessen",
        "Potsdam",
        "Mecklenburg",
        "Niedersachsen",
        "Hamburg",
        "Süddeutschland",
        "Bremen",
        "Russland",
        "Deutschlands",
        "BRD",
        "Litauen",
        "Mecklenburg-Vorpommern",
        "DDR",
        "West-Berlin",
        "Saarland",
        "Lettland",
        "Hannover",
        "Rostock",
        "Sachsen-Anhalt",
        "Pommern",
        "Schweden",
        "Deutsche",
        "deutschen",
        "Westfalen",
    ];

    #[test]
    fn cosine_similarity_is_correctly_converted_to_angular_similarity() {
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(1f32).unwrap()
        })
        .angular_similarity()
        .abs_diff_eq(&1f32, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0.70710678).unwrap()
        })
        .angular_similarity()
        .abs_diff_eq(&0.75, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0f32).unwrap()
        })
        .angular_similarity()
        .abs_diff_eq(&0.5f32, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(-1f32).unwrap()
        })
        .angular_similarity()
        .abs_diff_eq(&0f32, 1e-5));
    }

    #[test]
    fn cosine_similarity_is_correctly_converted_to_euclidean_distance() {
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(1f32).unwrap()
        })
        .euclidean_distance()
        .abs_diff_eq(&0f32, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0.70710678).unwrap()
        })
        .euclidean_distance()
        .abs_diff_eq(&0.76537, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0f32).unwrap()
        })
        .euclidean_distance()
        .abs_diff_eq(&2f32.sqrt(), 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(-1f32).unwrap()
        })
        .euclidean_distance()
        .abs_diff_eq(&2f32, 1e-5));
    }

    #[test]
    fn cosine_similarity_is_correctly_converted_to_euclidean_similarity() {
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(1f32).unwrap()
        })
        .euclidean_similarity()
        .abs_diff_eq(&1f32, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0.70710678).unwrap()
        })
        .euclidean_similarity()
        .abs_diff_eq(&0.61732, 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(0f32).unwrap()
        })
        .euclidean_similarity()
        .abs_diff_eq(&(1f32 - 1f32 / 2f32.sqrt()), 1e-5));
        assert!((WordSimilarityResult {
            word: "test",
            similarity: NotNan::new(-1f32).unwrap(),
        })
        .euclidean_similarity()
        .abs_diff_eq(&0f32, 1e-5));
    }

    #[test]
    fn test_similarity() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.word_similarity("Berlin", 40, None);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }

        let result = embeddings.word_similarity("Berlin", 10, None);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(10, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }

        let result = embeddings.word_similarity("Berlin", 40, Some(17));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_embedding_similarity() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
        let embedding = embeddings.embedding("Berlin").unwrap();
        let result = embeddings.embedding_similarity(embedding.view(), 10, None);
        assert!(result.is_some());
        let mut result = result.unwrap().into_iter();
        assert_eq!(10, result.len());
        assert_eq!(result.next().unwrap().word, "Berlin");

        for (idx, word_similarity) in result.into_iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_similarity_limit() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.word_similarity("Stuttgart", 10, None);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(10, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER_STUTTGART_10[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_analogy() {
        let f = File::open("testdata/analogy.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.analogy(["Paris", "Frankreich", "Berlin"], 40, None);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(ANALOGY_ORDER[idx], word_similarity.word)
        }

        let result = embeddings.analogy(["Paris", "Frankreich", "Berlin"], 40, Some(17));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(ANALOGY_ORDER[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_analogy_absent() {
        let f = File::open("testdata/analogy.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        assert_eq!(
            embeddings.analogy(["Foo", "Frankreich", "Berlin"], 40, None),
            Err([false, true, true])
        );
        assert_eq!(
            embeddings.analogy(["Paris", "Foo", "Berlin"], 40, None),
            Err([true, false, true])
        );
        assert_eq!(
            embeddings.analogy(["Paris", "Frankreich", "Foo"], 40, None),
            Err([true, true, false])
        );
    }
}
