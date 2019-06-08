//! Traits and trait implementations for similarity queries.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{s, Array1, ArrayView1, ArrayView2};
use ordered_float::NotNan;

use crate::embeddings::Embeddings;
use crate::storage::StorageView;
use crate::util::l2_normalize;
use crate::vocab::Vocab;

/// A word with its similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug, Eq, PartialEq)]
pub struct WordSimilarity<'a> {
    pub similarity: NotNan<f32>,
    pub word: &'a str,
}

impl<'a> Ord for WordSimilarity<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.similarity.cmp(&self.similarity) {
            Ordering::Equal => self.word.cmp(other.word),
            ordering => ordering,
        }
    }
}

impl<'a> PartialOrd for WordSimilarity<'a> {
    fn partial_cmp(&self, other: &WordSimilarity) -> Option<Ordering> {
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
    /// At most, `limit` results are returned.
    fn analogy(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
    ) -> Option<Vec<WordSimilarity>> {
        self.analogy_masked(word1, word2, word3, limit, [true, true, true])
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
    fn analogy_masked(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        remove: [bool; 3],
    ) -> Option<Vec<WordSimilarity>>;
}

impl<V, S> Analogy for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn analogy_masked(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        remove: [bool; 3],
    ) -> Option<Vec<WordSimilarity>> {
        {
            self.analogy_by_masked(word1, word2, word3, limit, remove, |embeds, embed| {
                embeds.dot(&embed)
            })
        }
    }
}
/// Trait for analogy queries with a custom similarity function.
pub trait AnalogyBy {
    /// Perform an analogy query using the given similarity function.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    fn analogy_by<F>(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        self.analogy_by_masked(word1, word2, word3, limit, [true, true, true], similarity)
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
    ///
    /// `remove` specifies which parts of the queries are excluded from the
    /// output candidates. If `remove[0]` is `true`, `word1` cannot be
    /// returned as an answer to the query.
    fn analogy_by_masked<F>(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        remove: [bool; 3],
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl<V, S> AnalogyBy for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn analogy_by_masked<F>(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        remove: [bool; 3],
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embedding1 = self.embedding(word1)?;
        let embedding2 = self.embedding(word2)?;
        let embedding3 = self.embedding(word3)?;

        let mut embedding = (&embedding2.as_view() - &embedding1.as_view()) + embedding3.as_view();
        l2_normalize(embedding.view_mut());

        let skip = [word1, word2, word3]
            .iter()
            .zip(remove.iter())
            .filter(|(_, &exclude)| exclude)
            .map(|(word, _)| word.to_owned())
            .collect();

        Some(self.similarity_(embedding.view(), &skip, limit, similarity))
    }
}

/// Trait for similarity queries.
pub trait Similarity {
    /// Find words that are similar to the query word.
    ///
    /// The similarity between two words is defined by the dot product of
    /// the embeddings. If the vectors are unit vectors (e.g. by virtue of
    /// calling `normalize`), this is the cosine similarity. At most, `limit`
    /// results are returned.
    fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>>;
}

impl<V, S> Similarity for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>> {
        self.similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }
}

/// Trait for similarity queries with a custom similarity function.
pub trait SimilarityBy {
    /// Find words that are similar to the query word using the given similarity
    /// function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the word vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    fn similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl<V, S> SimilarityBy for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embed = self.embedding(word)?;
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed.as_view(), &skip, limit, similarity))
    }
}

trait SimilarityPrivate {
    fn similarity_<F>(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        similarity: F,
    ) -> Vec<WordSimilarity>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl<V, S> SimilarityPrivate for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn similarity_<F>(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        mut similarity: F,
    ) -> Vec<WordSimilarity>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sims = similarity(
            self.storage().view().slice(s![0..self.vocab().len(), ..]),
            embed.view(),
        );

        let mut results = BinaryHeap::with_capacity(limit);
        for (idx, &sim) in sims.iter().enumerate() {
            let word = &self.vocab().words()[idx];

            // Don't add words that we are explicitly asked to skip.
            if skip.contains(word.as_str()) {
                continue;
            }

            let word_similarity = WordSimilarity {
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

        results.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {

    use std::fs::File;
    use std::io::BufReader;

    use crate::embeddings::Embeddings;
    use crate::similarity::{Analogy, Similarity};
    use crate::word2vec::ReadWord2Vec;

    static SIMILARITY_ORDER_STUTTGART_10: &'static [&'static str] = &[
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

    static SIMILARITY_ORDER: &'static [&'static str] = &[
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

    static ANALOGY_ORDER: &'static [&'static str] = &[
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
    fn test_similarity() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.similarity("Berlin", 40);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }

        let result = embeddings.similarity("Berlin", 10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(10, result.len());

        println!("{:?}", result);

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_similarity_limit() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.similarity("Stuttgart", 10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(10, result.len());

        println!("{:?}", result);

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER_STUTTGART_10[idx], word_similarity.word)
        }
    }

    #[test]
    fn test_analogy() {
        let f = File::open("testdata/analogy.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        let result = embeddings.analogy("Paris", "Frankreich", "Berlin", 40);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(ANALOGY_ORDER[idx], word_similarity.word)
        }
    }

}
