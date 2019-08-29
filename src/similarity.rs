//! Traits and trait implementations for similarity queries.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{s, Array1, ArrayView1, ArrayView2};
use ordered_float::NotNan;

use crate::chunks::storage::{CowArray1, Storage, StorageView};
use crate::chunks::vocab::Vocab;
use crate::embeddings::Embeddings;
use crate::util::l2_normalize;

/// A word with its similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug, Eq, PartialEq)]
pub struct WordSimilarityResult<'a> {
    pub similarity: NotNan<f32>,
    pub word: &'a str,
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
    fn analogy(
        &self,
        query: [&str; 3],
        limit: usize,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]> {
        self.analogy_masked(query, [true, true, true], limit)
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
    ///`Result::Err` is returned when no embedding could be computed
    /// for one or more of the tokens, indicating which of the tokens
    /// were present.
    fn analogy_masked(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
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
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]> {
        {
            self.analogy_by_masked(query, remove, limit, |embeds, embed| embeds.dot(&embed))
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
    ///
    ///`Result::Err` is returned when no embedding could be computed
    /// for one or more of the tokens, indicating which of the tokens
    /// were present.
    fn analogy_by<F>(
        &self,
        query: [&str; 3],
        limit: usize,
        similarity: F,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        self.analogy_by_masked(query, [true, true, true], limit, similarity)
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
    ///
    ///`Result::Err` is returned when no embedding could be computed
    /// for one or more of the tokens, indicating which of the tokens
    /// were present.
    fn analogy_by_masked<F>(
        &self,
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
        similarity: F,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]>
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
        query: [&str; 3],
        remove: [bool; 3],
        limit: usize,
        similarity: F,
    ) -> Result<Vec<WordSimilarityResult>, [bool; 3]>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let [embedding1, embedding2, embedding3] = lookup_words3(self, query)?;

        let mut embedding = (&embedding2.as_view() - &embedding1.as_view()) + embedding3.as_view();
        l2_normalize(embedding.view_mut());

        let skip = query
            .iter()
            .zip(remove.iter())
            .filter(|(_, &exclude)| exclude)
            .map(|(word, _)| word.to_owned())
            .collect();

        Ok(self.similarity_(embedding.view(), &skip, limit, similarity))
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
    fn word_similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarityResult>>;

    /// Find words that are similar to the query word using the given similarity
    /// function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the word vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    fn word_similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl<V, S> WordSimilarity for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn word_similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarityResult>> {
        self.word_similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }

    fn word_similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embed = self.embedding(word)?;
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed.as_view(), &skip, limit, similarity))
    }
}

/// Trait for embedding similarity queries.
pub trait EmbeddingSimilarity {
    /// Find words that are similar to the query embedding.
    ///
    /// The similarity between the query embedding and other embeddings is
    /// defined by the dot product of the embeddings. If the vectors are unit
    /// vectors (e.g. by virtue of calling `normalize`), this is the cosine
    /// similarity. At most, `limit` results are returned.
    fn embedding_similarity(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
    ) -> Option<Vec<WordSimilarityResult>> {
        self.embedding_similarity_masked(query, limit, &HashSet::new())
    }

    /// Find words that are similar to the query embedding while skipping
    /// certain words.
    ///
    /// The similarity between the query embedding and other embeddings is
    /// defined by the dot product of the embeddings. If the vectors are unit
    /// vectors (e.g. by virtue of calling `normalize`), this is the cosine
    /// similarity. At most, `limit` results are returned.
    fn embedding_similarity_masked(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skips: &HashSet<&str>,
    ) -> Option<Vec<WordSimilarityResult>>;

    /// Find words that are similar to the query embedding using the given
    /// similarity function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the query vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    fn embedding_similarity_by<F>(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skip: &HashSet<&str>,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
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
    ) -> Option<Vec<WordSimilarityResult>> {
        self.embedding_similarity_by(query, limit, skip, |embeds, embed| embeds.dot(&embed))
    }

    fn embedding_similarity_by<F>(
        &self,
        query: ArrayView1<f32>,
        limit: usize,
        skip: &HashSet<&str>,
        similarity: F,
    ) -> Option<Vec<WordSimilarityResult>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        Some(self.similarity_(query, skip, limit, similarity))
    }
}

trait SimilarityPrivate {
    fn similarity_<F>(
        &self,
        embed: ArrayView1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        similarity: F,
    ) -> Vec<WordSimilarityResult>
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
    ) -> Vec<WordSimilarityResult>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sims = similarity(
            self.storage()
                .view()
                .slice(s![0..self.vocab().words_len(), ..]),
            embed.view(),
        );

        let mut results = BinaryHeap::with_capacity(limit);
        for (idx, &sim) in sims.iter().enumerate() {
            let word = &self.vocab().words()[idx];

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

        results.into_sorted_vec()
    }
}

fn lookup_words3<'a, V, S>(
    embeddings: &'a Embeddings<V, S>,
    query: [&str; 3],
) -> Result<[CowArray1<'a, f32>; 3], [bool; 3]>
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

    use crate::compat::word2vec::ReadWord2Vec;
    use crate::embeddings::Embeddings;
    use crate::similarity::{Analogy, EmbeddingSimilarity, WordSimilarity};

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

        let result = embeddings.word_similarity("Berlin", 40);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(40, result.len());

        for (idx, word_similarity) in result.iter().enumerate() {
            assert_eq!(SIMILARITY_ORDER[idx], word_similarity.word)
        }

        let result = embeddings.word_similarity("Berlin", 10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(10, result.len());

        println!("{:?}", result);

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
        let result = embeddings.embedding_similarity(embedding.as_view(), 10);
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

        let result = embeddings.word_similarity("Stuttgart", 10);
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

        let result = embeddings.analogy(["Paris", "Frankreich", "Berlin"], 40);
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
            embeddings.analogy(["Foo", "Frankreich", "Berlin"], 40),
            Err([false, true, true])
        );
        assert_eq!(
            embeddings.analogy(["Paris", "Foo", "Berlin"], 40),
            Err([true, false, true])
        );
        assert_eq!(
            embeddings.analogy(["Paris", "Frankreich", "Foo"], 40),
            Err([true, true, false])
        );
    }

}
