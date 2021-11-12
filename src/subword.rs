//! Utilities for subword units.

use std::cmp;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;

use fnv::FnvHasher;
use smallvec::{smallvec, SmallVec};

use crate::util::CollectWithCapacity;

pub type NGramVec = SmallVec<[u64; 4]>;

/// N-Gram indexer
///
/// An indexer maps an n-gram to an index in the subword embedding
/// matrix.
pub trait Indexer {
    /// Map an n-gram to indices in the subword embedding matrix.
    fn index_ngram(&self, ngram: &StrWithCharLen) -> NGramVec;

    /// Return the (exclusive) upper bound of this indexer.
    fn upper_bound(&self) -> u64;

    /// Indicates whether this Indexer never fails to produce an index.
    fn infallible() -> bool;
}

/// N-Gram indexer with bucketing.
pub trait BucketIndexer: Indexer {
    /// Create a new indexer.
    ///
    /// The buckets argument is the number of buckets or the
    /// bucket exponent (depending on the implementation).
    fn new(buckets: usize) -> Self;

    /// Get the number of buckets.
    ///
    /// Depending on the indexer, this may be the actual number of
    /// buckets or the bucket exponent.
    fn buckets(&self) -> usize;
}

/// Indexer using a hash function.
///
/// This indexer first hashes a given n-gram and then maps the
/// resulting hash into *2^buckets_exp* buckets.
///
/// The largest possible bucket exponent is 64.
pub struct HashIndexer<H> {
    buckets_exp: usize,
    mask: u64,
    _phantom: PhantomData<H>,
}

impl<H> BucketIndexer for HashIndexer<H>
where
    H: Default + Hasher,
{
    /// Construct a `HashIndexer`.
    ///
    /// The largest possible bucket exponent is 64.
    fn new(buckets_exp: usize) -> Self {
        assert!(
            buckets_exp <= 64,
            "The largest possible buckets exponent is 64."
        );

        let mask = if buckets_exp == 64 {
            !0
        } else {
            (1 << buckets_exp) - 1
        };

        HashIndexer {
            buckets_exp,
            mask,
            _phantom: PhantomData,
        }
    }

    fn buckets(&self) -> usize {
        self.buckets_exp as usize
    }
}

impl<H> Clone for HashIndexer<H> {
    fn clone(&self) -> Self {
        HashIndexer {
            buckets_exp: self.buckets_exp,
            mask: self.mask,
            _phantom: PhantomData,
        }
    }
}

impl<H> Copy for HashIndexer<H> {}

impl<H> fmt::Debug for HashIndexer<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        // std::intrinsics::type_name requires nightly
        write!(f, "HashIndexer<impl Hasher> {{ mask: {} }}", self.mask)
    }
}

impl<H> Eq for HashIndexer<H> {}

impl<H> Indexer for HashIndexer<H>
where
    H: Default + Hasher,
{
    fn index_ngram(&self, ngram: &StrWithCharLen) -> NGramVec {
        let mut hasher = H::default();
        ngram.hash(&mut hasher);
        smallvec![hasher.finish() & self.mask]
    }

    fn upper_bound(&self) -> u64 {
        // max val is <= 64
        2u64.pow(self.buckets_exp as u32)
    }

    fn infallible() -> bool {
        true
    }
}

impl<H> PartialEq for HashIndexer<H> {
    fn eq(&self, other: &Self) -> bool {
        self.mask.eq(&other.mask)
    }
}

/// Standard hash-based indexer in finalfusion.
pub type FinalfusionHashIndexer = HashIndexer<FnvHasher>;

/// Indexer for explicitly stored NGrams.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplicitIndexer {
    ngrams: Vec<String>,
    index: HashMap<String, u64>,
    bound: usize,
}

impl ExplicitIndexer {
    pub fn ngrams(&self) -> &[String] {
        &self.ngrams
    }
}

impl ExplicitIndexer {
    /// Construct a new explicit indexer.
    ///
    /// Panics when there are duplicate ngrams.
    pub fn new(ngrams: impl Into<Vec<String>>) -> Self {
        let ngrams = ngrams.into();
        let index = ngrams
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, ngram)| (ngram, idx as u64))
            .collect::<HashMap<String, u64>>();
        assert_eq!(
            index.len(),
            ngrams.len(),
            "ngrams contained duplicate entries."
        );
        let bound = index.len();
        ExplicitIndexer {
            ngrams,
            index,
            bound,
        }
    }

    /// Construct a new explicit indexer with given indices.
    ///
    /// The `(String, u64)` tuples resemble the original `subword -> index` mapping. This mapping
    /// does not need to be perfect, i.e. multiple subwords can map to the same index as it is
    /// common with bucketed indexing.
    ///
    /// This constructor numbers the original indices as they appear and assigns a new index
    /// accordingly. After construction, subwords that originally had the same index will still be
    /// indexed by a common number. It is guaranteed that the new indices cover
    /// `(0..n_original_indices)` where `n_original_indices` is the number of unique indices in the
    /// `subword -> index` mapping.
    ///
    /// The second item in the returned tuple holds the `provided_index -> new_index` mapping.
    /// I.e.: the `i`th unique `provided_index` in `ngram_tuples` is mapped to the `new_index` `i`.
    ///
    /// Panics when there are duplicate ngrams.
    pub fn new_with_indices(
        ngram_tuples: impl IntoIterator<Item = (String, u64)>,
    ) -> (Self, HashMap<u64, usize>) {
        let ngram_tuples = ngram_tuples.into_iter();
        let mut old_to_new_indices = HashMap::with_capacity(ngram_tuples.size_hint().0);
        let mut index = HashMap::with_capacity(ngram_tuples.size_hint().0);
        let mut ngrams = Vec::with_capacity(ngram_tuples.size_hint().0);
        for (ngram, bucket) in ngram_tuples {
            let cur_idx = old_to_new_indices.len();
            let new_idx = *old_to_new_indices.entry(bucket).or_insert(cur_idx);
            assert!(
                index.insert(ngram.clone(), new_idx as u64).is_none(),
                "ngrams contains duplicate entries."
            );
            ngrams.push(ngram);
        }
        let bound = old_to_new_indices.len();
        (
            ExplicitIndexer {
                ngrams,
                index,
                bound,
            },
            old_to_new_indices,
        )
    }
}

impl Indexer for ExplicitIndexer {
    fn index_ngram(&self, ngram: &StrWithCharLen) -> NGramVec {
        match self.index.get(ngram.inner) {
            Some(&idx) => smallvec![idx],
            None => smallvec![],
        }
    }

    fn upper_bound(&self) -> u64 {
        self.bound as u64
    }

    fn infallible() -> bool {
        false
    }
}

/// A string reference with its length in characters.
pub struct StrWithCharLen<'a> {
    inner: &'a str,
    char_len: usize,
}

impl<'a> From<&'a str> for StrWithCharLen<'a> {
    fn from(s: &'a str) -> Self {
        StrWithCharLen::new(s)
    }
}

impl<'a> StrWithCharLen<'a> {
    /// Construct `StrWithCharLen`.
    ///
    /// Counts the number of chars in a `&str` and constructs a `StrWithCharLen` from it.
    pub fn new(s: &'a str) -> Self {
        let char_len = s.chars().count();
        StrWithCharLen { inner: s, char_len }
    }

    pub fn as_str(&self) -> &str {
        self.inner
    }

    pub fn char_len(&self) -> usize {
        self.char_len
    }
}

impl<'a> Deref for StrWithCharLen<'a> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a> Hash for StrWithCharLen<'a> {
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        hasher.write(&(self.char_len as u64).to_le_bytes());
        self.inner
            .chars()
            .for_each(|ch| hasher.write(&(ch as u32).to_le_bytes()));
    }
}

/// Iterator over n-grams in a sequence.
///
/// N-grams provides an iterator over the n-grams in a sentence between a
/// minimum and maximum length.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams are produced.
pub struct NGrams<'a> {
    max_n: usize,
    min_n: usize,
    string: &'a str,
    char_offsets: VecDeque<usize>,
    ngram_len: usize,
}

impl<'a> NGrams<'a> {
    /// Create a new n-ngram iterator.
    ///
    /// The iterator will create n-ngrams of length *[min_n, max_n]*
    pub fn new(string: &'a str, min_n: usize, max_n: usize) -> Self {
        assert!(min_n != 0, "The minimum n-gram length cannot be zero.");
        assert!(
            min_n <= max_n,
            "The maximum length should be equal to or greater than the minimum length."
        );

        // Get the byte offsets of the characters in `string`.
        let char_offsets = string
            .char_indices()
            .map(|(idx, _)| idx)
            .collect_with_capacity::<VecDeque<_>>(string.len());

        let ngram_len = cmp::min(max_n, char_offsets.len());

        NGrams {
            max_n,
            min_n,
            string,
            char_offsets,
            ngram_len,
        }
    }
}

impl<'a> Iterator for NGrams<'a> {
    type Item = StrWithCharLen<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // If the n-grams for the current suffix are exhausted,
        // move to the next suffix.
        if self.ngram_len < self.min_n {
            // Remove first character, to get the next suffix.
            self.char_offsets.pop_front();

            // If the suffix is smaller than the minimal n-gram
            // length, the iterator is exhausted.
            if self.char_offsets.len() < self.min_n {
                return None;
            }

            // Get the maximum n-gram length for this suffix.
            self.ngram_len = cmp::min(self.max_n, self.char_offsets.len());
        }

        let ngram = if self.ngram_len == self.char_offsets.len() {
            &self.string[self.char_offsets[0]..]
        } else {
            &self.string[self.char_offsets[0]..self.char_offsets[self.ngram_len]]
        };

        let ngram_with_len = StrWithCharLen {
            inner: ngram,
            char_len: self.ngram_len,
        };

        self.ngram_len -= 1;

        Some(ngram_with_len)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let cap_approx = (self.max_n - self.min_n + 1) * self.char_offsets.len();
        (cap_approx, Some(cap_approx))
    }
}

/// What to compute subwords for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndicesScope {
    /// Only substring indices.
    Substrings,

    /// String and substring indices.
    StringAndSubstrings,
}

/// Trait returning iterators over subwords and indices.
///
/// Defines methods to iterate over the subwords and
/// their corresponding indices as assigned through the
/// given `Indexer`. The `Indexer` can allow collisions.
pub trait SubwordIndices<'a, 'b, I>
where
    I: Indexer + 'b,
{
    type Iter: Iterator<Item = (&'a str, NGramVec)>;

    /// Return an iterator over the subword indices of a string.
    ///
    /// The n-grams that are used are of length *[min_n, max_n]*, these are
    /// mapped to indices using the given indexer.
    fn subword_indices(
        &'a self,
        min_n: usize,
        max_n: usize,
        indexer: &'b I,
        scope: IndicesScope,
    ) -> Box<dyn Iterator<Item = u64> + 'a>
    where
        'b: 'a,
    {
        Box::new(
            self.subword_indices_with_ngrams(min_n, max_n, indexer, scope)
                .flat_map(|(_, indices)| indices),
        )
    }

    /// Return an iterator over the subwords and subword indices of a string.
    ///
    /// The n-grams that are used are of length *[min_n, max_n]*, these are
    /// mapped to indices using the given indexer.
    fn subword_indices_with_ngrams(
        &'a self,
        min_n: usize,
        max_n: usize,
        indexer: &'b I,
        scope: IndicesScope,
    ) -> Self::Iter;
}

impl<'a, 'b, I> SubwordIndices<'a, 'b, I> for str
where
    I: Indexer + 'b,
{
    type Iter = NGramsIndicesIter<'a, 'b, I>;
    fn subword_indices_with_ngrams(
        &'a self,
        min_n: usize,
        max_n: usize,
        indexer: &'b I,
        scope: IndicesScope,
    ) -> Self::Iter {
        NGramsIndicesIter::new(self, min_n, max_n, indexer, scope)
    }
}

/// Iterator over the n-grams in a word and the corresponding subword indices.
///
/// `NGramsIndicesIter` is an iterator that produces the n-grams in a word and
/// the corresponding subword indices as tuples `(ngram, index)`.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams and their indices are produced.
pub struct NGramsIndicesIter<'a, 'b, I> {
    string: Option<&'a str>,
    ngrams: NGrams<'a>,
    indexer: &'b I,
}

impl<'a, 'b, I> NGramsIndicesIter<'a, 'b, I> {
    /// Create a new ngrams-indices iterator.
    ///
    /// The iterator will create all ngrams of length *[min_n, max_n]* and corresponding
    /// subword indices.
    pub fn new(
        string: &'a str,
        min_n: usize,
        max_n: usize,
        indexer: &'b I,
        scope: IndicesScope,
    ) -> Self {
        let ngrams = NGrams::new(string, min_n, max_n);
        let string = match scope {
            IndicesScope::Substrings => None,
            IndicesScope::StringAndSubstrings => Some(string),
        };

        NGramsIndicesIter {
            ngrams,
            indexer,
            string,
        }
    }
}

impl<'a, 'b, I> Iterator for NGramsIndicesIter<'a, 'b, I>
where
    I: Indexer,
{
    type Item = (&'a str, NGramVec);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(string) = self.string {
            self.string = None;
            return Some((string, self.indexer.index_ngram(&string.into())));
        }

        self.ngrams
            .next()
            .map(|ngram| (ngram.inner, self.indexer.index_ngram(&ngram)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ngrams.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::chunks::vocab::IndicesScope;
    use lazy_static::lazy_static;
    use maplit::hashmap;
    use smallvec::smallvec;

    use super::{BucketIndexer, FinalfusionHashIndexer, NGrams, SubwordIndices};
    use crate::subword::NGramVec;

    #[test]
    fn ngrams_test() {
        let mut hello_check: Vec<&str> = vec![
            "h", "he", "hel", "e", "el", "ell", "l", "ll", "llö", "l", "lö", "lö ", "ö", "ö ",
            "ö w", " ", " w", " wo", "w", "wo", "wor", "o", "or", "orl", "r", "rl", "rld", "l",
            "ld", "d",
        ];

        hello_check.sort_unstable();

        let mut hello_ngrams: Vec<_> = NGrams::new("hellö world", 1, 3).map(|s| s.inner).collect();
        hello_ngrams.sort_unstable();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let mut hello_check: Vec<&str> = vec![
            "he", "hel", "el", "ell", "ll", "llo", "lo", "lo ", "o ", "o w", " w", " wo", "wo",
            "wor", "or", "orl", "rl", "rld", "ld",
        ];

        hello_check.sort_unstable();

        let mut hello_ngrams: Vec<_> = NGrams::new("hello world", 2, 3).map(|s| s.inner).collect();
        hello_ngrams.sort_unstable();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn short_ngram_test() {
        let mut yep_check: Vec<&str> = vec!["ˈjə", "jəp", "ˈjəp"];
        yep_check.sort_unstable();

        let mut yep_ngrams: Vec<_> = NGrams::new("ˈjəp", 3, 6).map(|s| s.inner).collect();
        yep_ngrams.sort_unstable();

        assert_eq!(yep_check, yep_ngrams);
    }

    #[test]
    fn empty_ngram_test() {
        let check: &[&str] = &[];
        assert_eq!(
            NGrams::new("", 1, 3).map(|s| s.inner).collect::<Vec<_>>(),
            check
        );
    }

    #[test]
    #[should_panic]
    fn incorrect_min_n_test() {
        NGrams::new("", 0, 3);
    }

    #[test]
    #[should_panic]
    fn incorrect_max_n_test() {
        NGrams::new("", 2, 1);
    }

    lazy_static! {
        static ref SUBWORD_TESTS_2: HashMap<&'static str, Vec<u64>> = hashmap! {
            "<Daniël>" =>
                vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
            "<hallo>" =>
                vec![0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3],
        };
    }

    lazy_static! {
        static ref SUBWORD_TESTS_21: HashMap<&'static str, Vec<u64>> = hashmap! {
            "<Daniël>" =>
                vec![214157, 233912, 311961, 488897, 620206, 741276, 841219,
                     1167494, 1192256, 1489905, 1532271, 1644730, 1666166,
                     1679745, 1680294, 1693100, 2026735, 2065822],
            "<hallo>" =>
                vec![75867, 104120, 136555, 456131, 599360, 722393, 938007,
                     985859, 1006102, 1163391, 1218704, 1321513, 1505861,
                     1892376],
        };
    }

    lazy_static! {
        static ref NGRAMS_INDICES_TESTS_36: HashMap<&'static str, Vec<(&'static str, NGramVec)>> =
            [
                (
                    "<Daniël>",
                    vec![
                        ("Dan", smallvec![214157]),
                        ("iël", smallvec![233912]),
                        ("Danië", smallvec![311961]),
                        ("iël>", smallvec![488897]),
                        ("niël>", smallvec![620206]),
                        ("anië", smallvec![741276]),
                        ("Dani", smallvec![841219]),
                        ("Daniël", smallvec![1167494]),
                        ("ani", smallvec![1192256]),
                        ("niël", smallvec![1489905]),
                        ("ël>", smallvec![1532271]),
                        ("nië", smallvec![1644730]),
                        ("<Dan", smallvec![1666166]),
                        ("aniël", smallvec![1679745]),
                        ("<Danië", smallvec![1680294]),
                        ("aniël>", smallvec![1693100]),
                        ("<Da", smallvec![2026735]),
                        ("<Dani", smallvec![2065822])
                    ]
                ),
                (
                    "<hallo>",
                    vec![
                        ("lo>", smallvec![75867]),
                        ("<hal", smallvec![104120]),
                        ("hallo>", smallvec![136555]),
                        ("hal", smallvec![456131]),
                        ("allo>", smallvec![599360]),
                        ("llo", smallvec![722393]),
                        ("all", smallvec![938007]),
                        ("<ha", smallvec![985859]),
                        ("hallo", smallvec![1006102]),
                        ("allo", smallvec![1163391]),
                        ("llo>", smallvec![1218704]),
                        ("<hallo", smallvec![1321513]),
                        ("<hall", smallvec![1505861]),
                        ("hall", smallvec![1892376])
                    ]
                )
            ]
            .iter()
            .cloned()
            .collect();
    }

    #[test]
    fn subword_indices_4_test() {
        // The goal of this test is to ensure that we are correctly bucketing
        // subwords. With a bucket exponent of 2, there are 2^2 = 4 buckets,
        // so we should see bucket numbers [0..3].

        let indexer = FinalfusionHashIndexer::new(2);
        for (word, indices_check) in SUBWORD_TESTS_2.iter() {
            let mut indices = word
                .subword_indices(3, 6, &indexer, IndicesScope::Substrings)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn subword_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the subword_indices() method hashes
        // to the same buckets in the future.

        let indexer = FinalfusionHashIndexer::new(21);
        for (word, indices_check) in SUBWORD_TESTS_21.iter() {
            let mut indices = word
                .subword_indices(3, 6, &indexer, IndicesScope::Substrings)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn ngrams_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the ngrams_indices() method hashes
        // to the same buckets in the future.

        let indexer = FinalfusionHashIndexer::new(21);
        for (word, ngrams_indices_check) in NGRAMS_INDICES_TESTS_36.iter() {
            let mut ngrams_indices_test = word
                .subword_indices_with_ngrams(3, 6, &indexer, IndicesScope::Substrings)
                .collect::<Vec<_>>();
            ngrams_indices_test.sort_by_key(|ngrams_indices_pairs| ngrams_indices_pairs.1.clone());
            for (iter_check, iter_test) in ngrams_indices_check.into_iter().zip(ngrams_indices_test)
            {
                assert_eq!(iter_check.0, iter_test.0);
            }
        }
    }
}
