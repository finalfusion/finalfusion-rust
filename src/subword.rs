use std::cmp;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use fnv::FnvHasher;

use crate::util::CollectWithCapacity;

/// A string reference with its length in characters.
struct StrWithCharLen<'a> {
    inner: &'a str,
    char_len: usize,
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
        self.char_len.hash(hasher);
        self.inner.chars().for_each(|ch| ch.hash(hasher));
    }
}

/// Iterator over n-grams in a sequence.
///
/// N-grams provides an iterator over the n-grams in a sentence between a
/// minimum and maximum length.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams are produced.
struct NGrams<'a> {
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
            min_n,
            max_n,
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
}

/// Extension trait for computing subword indices.
///
/// Subword indexing assigns an identifier to each subword (n-gram) of a
/// string. A subword is indexed by computing its hash and then mapping
/// the hash to a bucket.
///
/// Since a non-perfect hash function is used, multiple subwords can
/// map to the same index.
pub trait SubwordIndices {
    /// Return the subword indices of the subwords of a string.
    ///
    /// The n-grams that are used are of length *[min_n, max_n]*, these are
    /// mapped to indices into *2^buckets_exp* buckets.
    ///
    /// The largest possible bucket exponent is 64.
    fn subword_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<u64>;
}

impl SubwordIndices for str {
    fn subword_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<u64> {
        NGramsIndicesIter::new(self, min_n, max_n, buckets_exp)
            .map(|(_, idx)| idx)
            .collect()
    }
}

/// Iterator over the n-grams in a word and the corresponding subword indices.
///
/// `NGramsIndicesIter` is an iterator that produces the n-grams in a word and
/// the corresponding subword indices as tuples `(ngram, index)`.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams and their indices are produced.
struct NGramsIndicesIter<'a> {
    ngrams: NGrams<'a>,
    mask: u64,
}

impl<'a> NGramsIndicesIter<'a> {
    /// Create a new ngrams-indices iterator.
    ///
    /// The iterator will create all ngrams of length *[min_n, max_n]* and corresponding
    /// subword indices.
    pub fn new(string: &'a str, min_n: usize, max_n: usize, buckets_exp: usize) -> Self {
        assert!(
            buckets_exp <= 64,
            "The largest possible buckets exponent is 64."
        );

        let mask = if buckets_exp == 64 {
            !0
        } else {
            (1 << buckets_exp) - 1
        };

        let ngrams = NGrams::new(string, min_n, max_n);
        NGramsIndicesIter { ngrams, mask }
    }
}

impl<'a> Iterator for NGramsIndicesIter<'a> {
    type Item = (&'a str, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.ngrams.next().map(|ngram| {
            let mut hasher = FnvHasher::default();
            ngram.hash(&mut hasher);
            (ngram.inner, hasher.finish() & self.mask)
        })
    }
}

/// A trait for getting ngrams and their indices of a string.
///
/// N-gram indexing assigns an identifier to each subword (n-gram) of a
/// string. A subword is indexed by computing its hash and then mapping
/// the hash to a bucket.
///
/// Since a non-perfect hash function is used, multiple subwords can
/// map to the same index.
pub trait NGramsIndices {
    /// Return the ngrams and their indices of a string.
    ///
    /// The n-grams that are used are of length *[min_n, max_n]*, these are
    /// mapped to indices into *2^buckets_exp* buckets.
    ///
    /// The largest possible bucket exponent is 64.
    fn ngrams_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(&str, u64)>;
}

impl NGramsIndices for str {
    fn ngrams_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(&str, u64)> {
        NGramsIndicesIter::new(self, min_n, max_n, buckets_exp).collect()
    }
}

#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use maplit::hashmap;
    use std::collections::HashMap;

    use super::{NGrams, NGramsIndices, SubwordIndices};

    #[test]
    fn ngrams_test() {
        let mut hello_check: Vec<&str> = vec![
            "h", "he", "hel", "e", "el", "ell", "l", "ll", "llö", "l", "lö", "lö ", "ö", "ö ",
            "ö w", " ", " w", " wo", "w", "wo", "wor", "o", "or", "orl", "r", "rl", "rld", "l",
            "ld", "d",
        ];

        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new("hellö world", 1, 3).map(|s| s.inner).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let mut hello_check: Vec<&str> = vec![
            "he", "hel", "el", "ell", "ll", "llo", "lo", "lo ", "o ", "o w", " w", " wo", "wo",
            "wor", "or", "orl", "rl", "rld", "ld",
        ];

        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new("hello world", 2, 3).map(|s| s.inner).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn short_ngram_test() {
        let mut yep_check: Vec<&str> = vec!["ˈjə", "jəp", "ˈjəp"];
        yep_check.sort();

        let mut yep_ngrams: Vec<_> = NGrams::new("ˈjəp", 3, 6).map(|s| s.inner).collect();
        yep_ngrams.sort();

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
        static ref NGRAMS_INDICES_TESTS_36: HashMap<&'static str, Vec<(&'static str, u64)>> = [
            (
                "<Daniël>",
                vec![
                    ("Dan", 214157),
                    ("iël", 233912),
                    ("Danië", 311961),
                    ("iël>", 488897),
                    ("niël>", 620206),
                    ("anië", 741276),
                    ("Dani", 841219),
                    ("Daniël", 1167494),
                    ("ani", 1192256),
                    ("niël", 1489905),
                    ("ël>", 1532271),
                    ("nië", 1644730),
                    ("<Dan", 1666166),
                    ("aniël", 1679745),
                    ("<Danië", 1680294),
                    ("aniël>", 1693100),
                    ("<Da", 2026735),
                    ("<Dani", 2065822)
                ]
            ),
            (
                "<hallo>",
                vec![
                    ("lo>", 75867),
                    ("<hal", 104120),
                    ("hallo>", 136555),
                    ("hal", 456131),
                    ("allo>", 599360),
                    ("llo", 722393),
                    ("all", 938007),
                    ("<ha", 985859),
                    ("hallo", 1006102),
                    ("allo", 1163391),
                    ("llo>", 1218704),
                    ("<hallo", 1321513),
                    ("<hall", 1505861),
                    ("hall", 1892376)
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

        for (word, indices_check) in SUBWORD_TESTS_2.iter() {
            let mut indices = word.subword_indices(3, 6, 2);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn subword_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the subword_indices() method hashes
        // to the same buckets in the future.

        for (word, indices_check) in SUBWORD_TESTS_21.iter() {
            let mut indices = word.subword_indices(3, 6, 21);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn ngrams_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the ngrams_indices() method hashes
        // to the same buckets in the future.

        for (word, ngrams_indices_check) in NGRAMS_INDICES_TESTS_36.iter() {
            let mut ngrams_indices_test = word.ngrams_indices(3, 6, 21);
            ngrams_indices_test.sort_by_key(|ngrams_indices_pairs| ngrams_indices_pairs.1);
            for (iter_check, iter_test) in ngrams_indices_check.into_iter().zip(ngrams_indices_test)
            {
                assert_eq!(iter_check.0, iter_test.0);
            }
        }
    }
}