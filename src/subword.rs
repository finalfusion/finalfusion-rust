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
        assert!(
            buckets_exp <= 64,
            "The largest possible buckets exponent is 64."
        );

        let mask = if buckets_exp == 64 {
            !0
        } else {
            (1 << buckets_exp) - 1
        };

        // Rough approximation, is correct for ASCII, avoids resizes
        // when the string contains non-ASCII characters.
        let mut indices = Vec::with_capacity((max_n - min_n + 1) * self.len());
        for ngram in NGrams::new(self, min_n, max_n) {
            let mut hasher = FnvHasher::default();
            ngram.hash(&mut hasher);
            indices.push(hasher.finish() & mask);
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use maplit::hashmap;
    use std::collections::HashMap;

    use super::{NGrams, SubwordIndices};

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
}
