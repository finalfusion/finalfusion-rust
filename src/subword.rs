use std::cmp;
use std::hash::{Hash, Hasher};

use fnv::FnvHasher;

/// Iterator over n-grams in a sequence.
///
/// N-grams provides an iterator over the n-grams in a sentence between a
/// minimum and maximum length.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams are produced.
pub struct NGrams<'a, T>
where
    T: 'a,
{
    max_n: usize,
    min_n: usize,
    seq: &'a [T],
    ngram: &'a [T],
}

impl<'a, T> NGrams<'a, T> {
    /// Create a new n-ngram iterator.
    ///
    /// The iterator will create n-ngrams of length *[min_n, max_n]*
    pub fn new(seq: &'a [T], min_n: usize, max_n: usize) -> Self {
        assert!(min_n != 0, "The minimum n-gram length cannot be zero.");
        assert!(
            min_n <= max_n,
            "The maximum length should be equal to or greater than the minimum length."
        );

        let upper = cmp::min(max_n, seq.len());

        NGrams {
            min_n,
            max_n,
            seq,
            ngram: &seq[..upper],
        }
    }
}

impl<'a, T> Iterator for NGrams<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ngram.len() < self.min_n {
            if self.seq.len() <= self.min_n {
                return None;
            }

            self.seq = &self.seq[1..];

            let upper = cmp::min(self.max_n, self.seq.len());
            self.ngram = &self.seq[..upper];
        }

        let ngram = self.ngram;

        self.ngram = &self.ngram[..self.ngram.len() - 1];

        Some(ngram)
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
    fn ngrams_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(String, u64)>;
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

        let chars: Vec<_> = self.chars().collect();

        let mut indices = Vec::with_capacity((max_n - min_n + 1) * chars.len());
        for ngram in NGrams::new(&chars, min_n, max_n) {
            let mut hasher = FnvHasher::default();
            ngram.hash(&mut hasher);
            indices.push(hasher.finish() & mask);
        }

        indices
    }

    fn ngrams_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(String, u64)> {
        assert!(
            buckets_exp <= 64,
            "The largest possible buckets exponent is 64."
        );

        let mask = if buckets_exp == 64 {
            !0
        } else {
            (1 << buckets_exp) - 1
        };
        let chars: Vec<_> = self.chars().collect();

        let mut ngrams_indices = Vec::with_capacity((max_n - min_n + 1) * chars.len());
        for ngram in NGrams::new(&chars, min_n, max_n) {
            let mut hasher = FnvHasher::default();
            ngram.hash(&mut hasher);
            ngrams_indices.push((ngram.into_iter().collect(), hasher.finish() & mask));
        }

        ngrams_indices
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
        let hello_chars: Vec<_> = "hellö world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h'],
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l'],
            &['l', 'l'],
            &['l', 'l', 'ö'],
            &['l'],
            &['l', 'ö'],
            &['l', 'ö', ' '],
            &['ö'],
            &['ö', ' '],
            &['ö', ' ', 'w'],
            &[' '],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l'],
            &['l', 'd'],
            &['d'],
        ];

        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 1, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l', 'd'],
        ];
        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 2, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn empty_ngram_test() {
        let check: &[&[char]] = &[];
        assert_eq!(NGrams::<char>::new(&[], 1, 3).collect::<Vec<_>>(), check);
    }

    #[test]
    #[should_panic]
    fn incorrect_min_n_test() {
        NGrams::<char>::new(&[], 0, 3);
    }

    #[test]
    #[should_panic]
    fn incorrect_max_n_test() {
        NGrams::<char>::new(&[], 2, 1);
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
        static ref NGRAMS_INDICES_TESTS_36: HashMap<&'static str, Vec<(&'static str, u64)>> = hashmap! {
            "<Daniël>" =>
                vec![("Dan",214157), ("iël",233912), ("Danië",311961), ("iël>",488897), ("niël>",620206), ("anië",741276), ("Dani",841219),
                     ("Daniël",1167494), ("ani",1192256), ("niël",1489905), ("ël>",1532271), ("nië",1644730), ("<Dan",1666166),
                     ("aniël",1679745), ("<Danië",1680294), ("aniël>",1693100), ("<Da",2026735), ("<Dani",2065822)],
            "<hallo>" =>
                vec![("lo>",75867), ("<hal",104120), ("hallo>",136555), ("hal",456131), ("allo>",599360), ("llo",722393), ("all",938007),
                     ("<ha",985859), ("hallo",1006102), ("allo",1163391), ("llo>",1218704), ("<hallo",1321513), ("<hall",1505861),
                     ("hall",1892376)],
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

    #[test]
    fn ngrams_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the subword_indices() method hashes
        // to the same buckets in the future.

        for (word, ngrams_indices_check) in NGRAMS_INDICES_TESTS_36.iter() {
            let mut ngrams_indices_test = word.ngrams_indices(3, 6, 21);
            ngrams_indices_test.sort_by_key(|ngrams_indices_pairs| ngrams_indices_pairs.1);
            for (iter_check, iter_test) in ngrams_indices_check.into_iter().zip(ngrams_indices_test)
            {
                assert_eq!(iter_check.0, &iter_test.0);
            }
        }
    }
}
