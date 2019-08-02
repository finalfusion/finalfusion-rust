use std::i32;

use crate::subword::{BucketIndexer, Indexer, StrWithCharLen};

/// fastText-compatible subword indexer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastTextIndexer {
    // fastText is inconsistent with types when it comes to buckets,
    // the data types are:
    //
    // - buckets: int
    // - hash: uint32_t
    // - bucket: int32_t
    //
    // We will make the following assumptions: (1) the range of
    // buckets is determined by int32_t; (2) the maximum number of
    // buckets is the maximum value of int32_t. We will verify
    // the maximum value in the constructor of FastTextIndexer.
    buckets: u32,
}

impl BucketIndexer for FastTextIndexer {
    /// Construct a FastTextIndexer instance
    ///
    /// `buckets` is the (exact) number of buckets to use.
    fn new(buckets: usize) -> Self {
        assert!(
            buckets <= i32::MAX as usize,
            "The largest possible number of buckets is: {}",
            i32::MAX
        );

        FastTextIndexer {
            buckets: buckets as u32,
        }
    }

    fn buckets(&self) -> usize {
        self.buckets as usize
    }
}

impl Indexer for FastTextIndexer {
    fn index_ngram(&self, ngram: &StrWithCharLen) -> u64 {
        u64::from(fasttext_hash(ngram.as_str()) % self.buckets)
    }
}

/// fastText FNV-1a implementation.
///
/// The fastText implementation of FNV-1a has a bug caused
/// by sign extension on compilers wher char is signed:
///
/// https://github.com/facebookresearch/fastText/issues/539
///
/// This implementation 'emulates' the bug for compatibility
/// with pretrained fastText embeddings.
fn fasttext_hash(ngram: &str) -> u32 {
    let mut h = 2_166_136_261;

    for byte in ngram.bytes() {
        // Cast bytes to i8, so that sign-extension is applied when
        // widening to u32.
        h ^= (byte as i8) as u32;
        h = h.wrapping_mul(16_777_619);
    }

    h
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::iter::FromIterator;

    use lazy_static::lazy_static;

    use super::FastTextIndexer;
    use crate::subword::{BucketIndexer, SubwordIndices};

    lazy_static! {
        // Subword indices were verified against fastText output.
        static ref SUBWORD_TESTS: HashMap<&'static str, Vec<u64>> = HashMap::from_iter(vec![
            (
                "<Daniël>",
                vec![
                    69886, 84537, 338340, 441697, 448390, 468430, 504093, 573175, 749365, 804851,
                    811506, 991985, 1022467, 1105725, 1249224, 1418443, 1493412, 1880616
                ]
            ),
            (
                "<überspringen>",
                vec![
                    79599, 119685, 255527, 263610, 352266, 385524, 403356, 421853, 485366, 488156,
                    586161, 619228, 629649, 642367, 716781, 751724, 754367, 771707, 799583, 887882,
                    894109, 904527, 908492, 978563, 991164, 992241, 1142035, 1230973, 1278156,
                    1350653, 1414694, 1513262, 1533308, 1607098, 1607788, 1664269, 1712300,
                    1749574, 1793082, 1891605, 1934955, 1992797
                ]
            ),
        ]);

        // Subword indices were verified against fastText output.
        static ref SUBWORD_TESTS_5_5: HashMap<&'static str, Vec<u64>> = HashMap::from_iter(vec![
            ("<Daniël>", vec![441697, 749365, 1105725, 1880616]),
            (
                "<überspringen>",
                vec![
                    79599, 352266, 385524, 629649, 716781, 978563, 991164, 1230973, 1350653,
                    1992797
                ]
            )
        ]);
    }

    #[test]
    fn subword_indices_test() {
        let indexer = FastTextIndexer::new(2_000_000);
        for (word, indices_check) in SUBWORD_TESTS.iter() {
            let mut indices = word.subword_indices(3, 6, &indexer);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn subword_indices_test_5_5() {
        let indexer = FastTextIndexer::new(2_000_000);
        for (word, indices_check) in SUBWORD_TESTS_5_5.iter() {
            let mut indices = word.subword_indices(5, 5, &indexer);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }
}
