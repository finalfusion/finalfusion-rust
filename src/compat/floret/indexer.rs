use std::io::Cursor;

use murmur3::murmur3_x64_128;
use smallvec::smallvec;

use crate::subword::{Indexer, NGramVec, StrWithCharLen};

/// floret subword indexer.
///
/// By default, floret does not use a separate word embedding matrix. Every
/// n-gram and the full word is mapped to 1 to 4 hash functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloretIndexer {
    n_buckets: u64,
    seed: u32,
    n_hashes: u32,
}

impl FloretIndexer {
    pub fn new(n_buckets: u64, n_hashes: u32, seed: u32) -> Self {
        assert!(
            n_hashes > 0 && n_hashes <= 4,
            "Floret indexer needs 1 to 4 hashes, got {}",
            n_hashes
        );

        assert_ne!(n_buckets, 0, "Floret needs at least 1 bucket.");

        Self {
            n_buckets,
            n_hashes,
            seed,
        }
    }
}

impl Indexer for FloretIndexer {
    fn index_ngram(&self, ngram: &StrWithCharLen) -> NGramVec {
        let hash = murmur3_x64_128(&mut Cursor::new(ngram.as_bytes()), self.seed)
            .expect("Murmur hash failed");

        let mut hash_array = [0; 4];
        hash_array[0] = hash as u32;
        hash_array[1] = (hash >> 32) as u32;
        hash_array[2] = (hash >> 64) as u32;
        hash_array[3] = (hash >> 96) as u32;

        let mut indices = smallvec![0; self.n_hashes as usize];
        for i in 0..self.n_hashes as usize {
            indices[i] = hash_array[i] as u64 % self.n_buckets;
        }

        indices
    }

    fn upper_bound(&self) -> u64 {
        self.n_buckets
    }

    fn infallible() -> bool {
        true
    }
}
