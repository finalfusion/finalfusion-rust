use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

use finalfusion::subword::{BucketIndexer, FinalfusionHashIndexer, Indexer, SubwordIndices};

const MIN_N: usize = 3;
const MAX_N: usize = 6;
const WORD_LENGTH: usize = 10;

fn subwords(string: &str, min_n: usize, max_n: usize, indexer: &impl Indexer) -> u64 {
    // Sum the subword indices, to ensure that the benchmark
    // evaluates them.
    string
        .subword_indices(min_n, max_n, indexer)
        .into_iter()
        .fold(0, |sum, v| sum.wrapping_add(v))
}

fn ngrams_benchmark(c: &mut Criterion) {
    let rng = thread_rng();
    let string =
        String::from_utf8(rng.sample_iter(&Alphanumeric).take(WORD_LENGTH).collect()).unwrap();

    let indexer = FinalfusionHashIndexer::new(21);

    c.bench_function("subwords-len-10-minn-3-maxn-6", move |b| {
        b.iter(|| subwords(&string, black_box(MIN_N), black_box(MAX_N), &indexer))
    });
}

criterion_group!(benches, ngrams_benchmark);
criterion_main!(benches);
