use std::fs::File;
use std::io::BufReader;

use criterion::{criterion_group, criterion_main, Criterion};
use finalfusion::chunks::vocab::WordIndex;
use finalfusion::prelude::*;

mod data;
use data::load_corpus;

const EMBEDDINGS: &str = "benches/de-structgram-20190426.fifu";

fn read_embeddings() -> Embeddings<VocabWrap, StorageWrap> {
    let f = File::open(EMBEDDINGS).expect("Embedding file missing, run fetch-data.sh");
    Embeddings::read_embeddings(&mut BufReader::new(f)).unwrap()
}

fn mmap_embeddings() -> Embeddings<VocabWrap, StorageWrap> {
    let f = File::open(EMBEDDINGS).expect("Embedding file missing, run fetch-data.sh");
    Embeddings::mmap_embeddings(&mut BufReader::new(f)).unwrap()
}

fn allround_iter() -> impl Iterator<Item = String> + Clone {
    let corpus = load_corpus();
    corpus.into_iter()
}

fn known_iter<'a>(
    embeds: &'a Embeddings<VocabWrap, StorageWrap>,
) -> impl 'a + Iterator<Item = String> + Clone {
    allround_iter().filter_map(move |w| match embeds.vocab().idx(&w) {
        Some(WordIndex::Word(_)) => Some(w),
        _ => None,
    })
}

fn unknown_iter<'a>(
    embeds: &'a Embeddings<VocabWrap, StorageWrap>,
) -> impl 'a + Iterator<Item = String> + Clone {
    allround_iter().filter_map(move |w| match embeds.vocab().idx(&w) {
        Some(WordIndex::Subword(_)) => Some(w),
        _ => None,
    })
}

fn ndarray_benchmark(c: &mut Criterion) {
    let embeds = read_embeddings();
    let mut allround_iter = allround_iter().cycle();
    c.bench_function("ndarray-lookup-allround", |b| {
        b.iter(|| embeds.embedding(&allround_iter.next().unwrap()))
    });

    let mut known_iter = known_iter(&embeds).cycle();
    c.bench_function("ndarray-lookup-known", |b| {
        b.iter(|| embeds.embedding(&known_iter.next().unwrap()))
    });

    let mut unknown_iter = unknown_iter(&embeds).cycle();
    c.bench_function("ndarray-lookup-unknown", |b| {
        b.iter(|| embeds.embedding(&unknown_iter.next().unwrap()))
    });
}

fn ndarray_mmap_benchmark(c: &mut Criterion) {
    let embeds = mmap_embeddings();
    let mut allround_iter = allround_iter().cycle();
    c.bench_function("ndarray-mmap-lookup-allround", |b| {
        b.iter(|| embeds.embedding(&allround_iter.next().unwrap()))
    });

    let mut known_iter = known_iter(&embeds).cycle();
    c.bench_function("ndarray-mmap-lookup-known", |b| {
        b.iter(|| embeds.embedding(&known_iter.next().unwrap()))
    });

    let mut unknown_iter = unknown_iter(&embeds).cycle();
    c.bench_function("ndarray-mmap-lookup-unknown", |b| {
        b.iter(|| embeds.embedding(&unknown_iter.next().unwrap()))
    });
}

criterion_group!(ndarray_benches, ndarray_benchmark, ndarray_mmap_benchmark);
criterion_main!(ndarray_benches);
