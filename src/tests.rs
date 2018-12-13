use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use ndarray::Array1;

use super::*;

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
fn test_read_word2vec_binary() {
    let f = File::open("testdata/similarity.bin").unwrap();
    let mut reader = BufReader::new(f);
    let mut embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
    embeddings.normalize();
    assert_eq!(41, embeddings.len());
    assert_eq!(100, embeddings.embed_len());
}

#[test]
fn test_word2vec_binary_roundtrip() {
    let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
    let mut check = Vec::new();
    reader.read_to_end(&mut check).unwrap();

    // Read embeddings.
    reader.seek(SeekFrom::Start(0)).unwrap();
    let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

    // Write embeddings to a byte vector.
    let mut output = Vec::new();
    embeddings.write_word2vec_binary(&mut output).unwrap();

    assert_eq!(check, output);
}

#[test]
fn test_builder() {
    let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
    let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

    let mut builder = Builder::new();
    for (word, embed) in embeddings.iter() {
        builder.push(word, embed.to_owned()).unwrap();
    }

    let embeddings_builder = builder.build().unwrap();

    assert_eq!(embeddings.data(), embeddings_builder.data());
    assert_eq!(embeddings.indices(), embeddings_builder.indices());
    assert_eq!(embeddings.words(), embeddings_builder.words());
    assert_eq!(embeddings.embed_len(), embeddings_builder.embed_len());
}

#[test]
fn test_builder_duplicate() {
    let mut builder = Builder::new();
    builder
        .push("hello", Array1::from_vec(vec![1.0, 2.0, 3.0]))
        .unwrap();
    assert!(builder
        .push("hello", Array1::from_vec(vec![4.0, 5.0, 6.0]))
        .is_err());
}

#[test]
fn test_builder_invalid_embedding_length() {
    let mut builder = Builder::new();
    builder
        .push("hello", Array1::from_vec(vec![1.0, 2.0, 3.0]))
        .unwrap();
    assert!(builder
        .push("hello", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]))
        .is_err());
}

#[test]
fn test_similarity() {
    let f = File::open("testdata/similarity.bin").unwrap();
    let mut reader = BufReader::new(f);
    let mut embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
    embeddings.normalize();

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
    let mut embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
    embeddings.normalize();

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
    let mut embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
    embeddings.normalize();

    let result = embeddings.analogy("Paris", "Frankreich", "Berlin", 40);
    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(40, result.len());

    for (idx, word_similarity) in result.iter().enumerate() {
        assert_eq!(ANALOGY_ORDER[idx], word_similarity.word)
    }
}
