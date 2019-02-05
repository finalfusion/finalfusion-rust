use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::vocab::Vocab;
use crate::word2vec::{ReadWord2Vec, WriteWord2Vec};
use crate::Embeddings;

#[test]
fn test_read_word2vec_binary() {
    let f = File::open("testdata/similarity.bin").unwrap();
    let mut reader = BufReader::new(f);
    let mut embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();
    embeddings.normalize();
    assert_eq!(41, embeddings.vocab().len());
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
