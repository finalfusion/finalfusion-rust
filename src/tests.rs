use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

use crate::chunks::vocab::Vocab;
use crate::compat::word2vec::{ReadWord2Vec, ReadWord2VecRaw, WriteWord2Vec};
use crate::embeddings::Embeddings;

#[test]
fn test_read_word2vec_binary() {
    let f = File::open("testdata/similarity.bin").unwrap();
    let mut reader = BufReader::new(f);
    let embeddings = Embeddings::read_word2vec_binary_raw(&mut reader).unwrap();
    assert_eq!(41, embeddings.vocab().len());
    assert_eq!(100, embeddings.dims());
}

#[test]
fn test_word2vec_binary_roundtrip() {
    let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
    let mut check = Vec::new();
    reader.read_to_end(&mut check).unwrap();

    // Read embeddings.
    reader.seek(SeekFrom::Start(0)).unwrap();
    let embeddings = Embeddings::read_word2vec_binary_raw(&mut reader).unwrap();

    // Write embeddings to a byte vector.
    let mut output = Vec::new();
    embeddings
        .write_word2vec_binary(&mut output, false)
        .unwrap();

    assert_eq!(check, output);
}

#[test]
fn test_word2vec_binary_write_unnormalized() {
    let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());

    // Read unnormalized embeddings
    let embeddings_check = Embeddings::read_word2vec_binary_raw(&mut reader).unwrap();

    // Read normalized embeddings.
    reader.seek(SeekFrom::Start(0)).unwrap();
    let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

    // Write embeddings to a byte vector.
    let mut output = Vec::new();
    embeddings.write_word2vec_binary(&mut output, true).unwrap();

    let embeddings = Embeddings::read_word2vec_binary_raw(&mut Cursor::new(&output)).unwrap();

    assert!(embeddings
        .storage()
        .0
        .all_close(&embeddings_check.storage().0, 1e-6));
}
