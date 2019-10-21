//! Prelude exports the most commonly-used types and traits.

pub use crate::chunks::storage::{StorageViewWrap, StorageWrap};

pub use crate::chunks::vocab::VocabWrap;

pub use crate::compat::fasttext::ReadFastText;

pub use crate::compat::text::{ReadText, ReadTextDims};

pub use crate::compat::word2vec::ReadWord2Vec;

pub use crate::embeddings::Embeddings;

pub use crate::io::{MmapEmbeddings, ReadEmbeddings};

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use crate::prelude::*;

    #[test]
    fn prelude_allows_embedding_lookups() {
        let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
        let embeds: Embeddings<VocabWrap, StorageWrap> =
            Embeddings::read_embeddings(&mut reader).unwrap();

        assert!(embeds.embedding("Berlin").is_some());
    }

    #[test]
    fn prelude_allows_embedding_view_lookups() {
        let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
        let embeds_view: Embeddings<VocabWrap, StorageViewWrap> =
            Embeddings::read_embeddings(&mut reader).unwrap();
        assert!(embeds_view.embedding("Berlin").is_some());
    }

    #[test]
    fn prelude_allows_embedding_mmap_lookups() {
        let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
        let embeds_view: Embeddings<VocabWrap, StorageWrap> =
            Embeddings::mmap_embeddings(&mut reader).unwrap();
        assert!(embeds_view.embedding("Berlin").is_some());
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn prelude_allows_embedding_mmap_view_lookups() {
        let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
        let embeds_view: Embeddings<VocabWrap, StorageViewWrap> =
            Embeddings::mmap_embeddings(&mut reader).unwrap();
        assert!(embeds_view.embedding("Berlin").is_some());
    }

    #[test]
    fn prelude_allows_reading_fasttext() {
        let mut reader = BufReader::new(File::open("testdata/fasttext.bin").unwrap());
        Embeddings::read_fasttext(&mut reader).unwrap();
    }

    #[test]
    fn prelude_allows_reading_text() {
        let mut reader = BufReader::new(File::open("testdata/similarity.nodims").unwrap());
        Embeddings::read_text(&mut reader).unwrap();
    }

    #[test]
    fn prelude_allows_reading_text_dims() {
        let mut reader = BufReader::new(File::open("testdata/similarity.txt").unwrap());
        Embeddings::read_text_dims(&mut reader).unwrap();
    }

    #[test]
    fn prelude_allows_reading_word2vec() {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary(&mut reader).unwrap();
    }
}
