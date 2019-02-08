use failure::{format_err, Error};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingFormat {
    Rust2Vec,
    Rust2VecMmap,
    Word2Vec,
    Text,
    TextDims,
}

impl EmbeddingFormat {
    pub fn try_from(format: impl AsRef<str>) -> Result<Self, Error> {
        use EmbeddingFormat::*;

        match format.as_ref() {
            "rust2vec" => Ok(Rust2Vec),
            "rust2vec_mmap" => Ok(Rust2VecMmap),
            "word2vec" => Ok(Word2Vec),
            "text" => Ok(Text),
            "textdims" => Ok(TextDims),
            unknown => Err(format_err!("Unknown embedding format: {}", unknown)),
        }
    }
}
