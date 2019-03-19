use std::fs::File;
use std::io::BufReader;

use failure::{format_err, Error, ResultExt};

use finalfusion::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingFormat {
    FinalFusion,
    FinalFusionMmap,
    Word2Vec,
    Text,
    TextDims,
}

impl EmbeddingFormat {
    pub fn try_from(format: impl AsRef<str>) -> Result<Self, Error> {
        use EmbeddingFormat::*;

        match format.as_ref() {
            "finalfusion" => Ok(FinalFusion),
            "finalfusion_mmap" => Ok(FinalFusionMmap),
            "word2vec" => Ok(Word2Vec),
            "text" => Ok(Text),
            "textdims" => Ok(TextDims),
            unknown => Err(format_err!("Unknown embedding format: {}", unknown)),
        }
    }
}

pub fn read_embeddings_view(
    filename: &str,
    embedding_format: EmbeddingFormat,
) -> Result<Embeddings<VocabWrap, StorageViewWrap>, Error> {
    let f = File::open(filename).context("Cannot open embeddings file")?;
    let mut reader = BufReader::new(f);

    use EmbeddingFormat::*;
    let embeddings = match embedding_format {
        FinalFusion => ReadEmbeddings::read_embeddings(&mut reader),
        FinalFusionMmap => MmapEmbeddings::mmap_embeddings(&mut reader),
        Word2Vec => ReadWord2Vec::read_word2vec_binary(&mut reader, true).map(Embeddings::into),
        Text => ReadText::read_text(&mut reader, true).map(Embeddings::into),
        TextDims => ReadTextDims::read_text_dims(&mut reader, true).map(Embeddings::into),
    }
    .context("Cannot read embeddings")?;

    Ok(embeddings)
}
