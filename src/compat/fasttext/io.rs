use std::io::BufRead;

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{s, Array2, ErrorKind as ShapeErrorKind, ShapeError};
use serde::Serialize;
use toml::Value;

use crate::chunks::metadata::Metadata;
use crate::chunks::norms::NdNorms;
use crate::chunks::storage::{NdArray, Storage, StorageViewMut};
use crate::chunks::vocab::{FastTextSubwordVocab, SubwordIndices, Vocab};
use crate::embeddings::Embeddings;
use crate::io::{Error, ErrorKind, Result};
use crate::subword::BucketIndexer;
use crate::util::{l2_normalize_array, read_string};

use super::FastTextIndexer;

const FASTTEXT_FILEFORMAT_MAGIC: u32 = 793_712_314;
const FASTTEXT_VERSION: u32 = 12;

/// Read embeddings in the fastText format.
pub trait ReadFastText
where
    Self: Sized,
{
    /// Read embeddings in the fastText format.
    fn read_fasttext(reader: &mut impl BufRead) -> Result<Self>;
}

impl ReadFastText for Embeddings<FastTextSubwordVocab, NdArray> {
    fn read_fasttext(mut reader: &mut impl BufRead) -> Result<Self> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot fastText read magic", e))?;
        if magic != FASTTEXT_FILEFORMAT_MAGIC {
            return Err(ErrorKind::Format(format!(
                "Expected {} as magic, got: {}",
                FASTTEXT_FILEFORMAT_MAGIC, magic
            ))
            .into());
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read fastText version", e))?;
        if version > FASTTEXT_VERSION {
            return Err(ErrorKind::Format(format!(
                "Expected {} as version, got: {}",
                FASTTEXT_VERSION, version
            ))
            .into());
        }

        let config = Config::read(&mut reader)?;

        let vocab = read_vocab(&config, &mut reader)?;

        let is_quantized = reader
            .read_u8()
            .map_err(|e| ErrorKind::io_error("Cannot read quantization information", e))?;
        if is_quantized == 1 {
            return Err(
                ErrorKind::Format("Quantized fastText models are not supported".into()).into(),
            );
        }

        // Read and prepare storage.
        let mut storage = read_embeddings(&mut reader)?;
        add_subword_embeddings(&vocab, &mut storage);
        #[allow(clippy::deref_addrof)]
        let norms = NdNorms(l2_normalize_array(
            storage.view_mut().slice_mut(s![0..vocab.len(), ..]),
        ));

        // Verify that vocab and storage shapes match.
        if storage.shape().0 != vocab.len() + config.bucket as usize {
            return Err(Error::Shape(ShapeError::from_kind(
                ShapeErrorKind::IncompatibleShape,
            )));
        }

        let metadata = Value::try_from(config).map_err(|e| {
            ErrorKind::Format(format!("Cannot serialize model metadata to TOML: {}", e))
        })?;

        Ok(Embeddings::new(
            Some(Metadata(metadata)),
            vocab,
            storage,
            norms,
        ))
    }
}

/// fastText model configuration.
#[derive(Copy, Clone, Debug, Serialize)]
struct Config {
    dims: u32,
    window_size: u32,
    epoch: u32,
    min_count: u32,
    neg: u32,
    word_ngrams: u32,
    loss: Loss,
    model: Model,
    bucket: u32,
    min_n: u32,
    max_n: u32,
    lr_update_rate: u32,
    sampling_threshold: f64,
}

impl Config {
    /// Read fastText model configuration.
    fn read<R>(reader: &mut R) -> Result<Config>
    where
        R: BufRead,
    {
        let dims = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of dimensions", e))?;
        let window_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read window size", e))?;
        let epoch = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of epochs", e))?;
        let min_count = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read minimum count", e))?;
        let neg = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read negative samples", e))?;
        let word_ngrams = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read word n-gram length", e))?;
        let loss = Loss::read(reader)?;
        let model = Model::read(reader)?;
        let bucket = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of buckets", e))?;
        let min_n = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read minimum subword length", e))?;
        let max_n = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read maximum subword length", e))?;
        let lr_update_rate = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read LR update rate", e))?;
        let sampling_threshold = reader
            .read_f64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read sampling threshold", e))?;

        Ok(Config {
            dims,
            window_size,
            epoch,
            min_count,
            neg,
            word_ngrams,
            loss,
            model,
            bucket,
            min_n,
            max_n,
            lr_update_rate,
            sampling_threshold,
        })
    }
}

/// fastText loss type.
#[derive(Copy, Clone, Debug, Serialize)]
enum Loss {
    HierarchicalSoftmax,
    NegativeSampling,
    Softmax,
}

impl Loss {
    fn read<R>(reader: &mut R) -> Result<Loss>
    where
        R: BufRead,
    {
        let loss = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read loss type", e))?;

        use self::Loss::*;
        match loss {
            1 => Ok(HierarchicalSoftmax),
            2 => Ok(NegativeSampling),
            3 => Ok(Softmax),
            l => Err(ErrorKind::Format(format!("Unknown loss: {}", l)).into()),
        }
    }
}

/// fastText model type.
#[derive(Copy, Clone, Debug, Serialize)]
enum Model {
    CBOW,
    SkipGram,
    Supervised,
}

impl Model {
    fn read<R>(reader: &mut R) -> Result<Model>
    where
        R: BufRead,
    {
        let model = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read model type", e))?;

        use self::Model::*;
        match model {
            1 => Ok(CBOW),
            2 => Ok(SkipGram),
            3 => Ok(Supervised),
            m => Err(ErrorKind::Format(format!("Unknown model: {}", m)).into()),
        }
    }
}

/// Add subword embeddings to word embeddings.
///
/// fastText stores word embeddings without subword embeddings. This method
/// adds the subword embeddings.
fn add_subword_embeddings(vocab: &FastTextSubwordVocab, embeds: &mut NdArray) {
    for (idx, word) in vocab.words().iter().enumerate() {
        if let Some(indices) = vocab.subword_indices(word) {
            let n_embeds = indices.len() + 1;

            // Sum the embedding and its subword embeddings.
            let mut embed = embeds.embedding(idx).into_owned();
            for subword_idx in indices {
                embed += &embeds.embedding(subword_idx).as_view();
            }

            // Compute the average embedding.
            embed /= n_embeds as f32;

            embeds.view_mut().row_mut(idx).assign(&embed);
        }
    }
}

/// Read the embedding matrix.
fn read_embeddings<R>(reader: &mut R) -> Result<NdArray>
where
    R: BufRead,
{
    let m = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read number of embedding matrix rows", e))?;
    let n = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read number of embedding matrix columns", e))?;

    let mut data = vec![0.0; (m * n) as usize];
    reader
        .read_f32_into::<LittleEndian>(&mut data)
        .map_err(|e| ErrorKind::io_error("Cannot read embeddings", e))?;

    let data = Array2::from_shape_vec((m as usize, n as usize), data).map_err(Error::Shape)?;

    Ok(NdArray(data))
}

/// Read the vocabulary.
fn read_vocab<R>(config: &Config, reader: &mut R) -> Result<FastTextSubwordVocab>
where
    R: BufRead,
{
    let size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read vocabulary size", e))?;
    reader
        .read_u32::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read number of words", e))?;

    let n_labels = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot number of labels", e))?;
    if n_labels > 0 {
        return Err(
            ErrorKind::Format("fastText prediction models are not supported".into()).into(),
        );
    }

    reader
        .read_u64::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read number of tokens", e))?;

    let prune_idx_size = reader
        .read_i64::<LittleEndian>()
        .map_err(|e| ErrorKind::io_error("Cannot read pruned vocabulary size", e))?;
    if prune_idx_size > 0 {
        return Err(ErrorKind::Format("Pruned vocabularies are not supported".into()).into());
    }

    let mut words = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let word = read_string(reader, 0, false)?;
        reader
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read word frequency", e))?;
        let entry_type = reader
            .read_u8()
            .map_err(|e| ErrorKind::io_error("Cannot read entry type", e))?;
        if entry_type != 0 {
            return Err(ErrorKind::Format("Non-word entry".into()).into());
        }

        words.push(word)
    }

    Ok(FastTextSubwordVocab::new(
        words,
        config.min_n,
        config.max_n,
        FastTextIndexer::new(config.bucket as usize),
    ))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use approx::assert_abs_diff_eq;

    use super::ReadFastText;
    use crate::embeddings::Embeddings;
    use crate::similarity::WordSimilarity;

    #[test]
    fn test_read_fasttext() {
        let f = File::open("testdata/fasttext.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_fasttext(&mut reader).unwrap();
        let results = embeddings.word_similarity("über", 3).unwrap();
        assert_eq!(results[0].word, "auf");
        assert_abs_diff_eq!(*results[0].similarity, 0.568513, epsilon = 1e-6);
        assert_eq!(results[1].word, "vor");
        assert_abs_diff_eq!(*results[1].similarity, 0.551551, epsilon = 1e-6);
        assert_eq!(results[2].word, "durch");
        assert_abs_diff_eq!(*results[2].similarity, 0.547349, epsilon = 1e-6);
    }

    #[test]
    fn test_read_fasttext_unknown() {
        let f = File::open("testdata/fasttext.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_fasttext(&mut reader).unwrap();
        let results = embeddings.word_similarity("unknown", 3).unwrap();
        assert_eq!(results[0].word, "einer");
        assert_abs_diff_eq!(*results[0].similarity, 0.691177, epsilon = 1e-6);
        assert_eq!(results[1].word, "und");
        assert_abs_diff_eq!(*results[1].similarity, 0.576449, epsilon = 1e-6);
        assert_eq!(results[2].word, "des");
        assert_abs_diff_eq!(*results[2].similarity, 0.570398, epsilon = 1e-6);
    }
}
