use std::convert::TryInto;
use std::io::{BufRead, Write};
use std::ops::Mul;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{s, Array2, ErrorKind as ShapeErrorKind, ShapeError};
use serde::Serialize;
use toml::Table;

use crate::chunks::metadata::Metadata;
use crate::chunks::norms::NdNorms;
use crate::chunks::storage::{NdArray, Storage, StorageViewMut};
use crate::chunks::vocab::{FastTextSubwordVocab, SubwordIndices, Vocab};
use crate::embeddings::Embeddings;
use crate::error::{Error, Result};
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

    /// Read embeddings in the fastText format lossily.
    ///
    /// In constrast to `read_fasttext`, this method does not fail
    /// on reading tokens with invalid UTF-8 byte sequences. Invalid
    /// UTF-8 sequences will be replaced by the unicode replacement
    /// character.
    fn read_fasttext_lossy(reader: &mut impl BufRead) -> Result<Self>;
}

impl ReadFastText for Embeddings<FastTextSubwordVocab, NdArray> {
    fn read_fasttext(reader: &mut impl BufRead) -> Result<Self> {
        Self::read_fasttext_private(reader, false)
    }

    fn read_fasttext_lossy(reader: &mut impl BufRead) -> Result<Self> {
        Self::read_fasttext_private(reader, true)
    }
}

/// Read embeddings in the fastText format.
pub trait ReadFastTextPrivate
where
    Self: Sized,
{
    /// Read embeddings in the fastText format.
    fn read_fasttext_private(reader: &mut impl BufRead, lossy: bool) -> Result<Self>;
}

impl ReadFastTextPrivate for Embeddings<FastTextSubwordVocab, NdArray> {
    fn read_fasttext_private(mut reader: &mut impl BufRead, lossy: bool) -> Result<Self> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot fastText read magic", e))?;
        if magic != FASTTEXT_FILEFORMAT_MAGIC {
            return Err(Error::Format(format!(
                "Expected {} as magic, got: {}",
                FASTTEXT_FILEFORMAT_MAGIC, magic
            )));
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read fastText version", e))?;
        if version > FASTTEXT_VERSION {
            return Err(Error::Format(format!(
                "Expected {} as version, got: {}",
                FASTTEXT_VERSION, version
            )));
        }

        let config = Config::read(&mut reader)?;

        let vocab = read_vocab(&config, &mut reader, lossy)?;

        let is_quantized = reader
            .read_u8()
            .map_err(|e| Error::read_error("Cannot read quantization information", e))?;
        if is_quantized == 1 {
            return Err(Error::Format(
                "Quantized fastText models are not supported".into(),
            ));
        }

        // Read and prepare storage.
        let mut storage = read_embeddings(&mut reader)?;
        add_subword_embeddings(&vocab, &mut storage);
        #[allow(clippy::deref_addrof)]
        let norms = NdNorms::new(l2_normalize_array(
            storage.view_mut().slice_mut(s![0..vocab.words_len(), ..]),
        ));

        // Verify that vocab and storage shapes match.
        if storage.shape().0 != vocab.words_len() + config.bucket as usize {
            return Err(Error::MatrixShape(ShapeError::from_kind(
                ShapeErrorKind::IncompatibleShape,
            )));
        }

        let metadata = Table::try_from(config).map_err(|e| {
            Error::Format(format!("Cannot serialize model metadata to TOML: {}", e))
        })?;

        Ok(Embeddings::new(
            Some(Metadata::new(metadata)),
            vocab,
            storage,
            norms,
        ))
    }
}

/// Read embeddings in the fastText format.
pub trait WriteFastText<W>
where
    W: Write,
{
    /// Write the embeddings to the given writer in fastText format.
    ///
    /// fastText embeddings contain metadata. All metadata that is not
    /// relevant to load and use the embeddings is set to zero when
    /// writing fastText embeddings.
    fn write_fasttext(&self, write: &mut W) -> Result<()>;
}

impl<W, S> WriteFastText<W> for Embeddings<FastTextSubwordVocab, S>
where
    W: Write,
    S: Storage,
{
    fn write_fasttext(&self, write: &mut W) -> Result<()> {
        let vocab = self.vocab();
        let mut config = Config::new();

        // Merge known parts of the configuration.
        config.bucket = vocab
            .indexer()
            .buckets()
            .try_into()
            .map_err(|_| Error::Overflow)?;
        config.dims = self
            .storage()
            .shape()
            .1
            .try_into()
            .map_err(|_| Error::Overflow)?;
        config.max_n = vocab.max_n();
        config.min_n = vocab.min_n();

        write
            .write_u32::<LittleEndian>(FASTTEXT_FILEFORMAT_MAGIC)
            .map_err(|e| Error::write_error("Cannot write fastText magic", e))?;
        write
            .write_u32::<LittleEndian>(FASTTEXT_VERSION)
            .map_err(|e| Error::write_error("Cannot write fastText version", e))?;

        config.write(write)?;

        write_vocab(write, self.vocab())?;

        write
            .write_u8(0)
            .map_err(|e| Error::write_error("Cannot write quantization status", e))?;

        write_embeddings(write, self)
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
    // Don't expose as `Default`, since the default values are quite
    // non-sensical.
    fn new() -> Self {
        Config {
            bucket: Default::default(),
            dims: Default::default(),
            epoch: Default::default(),
            loss: Loss::HierarchicalSoftmax,
            lr_update_rate: Default::default(),
            max_n: Default::default(),
            min_count: Default::default(),
            min_n: Default::default(),
            model: Model::Cbow,
            neg: Default::default(),
            sampling_threshold: Default::default(),
            window_size: Default::default(),
            word_ngrams: 1,
        }
    }

    /// Read fastText model configuration.
    fn read<R>(reader: &mut R) -> Result<Config>
    where
        R: BufRead,
    {
        let dims = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of dimensions", e))?;
        let window_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read window size", e))?;
        let epoch = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of epochs", e))?;
        let min_count = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read minimum count", e))?;
        let neg = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read negative samples", e))?;
        let word_ngrams = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read word n-gram length", e))?;
        let loss = Loss::read(reader)?;
        let model = Model::read(reader)?;
        let bucket = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read number of buckets", e))?;
        let min_n = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read minimum subword length", e))?;
        let max_n = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read maximum subword length", e))?;
        let lr_update_rate = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read LR update rate", e))?;
        let sampling_threshold = reader
            .read_f64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read sampling threshold", e))?;

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

    fn write<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write,
    {
        write
            .write_u32::<LittleEndian>(self.dims)
            .map_err(|e| Error::write_error("Cannot write number of dimensions", e))?;
        write
            .write_u32::<LittleEndian>(self.window_size)
            .map_err(|e| Error::write_error("Cannot write window size", e))?;
        write
            .write_u32::<LittleEndian>(self.epoch)
            .map_err(|e| Error::write_error("Cannot write number of epochs", e))?;
        write
            .write_u32::<LittleEndian>(self.min_count)
            .map_err(|e| Error::write_error("Cannot write minimum count", e))?;
        write
            .write_u32::<LittleEndian>(self.neg)
            .map_err(|e| Error::write_error("Cannot write negative samples", e))?;
        write
            .write_u32::<LittleEndian>(self.word_ngrams)
            .map_err(|e| Error::write_error("Cannot write word n-gram length", e))?;
        self.loss.write(write)?;
        self.model.write(write)?;
        write
            .write_u32::<LittleEndian>(self.bucket)
            .map_err(|e| Error::write_error("Cannot write number of buckets", e))?;
        write
            .write_u32::<LittleEndian>(self.min_n)
            .map_err(|e| Error::write_error("Cannot write minimum subword length", e))?;
        write
            .write_u32::<LittleEndian>(self.max_n)
            .map_err(|e| Error::write_error("Cannot write maximum subword length", e))?;
        write
            .write_u32::<LittleEndian>(self.lr_update_rate)
            .map_err(|e| Error::write_error("Cannot write LR update rate", e))?;
        write
            .write_f64::<LittleEndian>(self.sampling_threshold)
            .map_err(|e| Error::write_error("Cannot write sampling threshold", e))
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
            .map_err(|e| Error::read_error("Cannot read loss type", e))?;

        use self::Loss::*;
        match loss {
            1 => Ok(HierarchicalSoftmax),
            2 => Ok(NegativeSampling),
            3 => Ok(Softmax),
            l => Err(Error::Format(format!("Unknown loss: {}", l))),
        }
    }

    fn write<W>(self, write: &mut W) -> Result<()>
    where
        W: Write,
    {
        use self::Loss::*;
        let loss_id = match self {
            HierarchicalSoftmax => 1u32,
            NegativeSampling => 2,
            Softmax => 3,
        };

        write
            .write_u32::<LittleEndian>(loss_id)
            .map_err(|e| Error::write_error("Cannot write loss function", e))
    }
}

/// fastText model type.
#[derive(Copy, Clone, Debug, Serialize)]
enum Model {
    Cbow,
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
            .map_err(|e| Error::read_error("Cannot read model type", e))?;

        use self::Model::*;
        match model {
            1 => Ok(Cbow),
            2 => Ok(SkipGram),
            3 => Ok(Supervised),
            m => Err(Error::Format(format!("Unknown model: {}", m))),
        }
    }

    fn write<W>(self, write: &mut W) -> Result<()>
    where
        W: Write,
    {
        use self::Model::*;
        let model_id = match self {
            Cbow => 1u32,
            SkipGram => 2,
            Supervised => 3,
        };

        write
            .write_u32::<LittleEndian>(model_id)
            .map_err(|e| Error::write_error("Cannot write loss function", e))
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
                embed += &embeds.embedding(subword_idx).view();
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
        .map_err(|e| Error::read_error("Cannot read number of embedding matrix rows", e))?
        .try_into()
        .map_err(|_| Error::Overflow)?;
    let n = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read number of embedding matrix columns", e))?
        .try_into()
        .map_err(|_| Error::Overflow)?;

    // XXX: check overflow.
    let mut data = Array2::zeros((m, n));
    reader
        .read_f32_into::<LittleEndian>(data.as_slice_mut().unwrap())
        .map_err(|e| Error::read_error("Cannot read embeddings", e))?;

    Ok(NdArray::new(data))
}

/// Write embeddings.
fn write_embeddings<W, S>(
    write: &mut W,
    embeddings: &Embeddings<FastTextSubwordVocab, S>,
) -> Result<()>
where
    S: Storage,
    W: Write,
{
    let storage = embeddings.storage();
    let vocab = embeddings.vocab();

    write
        .write_u64::<LittleEndian>(storage.shape().0 as u64)
        .map_err(|e| Error::write_error("Cannot write number of embedding matrix rows", e))?;
    write
        .write_u64::<LittleEndian>(storage.shape().1 as u64)
        .map_err(|e| Error::write_error("Cannot write number of embedding matrix columns", e))?;

    for (word, embedding_with_norm) in embeddings.iter_with_norms() {
        let mut unnormalized_embedding =
            embedding_with_norm.embedding.mul(embedding_with_norm.norm);

        if let Some(subword_indices) = vocab.subword_indices(word) {
            unnormalized_embedding *= (subword_indices.len() + 1) as f32;

            for subword_index in subword_indices {
                unnormalized_embedding -= &storage.embedding(subword_index);
            }
        }

        for v in &unnormalized_embedding {
            write
                .write_f32::<LittleEndian>(*v)
                .map_err(|e| Error::write_error("Cannot write embedding", e))?;
        }
    }

    // Write subword embeddings.
    for idx in vocab.words_len()..vocab.vocab_len() {
        for v in storage.embedding(idx).view() {
            write
                .write_f32::<LittleEndian>(*v)
                .map_err(|e| Error::write_error("Cannot write subword embedding", e))?;
        }
    }

    Ok(())
}

/// Read the vocabulary.
fn read_vocab<R>(config: &Config, reader: &mut R, lossy: bool) -> Result<FastTextSubwordVocab>
where
    R: BufRead,
{
    let size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read vocabulary size", e))?;
    reader
        .read_u32::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read number of words", e))?;

    let n_labels = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read number of labels", e))?;
    if n_labels > 0 {
        return Err(Error::Format(
            "fastText prediction models are not supported".into(),
        ));
    }

    reader
        .read_u64::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read number of tokens", e))?;

    let prune_idx_size = reader
        .read_i64::<LittleEndian>()
        .map_err(|e| Error::read_error("Cannot read pruned vocabulary size", e))?;
    if prune_idx_size >= 0 {
        return Err(Error::Format(
            "Pruned vocabularies are not supported".into(),
        ));
    }

    let mut words = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let word = read_string(reader, 0, lossy)?;
        reader
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read word frequency", e))?;
        let entry_type = reader
            .read_u8()
            .map_err(|e| Error::read_error("Cannot read entry type", e))?;
        if entry_type != 0 {
            return Err(Error::Format("Non-word entry".into()));
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

/// Write the vocabulary.
fn write_vocab<W>(write: &mut W, vocab: &FastTextSubwordVocab) -> Result<()>
where
    W: Write,
{
    let words_len = vocab.words_len().try_into().map_err(|_| Error::Overflow)?;
    write
        .write_u32::<LittleEndian>(words_len)
        .map_err(|e| Error::write_error("Cannot write vocabulary size", e))?;
    write
        .write_u32::<LittleEndian>(words_len)
        .map_err(|e| Error::write_error("Cannot write number of words", e))?;
    write
        .write_u32::<LittleEndian>(0)
        .map_err(|e| Error::write_error("Cannot write number of labels", e))?;
    write
        .write_u64::<LittleEndian>(0)
        .map_err(|e| Error::write_error("Cannot write number of tokens", e))?;
    write
        .write_i64::<LittleEndian>(-1)
        .map_err(|e| Error::write_error("Cannot write pruned vocabulary size", e))?;

    for word in vocab.words() {
        write
            .write_all(word.as_bytes())
            .map_err(|e| Error::write_error("Cannot write word", e))?;
        write
            .write_u8(0)
            .map_err(|e| Error::write_error("Cannot write word terminator", e))?;
        write
            .write_u64::<LittleEndian>(0)
            .map_err(|e| Error::write_error("Cannot write word frequency", e))?;
        write
            .write_u8(0)
            .map_err(|e| Error::read_error("Cannot read entry type", e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor};

    use approx::assert_abs_diff_eq;

    use super::ReadFastText;
    use crate::compat::fasttext::WriteFastText;
    use crate::embeddings::Embeddings;
    use crate::similarity::WordSimilarity;
    use crate::storage::{NdArray, StorageView};
    use crate::vocab::FastTextSubwordVocab;

    fn read_fasttext() -> Embeddings<FastTextSubwordVocab, NdArray> {
        let f = File::open("testdata/fasttext.bin").unwrap();
        let mut reader = BufReader::new(f);
        Embeddings::read_fasttext(&mut reader).unwrap()
    }

    #[test]
    fn test_read_fasttext() {
        let embeddings = read_fasttext();
        let results = embeddings.word_similarity("über", 3, None).unwrap();
        assert_eq!(results[0].word(), "auf");
        assert_abs_diff_eq!(results[0].cosine_similarity(), 0.568513, epsilon = 1e-6);
        assert_eq!(results[1].word(), "vor");
        assert_abs_diff_eq!(results[1].cosine_similarity(), 0.551551, epsilon = 1e-6);
        assert_eq!(results[2].word(), "durch");
        assert_abs_diff_eq!(results[2].cosine_similarity(), 0.547349, epsilon = 1e-6);
    }

    #[test]
    fn test_read_fasttext_unknown() {
        let embeddings = read_fasttext();
        let results = embeddings.word_similarity("unknown", 3, None).unwrap();
        assert_eq!(results[0].word(), "einer");
        assert_abs_diff_eq!(results[0].cosine_similarity(), 0.691177, epsilon = 1e-6);
        assert_eq!(results[1].word(), "und");
        assert_abs_diff_eq!(results[1].cosine_similarity(), 0.576449, epsilon = 1e-6);
        assert_eq!(results[2].word(), "des");
        assert_abs_diff_eq!(results[2].cosine_similarity(), 0.570398, epsilon = 1e-6);
    }

    #[test]
    fn write_fasttext() {
        let check = read_fasttext();
        let mut write = Vec::new();
        check.write_fasttext(&mut write).unwrap();
        let mut read = Cursor::new(write);
        let embeddings = Embeddings::read_fasttext(&mut read).unwrap();

        assert_eq!(check.vocab(), embeddings.vocab());
        assert_abs_diff_eq!(check.storage().view(), embeddings.storage().view());
    }
}
