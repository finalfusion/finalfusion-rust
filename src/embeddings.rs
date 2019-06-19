//! Word embeddings.

use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::iter::Enumerate;
use std::mem;
use std::slice;

use failure::{ensure, Error};
use ndarray::Array1;
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use reductive::pq::TrainPQ;

use crate::io::{
    private::{ChunkIdentifier, Header, MmapChunk, ReadChunk, WriteChunk},
    MmapEmbeddings, ReadEmbeddings, WriteEmbeddings,
};
use crate::metadata::Metadata;
use crate::norms::{NdNorms, Norms};
use crate::storage::{
    CowArray, CowArray1, MmapArray, NdArray, Quantize as QuantizeStorage, QuantizedArray, Storage,
    StorageView, StorageViewWrap, StorageWrap,
};
use crate::util::l2_normalize;
use crate::vocab::{SimpleVocab, SubwordVocab, Vocab, VocabWrap, WordIndex};

/// Word embeddings.
///
/// This data structure stores word embeddings (also known as *word vectors*)
/// and provides some useful methods on the embeddings, such as similarity
/// and analogy queries.
#[derive(Debug)]
pub struct Embeddings<V, S> {
    metadata: Option<Metadata>,
    storage: S,
    vocab: V,
    norms: Option<NdNorms>,
}

impl<V, S> Embeddings<V, S>
where
    V: Vocab,
{
    /// Construct an embeddings from a vocabulary, storage, and norms.
    ///
    /// The embeddings for known words **must** be
    /// normalized. However, this is not verified due to the high
    /// computational cost.
    pub fn new(metadata: Option<Metadata>, vocab: V, storage: S, norms: NdNorms) -> Self {
        assert_eq!(
            vocab.len(),
            norms.0.len(),
            "Vocab and norms do not have the same length"
        );

        Embeddings {
            metadata,
            vocab,
            storage,
            norms: Some(norms),
        }
    }
}

impl<V, S> Embeddings<V, S> {
    pub(crate) fn new_without_norms(metadata: Option<Metadata>, vocab: V, storage: S) -> Self {
        Embeddings {
            metadata,
            vocab,
            storage,
            norms: None,
        }
    }

    /// Decompose embeddings in its vocabulary, storage, and
    /// optionally norms.
    pub fn into_parts(self) -> (Option<Metadata>, V, S, Option<NdNorms>) {
        (self.metadata, self.vocab, self.storage, self.norms)
    }

    /// Get metadata.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Get metadata mutably.
    pub fn metadata_mut(&mut self) -> Option<&mut Metadata> {
        self.metadata.as_mut()
    }

    /// Get embedding norms.
    pub fn norms(&self) -> Option<&NdNorms> {
        self.norms.as_ref()
    }

    /// Set metadata.
    ///
    /// Returns the previously-stored metadata.
    pub fn set_metadata(&mut self, mut metadata: Option<Metadata>) -> Option<Metadata> {
        mem::swap(&mut self.metadata, &mut metadata);
        metadata
    }

    /// Get the embedding storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get the vocabulary.
    pub fn vocab(&self) -> &V {
        &self.vocab
    }
}

#[allow(clippy::len_without_is_empty)]
impl<V, S> Embeddings<V, S>
where
    V: Vocab,
    S: Storage,
{
    /// Return the length (in vector components) of the word embeddings.
    pub fn dims(&self) -> usize {
        self.storage.shape().1
    }

    /// Get the embedding of a word.
    pub fn embedding(&self, word: &str) -> Option<CowArray1<f32>> {
        match self.vocab.idx(word)? {
            WordIndex::Word(idx) => Some(self.storage.embedding(idx)),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((self.storage.shape().1,));
                for idx in indices {
                    embed += &self.storage.embedding(idx).as_view();
                }

                l2_normalize(embed.view_mut());

                Some(CowArray::Owned(embed))
            }
        }
    }

    /// Get the embedding and original norm of a word.
    ///
    /// Returns for a word:
    ///
    /// * The word embedding.
    /// * The norm of the embedding before normalization to a unit vector.
    ///
    /// The original embedding can be reconstructed by multiplying all
    /// embedding components by the original norm.
    ///
    /// If the model does not have associated norms, *1* will be
    /// returned as the norm for vocabulary words.
    pub fn embedding_with_norm(&self, word: &str) -> Option<EmbeddingWithNorm> {
        match self.vocab.idx(word)? {
            WordIndex::Word(idx) => Some(EmbeddingWithNorm {
                embedding: self.storage.embedding(idx),
                norm: self.norms().map(|n| n.norm(idx)).unwrap_or(1.),
            }),
            WordIndex::Subword(indices) => {
                let mut embed = Array1::zeros((self.storage.shape().1,));
                for idx in indices {
                    embed += &self.storage.embedding(idx).as_view();
                }

                let norm = l2_normalize(embed.view_mut());

                Some(EmbeddingWithNorm {
                    embedding: CowArray::Owned(embed),
                    norm,
                })
            }
        }
    }

    /// Get an iterator over pairs of words and the corresponding embeddings.
    pub fn iter(&self) -> Iter {
        Iter {
            storage: &self.storage,
            inner: self.vocab.words().iter().enumerate(),
        }
    }

    /// Get an iterator over triples of words, embeddings, and norms.
    ///
    /// Returns an iterator that returns triples of:
    ///
    /// * A word.
    /// * Its word embedding.
    /// * The original norm of the embedding before normalization to a unit vector.
    ///
    /// The original embedding can be reconstructed by multiplying all
    /// embedding components by the original norm.
    ///
    /// If the model does not have associated norms, the norm is
    /// always *1*.
    pub fn iter_with_norms(&self) -> IterWithNorms {
        IterWithNorms {
            storage: &self.storage,
            norms: self.norms(),
            inner: self.vocab.words().iter().enumerate(),
        }
    }

    /// Get the vocabulary size.
    ///
    /// The vocabulary size excludes subword units.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }
}

macro_rules! impl_embeddings_from(
    ($vocab:ty, $storage:ty, $storage_wrap:ty) => {
        impl From<Embeddings<$vocab, $storage>> for Embeddings<VocabWrap, $storage_wrap> {
            fn from(from: Embeddings<$vocab, $storage>) -> Self {
                let (metadata, vocab, storage, norms) = from.into_parts();
                Embeddings {
                    metadata,
                    vocab: vocab.into(),
                    storage: storage.into(),
                    norms,
                }
            }
        }
    }
);

// Hmpf. We with the blanket From<T> for T implementation, we need
// specialization to generalize this.
impl_embeddings_from!(SimpleVocab, NdArray, StorageWrap);
impl_embeddings_from!(SimpleVocab, NdArray, StorageViewWrap);
impl_embeddings_from!(SimpleVocab, MmapArray, StorageWrap);
impl_embeddings_from!(SimpleVocab, MmapArray, StorageViewWrap);
impl_embeddings_from!(SimpleVocab, QuantizedArray, StorageWrap);
impl_embeddings_from!(SubwordVocab, NdArray, StorageWrap);
impl_embeddings_from!(SubwordVocab, NdArray, StorageViewWrap);
impl_embeddings_from!(SubwordVocab, MmapArray, StorageWrap);
impl_embeddings_from!(SubwordVocab, MmapArray, StorageViewWrap);
impl_embeddings_from!(SubwordVocab, QuantizedArray, StorageWrap);

impl<'a, V, S> IntoIterator for &'a Embeddings<V, S>
where
    V: Vocab,
    S: Storage,
{
    type Item = (&'a str, CowArray1<'a, f32>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<V, S> MmapEmbeddings for Embeddings<V, S>
where
    Self: Sized,
    V: ReadChunk,
    S: MmapChunk,
{
    fn mmap_embeddings(read: &mut BufReader<File>) -> Result<Self, Error> {
        let header = Header::read_chunk(read)?;
        let chunks = header.chunk_identifiers();
        ensure!(!chunks.is_empty(), "Embedding file without chunks.");

        let metadata = if header.chunk_identifiers()[0] == ChunkIdentifier::Metadata {
            Some(Metadata::read_chunk(read)?)
        } else {
            None
        };

        let vocab = V::read_chunk(read)?;
        let storage = S::mmap_chunk(read)?;
        let norms = NdNorms::read_chunk(read).ok();

        Ok(Embeddings {
            metadata,
            vocab,
            storage,
            norms,
        })
    }
}

impl<V, S> ReadEmbeddings for Embeddings<V, S>
where
    V: ReadChunk,
    S: ReadChunk,
{
    fn read_embeddings<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let header = Header::read_chunk(read)?;
        let chunks = header.chunk_identifiers();
        ensure!(!chunks.is_empty(), "Embedding file without chunks.");

        let metadata = if header.chunk_identifiers()[0] == ChunkIdentifier::Metadata {
            Some(Metadata::read_chunk(read)?)
        } else {
            None
        };

        let vocab = V::read_chunk(read)?;
        let storage = S::read_chunk(read)?;
        let norms = NdNorms::read_chunk(read).ok();

        Ok(Embeddings {
            metadata,
            vocab,
            storage,
            norms,
        })
    }
}

impl<V, S> WriteEmbeddings for Embeddings<V, S>
where
    V: WriteChunk,
    S: WriteChunk,
{
    fn write_embeddings<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        let mut chunks = match self.metadata {
            Some(ref metadata) => vec![metadata.chunk_identifier()],
            None => vec![],
        };

        chunks.extend_from_slice(&[
            self.vocab.chunk_identifier(),
            self.storage.chunk_identifier(),
        ]);

        Header::new(chunks).write_chunk(write)?;
        if let Some(ref metadata) = self.metadata {
            metadata.write_chunk(write)?;
        }

        self.vocab.write_chunk(write)?;
        self.storage.write_chunk(write)?;
        Ok(())
    }
}

/// Quantizable embedding matrix.
pub trait Quantize<V> {
    /// Quantize the embedding matrix.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    ///
    /// The xorshift PRNG is used for picking the initial quantizer
    /// centroids.
    fn quantize<T>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> Embeddings<V, QuantizedArray>
    where
        T: TrainPQ<f32>,
    {
        self.quantize_using::<T, _>(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            normalize,
            &mut XorShiftRng::from_entropy(),
        )
    }

    /// Quantize the embedding matrix using the provided RNG.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    fn quantize_using<T, R>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
        rng: &mut R,
    ) -> Embeddings<V, QuantizedArray>
    where
        T: TrainPQ<f32>,
        R: Rng;
}

impl<V, S> Quantize<V> for Embeddings<V, S>
where
    V: Vocab,
    S: StorageView,
{
    fn quantize_using<T, R>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
        rng: &mut R,
    ) -> Embeddings<V, QuantizedArray>
    where
        T: TrainPQ<f32>,
        R: Rng,
    {
        let quantized_storage = self.storage().quantize_using::<T, R>(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            normalize,
            rng,
        );

        Embeddings {
            metadata: self.metadata().cloned(),
            vocab: self.vocab.clone(),
            storage: quantized_storage,
            norms: self.norms().cloned(),
        }
    }
}

/// An embedding with its (pre-normalization) l2 norm.
pub struct EmbeddingWithNorm<'a> {
    pub embedding: CowArray1<'a, f32>,
    pub norm: f32,
}

impl<'a> EmbeddingWithNorm<'a> {
    // Compute the unnormalized embedding.
    pub fn into_unnormalized(self) -> Array1<f32> {
        let mut unnormalized = self.embedding.into_owned();
        unnormalized *= self.norm;
        unnormalized
    }
}

/// Iterator over embeddings.
pub struct Iter<'a> {
    storage: &'a Storage,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, CowArray1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(idx, word)| (word.as_str(), self.storage.embedding(idx)))
    }
}

/// Iterator over embeddings.
pub struct IterWithNorms<'a> {
    storage: &'a Storage,
    norms: Option<&'a NdNorms>,
    inner: Enumerate<slice::Iter<'a, String>>,
}

impl<'a> Iterator for IterWithNorms<'a> {
    type Item = (&'a str, EmbeddingWithNorm<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(idx, word)| {
            (
                word.as_str(),
                EmbeddingWithNorm {
                    embedding: self.storage.embedding(idx),
                    norm: self.norms.map(|n| n.0[idx]).unwrap_or(1.),
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Seek, SeekFrom};

    use toml::toml;

    use super::Embeddings;
    use crate::io::{MmapEmbeddings, ReadEmbeddings, WriteEmbeddings};
    use crate::metadata::Metadata;
    use crate::storage::{MmapArray, NdArray, StorageView};
    use crate::vocab::SimpleVocab;
    use crate::word2vec::ReadWord2VecRaw;

    fn test_embeddings() -> Embeddings<SimpleVocab, NdArray> {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary_raw(&mut reader).unwrap()
    }

    fn test_metadata() -> Metadata {
        Metadata(toml! {
            [hyperparameters]
            dims = 300
            ns = 5

            [description]
            description = "Test model"
            language = "de"
        })
    }

    #[test]
    fn mmap() {
        let check_embeds = test_embeddings();
        let mut reader = BufReader::new(File::open("testdata/similarity.fifu").unwrap());
        let embeds: Embeddings<SimpleVocab, MmapArray> =
            Embeddings::mmap_embeddings(&mut reader).unwrap();
        assert_eq!(embeds.vocab(), check_embeds.vocab());
        assert_eq!(embeds.storage().view(), check_embeds.storage().view());
    }

    #[test]
    fn write_read_simple_roundtrip() {
        let check_embeds = test_embeddings();
        let mut cursor = Cursor::new(Vec::new());
        check_embeds.write_embeddings(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let embeds: Embeddings<SimpleVocab, NdArray> =
            Embeddings::read_embeddings(&mut cursor).unwrap();
        assert_eq!(embeds.storage().view(), check_embeds.storage().view());
        assert_eq!(embeds.vocab(), check_embeds.vocab());
    }

    #[test]
    fn write_read_simple_metadata_roundtrip() {
        let mut check_embeds = test_embeddings();
        check_embeds.set_metadata(Some(test_metadata()));

        let mut cursor = Cursor::new(Vec::new());
        check_embeds.write_embeddings(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let embeds: Embeddings<SimpleVocab, NdArray> =
            Embeddings::read_embeddings(&mut cursor).unwrap();
        assert_eq!(embeds.storage().view(), check_embeds.storage().view());
        assert_eq!(embeds.vocab(), check_embeds.vocab());
    }
}
