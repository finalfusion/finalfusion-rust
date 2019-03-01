//! Word embeddings.

use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::iter::Enumerate;
use std::mem;
use std::slice;

use failure::{ensure, Error};
use ndarray::Array1;

use crate::io::{
    private::{ChunkIdentifier, Header, MmapChunk, ReadChunk, WriteChunk},
    MmapEmbeddings, ReadEmbeddings, WriteEmbeddings,
};
use crate::metadata::Metadata;
use crate::storage::{
    CowArray, CowArray1, MmapArray, NdArray, QuantizedArray, Storage, StorageViewWrap, StorageWrap,
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
}

impl<V, S> Embeddings<V, S> {
    /// Construct an embeddings from a vocabulary and storage.
    pub fn new(metadata: Option<Metadata>, vocab: V, storage: S) -> Self {
        Embeddings {
            metadata,
            vocab,
            storage,
        }
    }

    /// Decompose embeddings in its vocabulary and storage.
    pub fn into_parts(self) -> (Option<Metadata>, V, S) {
        (self.metadata, self.vocab, self.storage)
    }

    /// Get metadata.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Get metadata mutably.
    pub fn metadata_mut(&mut self) -> Option<&mut Metadata> {
        self.metadata.as_mut()
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

    /// Get an iterator over pairs of words and the corresponding embeddings.
    pub fn iter(&self) -> Iter {
        Iter {
            storage: &self.storage,
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
                let (metadata, vocab, storage) = from.into_parts();
                Embeddings::new(metadata, vocab.into(), storage.into())
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

        Ok(Embeddings {
            metadata,
            vocab,
            storage,
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

        Ok(Embeddings {
            metadata,
            vocab,
            storage,
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

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Seek, SeekFrom};

    use toml::{toml, toml_internal};

    use super::Embeddings;
    use crate::io::{MmapEmbeddings, ReadEmbeddings, WriteEmbeddings};
    use crate::metadata::Metadata;
    use crate::storage::{MmapArray, NdArray, StorageView};
    use crate::vocab::SimpleVocab;
    use crate::word2vec::ReadWord2Vec;

    fn test_embeddings() -> Embeddings<SimpleVocab, NdArray> {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary(&mut reader, false).unwrap()
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
