use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::iter::Enumerate;
use std::slice;

use failure::Error;
use ndarray::Array1;

use crate::io::{
    Header, MmapChunk, MmapEmbeddings, ReadChunk, ReadEmbeddings, WriteChunk, WriteEmbeddings,
};
use crate::storage::{CowArray, CowArray1, Storage, StorageViewWrap, StorageWrap};
use crate::util::l2_normalize;
use crate::vocab::{Vocab, VocabWrap, WordIndex};

/// A word similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug)]
pub struct WordSimilarity<'a> {
    pub word: &'a str,
    pub similarity: f32,
}

impl<'a> Ord for WordSimilarity<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.similarity > other.similarity {
            Ordering::Less
        } else if self.similarity < other.similarity {
            Ordering::Greater
        } else {
            self.word.cmp(other.word)
        }
    }
}

impl<'a> PartialOrd for WordSimilarity<'a> {
    fn partial_cmp(&self, other: &WordSimilarity) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Eq for WordSimilarity<'a> {}

impl<'a> PartialEq for WordSimilarity<'a> {
    fn eq(&self, other: &WordSimilarity) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

/// Word embeddings.
///
/// This data structure stores word embeddings (also known as *word vectors*)
/// and provides some useful methods on the embeddings, such as similarity
/// and analogy queries.
#[derive(Debug)]
pub struct Embeddings<V, S> {
    storage: S,
    vocab: V,
}

impl<V, S> Embeddings<V, S> {
    pub fn new(vocab: V, storage: S) -> Self {
        Embeddings { vocab, storage }
    }

    pub fn into_parts(self) -> (V, S) {
        (self.vocab, self.storage)
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
}

impl<V, S> Embeddings<V, S>
where
    V: Into<VocabWrap>,
    S: Into<StorageWrap>,
{
    pub fn into_storage(self) -> Embeddings<VocabWrap, StorageWrap> {
        // Note: we cannot implement the From/Into trait, because this would be
        // a specialization of the generic T -> T conversion in core.
        let (vocab, storage) = self.into_parts();
        Embeddings::new(vocab.into(), storage.into())
    }
}

impl<V, S> Embeddings<V, S>
where
    V: Into<VocabWrap>,
    S: Into<StorageViewWrap>,
{
    pub fn into_storage_view(self) -> Embeddings<VocabWrap, StorageViewWrap> {
        // Note: we cannot implement the From/Into trait, because this would be
        // a specialization of the generic T -> T conversion in core.
        let (vocab, storage) = self.into_parts();
        Embeddings::new(vocab.into(), storage.into())
    }
}

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
        Header::read_chunk(read)?;
        let vocab = V::read_chunk(read)?;
        let storage = S::mmap_chunk(read)?;

        Ok(Embeddings { vocab, storage })
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
        Header::read_chunk(read)?;
        let vocab = V::read_chunk(read)?;
        let storage = S::read_chunk(read)?;

        Ok(Embeddings { vocab, storage })
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
        let header = Header::new(vec![
            self.vocab.chunk_identifier(),
            self.storage.chunk_identifier(),
        ]);
        header.write_chunk(write)?;
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

    use super::Embeddings;
    use crate::io::{MmapEmbeddings, ReadEmbeddings, WriteEmbeddings};
    use crate::storage::{MmapArray, NdArray, StorageView};
    use crate::vocab::SimpleVocab;
    use crate::word2vec::ReadWord2Vec;

    fn test_embeddings() -> Embeddings<SimpleVocab, NdArray> {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        Embeddings::read_word2vec_binary(&mut reader, false).unwrap()
    }

    #[test]
    fn mmap() {
        let check_embeds = test_embeddings();
        let mut reader = BufReader::new(File::open("testdata/similarity.r2v").unwrap());
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
}
