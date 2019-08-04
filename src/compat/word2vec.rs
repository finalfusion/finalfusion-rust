//! Reader and writer for the word2vec binary format.
//!
//! Embeddings in the word2vec binary format are these formats are
//! read as follows:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use finalfusion::prelude::*;
//!
//! let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
//!
//! // Read the embeddings.
//! let embeddings = Embeddings::read_word2vec_binary(&mut reader)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("Berlin");
//! ```

use std::io::{BufRead, Write};
use std::mem;
use std::slice::from_raw_parts_mut;

use byteorder::{LittleEndian, WriteBytesExt};
use ndarray::{Array2, Axis};

use crate::chunks::norms::NdNorms;
use crate::chunks::storage::{CowArray, NdArray, Storage, StorageViewMut};
use crate::chunks::vocab::{SimpleVocab, Vocab};
use crate::embeddings::Embeddings;
use crate::io::{ErrorKind, Result};
use crate::util::{l2_normalize_array, read_number, read_string};

/// Method to construct `Embeddings` from a word2vec binary file.
///
/// This trait defines an extension to `Embeddings` to read the word embeddings
/// from a file in word2vec binary format.
pub trait ReadWord2Vec<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the embeddings from the given buffered reader.
    fn read_word2vec_binary(reader: &mut R) -> Result<Self>;
}

impl<R> ReadWord2Vec<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_word2vec_binary(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Embeddings::read_word2vec_binary_raw(reader)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms(norms)))
    }
}

/// Read raw, unnormalized embeddings.
pub(crate) trait ReadWord2VecRaw<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the embeddings from the given buffered reader.
    fn read_word2vec_binary_raw(reader: &mut R) -> Result<Self>;
}

impl<R> ReadWord2VecRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_word2vec_binary_raw(reader: &mut R) -> Result<Self> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        let mut matrix = Array2::zeros((n_words, embed_len));
        let mut words = Vec::with_capacity(n_words);

        for idx in 0..n_words {
            let word = read_string(reader, b' ')?;
            let word = word.trim();
            words.push(word.to_owned());

            let mut embedding = matrix.index_axis_mut(Axis(0), idx);

            {
                let mut embedding_raw = unsafe {
                    typed_to_bytes(embedding.as_slice_mut().expect("Matrix not contiguous"))
                };
                reader
                    .read_exact(&mut embedding_raw)
                    .map_err(|e| ErrorKind::io_error("Cannot read word embedding", e))?;
            }
        }

        Ok(Embeddings::new_without_norms(
            None,
            SimpleVocab::new(words),
            NdArray(matrix),
        ))
    }
}

unsafe fn typed_to_bytes<T>(slice: &mut [T]) -> &mut [u8] {
    from_raw_parts_mut(
        slice.as_mut_ptr() as *mut u8,
        slice.len() * mem::size_of::<T>(),
    )
}

/// Method to write `Embeddings` to a word2vec binary file.
///
/// This trait defines an extension to `Embeddings` to write the word embeddings
/// to a file in word2vec binary format.
pub trait WriteWord2Vec<W>
where
    W: Write,
{
    /// Write the embeddings from the given writer.
    ///
    /// If `unnormalize` is `true`, the norms vector is used to
    /// restore the original vector magnitudes.
    fn write_word2vec_binary(&self, w: &mut W, unnormalize: bool) -> Result<()>;
}

impl<W, V, S> WriteWord2Vec<W> for Embeddings<V, S>
where
    W: Write,
    V: Vocab,
    S: Storage,
{
    fn write_word2vec_binary(&self, w: &mut W, unnormalize: bool) -> Result<()>
    where
        W: Write,
    {
        writeln!(w, "{} {}", self.vocab().len(), self.dims())
            .map_err(|e| ErrorKind::io_error("Cannot write word embedding matrix shape", e))?;

        for (word, embed_norm) in self.iter_with_norms() {
            write!(w, "{} ", word).map_err(|e| ErrorKind::io_error("Cannot write token", e))?;

            let embed = if unnormalize {
                CowArray::Owned(embed_norm.into_unnormalized())
            } else {
                embed_norm.embedding
            };

            for v in embed.as_view() {
                w.write_f32::<LittleEndian>(*v)
                    .map_err(|e| ErrorKind::io_error("Cannot write embedding component", e))?;
            }

            w.write_all(&[0x0a])
                .map_err(|e| ErrorKind::io_error("Cannot write embedding separator", e))?;
        }

        Ok(())
    }
}
