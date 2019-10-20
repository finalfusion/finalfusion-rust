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

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{Array2, Axis, CowArray};

use crate::chunks::norms::NdNorms;
use crate::chunks::storage::{NdArray, Storage, StorageViewMut};
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

    /// Read the embeddings from the given buffered reader.
    ///
    /// In contrast to `read_word2vec_binary`, this constructor does
    /// not fail if a token contains invalid UTF-8. Instead, it will
    /// replace invalid UTF-8 characters by the replacement character.
    fn read_word2vec_binary_lossy(reader: &mut R) -> Result<Self>;
}

impl<R> ReadWord2Vec<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_word2vec_binary(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) =
            Embeddings::read_word2vec_binary_raw(reader, false)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms(norms)))
    }

    fn read_word2vec_binary_lossy(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) =
            Embeddings::read_word2vec_binary_raw(reader, true)?.into_parts();
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
    fn read_word2vec_binary_raw(reader: &mut R, lossy: bool) -> Result<Self>;
}

impl<R> ReadWord2VecRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_word2vec_binary_raw(reader: &mut R, lossy: bool) -> Result<Self> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        let mut matrix = Array2::zeros((n_words, embed_len));
        let mut words = Vec::with_capacity(n_words);

        for idx in 0..n_words {
            let word = read_string(reader, b' ', lossy)?;
            let word = word.trim();
            words.push(word.to_owned());

            let mut embedding = matrix.index_axis_mut(Axis(0), idx);

            reader
                .read_f32_into::<LittleEndian>(
                    embedding.as_slice_mut().expect("Matrix not contiguous"),
                )
                .map_err(|e| ErrorKind::io_error("Cannot read word embedding", e))?;
        }

        Ok(Embeddings::new_without_norms(
            None,
            SimpleVocab::new(words),
            NdArray::new(matrix),
        ))
    }
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
        writeln!(w, "{} {}", self.vocab().words_len(), self.dims())
            .map_err(|e| ErrorKind::io_error("Cannot write word embedding matrix shape", e))?;

        for (word, embed_norm) in self.iter_with_norms() {
            write!(w, "{} ", word).map_err(|e| ErrorKind::io_error("Cannot write token", e))?;

            let embed = if unnormalize {
                CowArray::from(embed_norm.into_unnormalized())
            } else {
                embed_norm.embedding
            };

            for v in embed.view() {
                w.write_f32::<LittleEndian>(*v)
                    .map_err(|e| ErrorKind::io_error("Cannot write embedding component", e))?;
            }

            w.write_all(&[0x0a])
                .map_err(|e| ErrorKind::io_error("Cannot write embedding separator", e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

    use approx::AbsDiffEq;

    use crate::chunks::storage::StorageView;
    use crate::chunks::vocab::Vocab;
    use crate::compat::word2vec::{ReadWord2Vec, ReadWord2VecRaw, WriteWord2Vec};
    use crate::embeddings::Embeddings;

    #[test]
    fn fails_on_invalid_utf8() {
        let f = File::open("testdata/utf8-incomplete.bin").unwrap();
        let mut reader = BufReader::new(f);
        assert!(Embeddings::read_word2vec_binary(&mut reader).is_err());
    }

    #[test]
    fn read_lossy() {
        let f = File::open("testdata/utf8-incomplete.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeds = Embeddings::read_word2vec_binary_lossy(&mut reader).unwrap();
        let words = embeds.vocab().words();
        assert_eq!(words, &["meren", "zee�n", "rivieren"]);
    }

    #[test]
    fn test_read_word2vec_binary() {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        let embeddings = Embeddings::read_word2vec_binary_raw(&mut reader, false).unwrap();
        assert_eq!(41, embeddings.vocab().words_len());
        assert_eq!(100, embeddings.dims());
    }

    #[test]
    fn test_word2vec_binary_roundtrip() {
        let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
        let mut check = Vec::new();
        reader.read_to_end(&mut check).unwrap();

        // Read embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_word2vec_binary_raw(&mut reader, false).unwrap();

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
        let embeddings_check = Embeddings::read_word2vec_binary_raw(&mut reader, false).unwrap();

        // Read normalized embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_word2vec_binary(&mut reader).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_word2vec_binary(&mut output, true).unwrap();

        let embeddings =
            Embeddings::read_word2vec_binary_raw(&mut Cursor::new(&output), false).unwrap();

        assert!(embeddings
            .storage()
            .view()
            .abs_diff_eq(&embeddings_check.storage().view(), 1e-6));
    }
}
