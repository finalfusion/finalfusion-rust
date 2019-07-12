//! Readers and writers for text formats.
//!
//! This module provides two readers/writers:
//!
//! 1. `ReadText`/`WriteText`: word embeddings in text format. In this
//!    format, each line contains a word followed by its
//!    embedding. The word and the embedding vector components are
//!    separated by a space. This format is used by GloVe.
//! 2. `ReadTextDims`/`WriteTextDims`: this format is the same as (1),
//!    but the data is preceded by a line with the shape of the
//!    embedding matrix. This format is used by word2vec's text
//!    output.
//!
//! For example:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use finalfusion::prelude::*;
//!
//! let mut reader = BufReader::new(File::open("testdata/similarity.txt").unwrap());
//!
//! // Read the embeddings. The second arguments specifies whether
//! // the embeddings should be normalized to unit vectors.
//! let embeddings = Embeddings::read_text_dims(&mut reader)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("Berlin");
//! ```

use std::io::{BufRead, Write};

use itertools::Itertools;
use ndarray::Array2;

use crate::embeddings::Embeddings;
use crate::io::{Error, ErrorKind, Result};
use crate::norms::NdNorms;
use crate::storage::{CowArray, NdArray, Storage, StorageViewMut};
use crate::util::{l2_normalize_array, read_number};
use crate::vocab::{SimpleVocab, Vocab};

/// Method to construct `Embeddings` from a text file.
///
/// This trait defines an extension to `Embeddings` to read the word embeddings
/// from a text stream. The text should contain one word embedding per line in
/// the following format:
///
/// *word0 component_1 component_2 ... component_n*
pub trait ReadText<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the embeddings from the given buffered reader.
    fn read_text(reader: &mut R) -> Result<Self>;
}

impl<R> ReadText<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_raw(reader)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms(norms)))
    }
}

pub(crate) trait ReadTextRaw<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the unnormalized embeddings from the given buffered reader.
    fn read_text_raw(reader: &mut R) -> Result<Self>;
}

impl<R> ReadTextRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_raw(reader: &mut R) -> Result<Self> {
        read_embeds(reader, None)
    }
}

/// Method to construct `Embeddings` from a text file with dimensions.
///
/// This trait defines an extension to `Embeddings` to read the word embeddings
/// from a text stream. The text must contain as the first line the shape of
/// the embedding matrix:
///
/// *vocab_size n_components*
///
/// The remainder of the stream should contain one word embedding per line in
/// the following format:
///
/// *word0 component_1 component_2 ... component_n*
pub trait ReadTextDims<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the embeddings from the given buffered reader.
    fn read_text_dims(reader: &mut R) -> Result<Self>;
}

impl<R> ReadTextDims<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_dims(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_dims_raw(reader)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms(norms)))
    }
}

pub(crate) trait ReadTextDimsRaw<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the unnormalized embeddings from the given buffered reader.
    fn read_text_dims_raw(reader: &mut R) -> Result<Self>;
}

impl<R> ReadTextDimsRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_dims_raw(reader: &mut R) -> Result<Self> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        read_embeds(reader, Some((n_words, embed_len)))
    }
}

fn read_embeds<R>(
    reader: &mut R,
    shape: Option<(usize, usize)>,
) -> Result<Embeddings<SimpleVocab, NdArray>>
where
    R: BufRead,
{
    let (mut words, mut data) = if let Some((n_words, dims)) = shape {
        (
            Vec::with_capacity(n_words),
            Vec::with_capacity(n_words * dims),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();

        let word = parts
            .next()
            .ok_or_else(|| ErrorKind::Format(String::from("Spurious empty line")))?
            .trim();
        words.push(word.to_owned());

        for part in parts {
            data.push(part.parse().map_err(|e| {
                ErrorKind::Format(format!("Cannot parse vector component '{}': {}", part, e))
            })?);
        }
    }

    let shape = if let Some((n_words, dims)) = shape {
        if words.len() != n_words {
            return Err(ErrorKind::Format(format!(
                "Incorrect vocabulary size, expected: {}, got: {}",
                n_words,
                words.len()
            ))
            .into());
        }

        if data.len() / n_words != dims {
            return Err(ErrorKind::Format(format!(
                "Incorrect embedding dimensionality, expected: {}, got: {}",
                dims,
                data.len() / n_words,
            ))
            .into());
        };

        (n_words, dims)
    } else {
        let dims = data.len() / words.len();
        (words.len(), dims)
    };

    let matrix = Array2::from_shape_vec(shape, data).map_err(Error::Shape)?;

    Ok(Embeddings::new_without_norms(
        None,
        SimpleVocab::new(words),
        NdArray(matrix),
    ))
}

/// Method to write `Embeddings` to a text file.
///
/// This trait defines an extension to `Embeddings` to write the word embeddings
/// as text. The text will contain one word embedding per line in the following
/// format:
///
/// *word0 component_1 component_2 ... component_n*
pub trait WriteText<W>
where
    W: Write,
{
    /// Read the embeddings from the given buffered reader.
    ///
    /// If `unnormalize` is `true`, the norms vector is used to
    /// restore the original vector magnitudes.
    fn write_text(&self, writer: &mut W, unnormalize: bool) -> Result<()>;
}

impl<W, V, S> WriteText<W> for Embeddings<V, S>
where
    W: Write,
    V: Vocab,
    S: Storage,
{
    fn write_text(&self, write: &mut W, unnormalize: bool) -> Result<()> {
        for (word, embed_norm) in self.iter_with_norms() {
            let embed = if unnormalize {
                CowArray::Owned(embed_norm.into_unnormalized())
            } else {
                embed_norm.embedding
            };

            let embed_str = embed.as_view().iter().map(ToString::to_string).join(" ");
            writeln!(write, "{} {}", word, embed_str)?;
        }

        Ok(())
    }
}

/// Method to write `Embeddings` to a text file.
///
/// This trait defines an extension to `Embeddings` to write the word embeddings
/// as text. The text will contain one word embedding per line in the following
/// format:
///
/// *word0 component_1 component_2 ... component_n*
pub trait WriteTextDims<W>
where
    W: Write,
{
    /// Write the embeddings to the given writer.
    ///
    /// If `unnormalize` is `true`, the norms vector is used to
    /// restore the original vector magnitudes.
    fn write_text_dims(&self, writer: &mut W, unnormalize: bool) -> Result<()>;
}

impl<W, V, S> WriteTextDims<W> for Embeddings<V, S>
where
    W: Write,
    V: Vocab,
    S: Storage,
{
    fn write_text_dims(&self, write: &mut W, unnormalize: bool) -> Result<()> {
        writeln!(write, "{} {}", self.vocab().len(), self.dims())?;
        self.write_text(write, unnormalize)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

    use crate::embeddings::Embeddings;
    use crate::storage::{NdArray, StorageView};
    use crate::vocab::{SimpleVocab, Vocab};
    use crate::word2vec::ReadWord2VecRaw;

    use super::{ReadText, ReadTextDimsRaw, ReadTextRaw, WriteText, WriteTextDims};

    fn read_word2vec() -> Embeddings<SimpleVocab, NdArray> {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        Embeddings::read_word2vec_binary_raw(&mut reader).unwrap()
    }

    #[test]
    fn read_text() {
        let f = File::open("testdata/similarity.nodims").unwrap();
        let mut reader = BufReader::new(f);
        let text_embeddings = Embeddings::read_text_raw(&mut reader).unwrap();

        let embeddings = read_word2vec();
        assert_eq!(text_embeddings.vocab().words(), embeddings.vocab().words());
        assert_eq!(
            text_embeddings.storage().view(),
            embeddings.storage().view()
        );
    }

    #[test]
    fn read_text_dims() {
        let f = File::open("testdata/similarity.txt").unwrap();
        let mut reader = BufReader::new(f);
        let text_embeddings = Embeddings::read_text_dims_raw(&mut reader).unwrap();

        let embeddings = read_word2vec();
        assert_eq!(text_embeddings.vocab().words(), embeddings.vocab().words());
        assert_eq!(
            text_embeddings.storage().view(),
            embeddings.storage().view()
        );
    }

    #[test]
    fn test_word2vec_text_roundtrip() {
        let mut reader = BufReader::new(File::open("testdata/similarity.nodims").unwrap());
        let mut check = String::new();
        reader.read_to_string(&mut check).unwrap();

        // Read embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_text_raw(&mut reader).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text(&mut output, false).unwrap();

        assert_eq!(check, String::from_utf8_lossy(&output));
    }

    #[test]
    fn test_word2vec_text_dims_roundtrip() {
        let mut reader = BufReader::new(File::open("testdata/similarity.txt").unwrap());
        let mut check = String::new();
        reader.read_to_string(&mut check).unwrap();

        // Read embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_text_dims_raw(&mut reader).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text_dims(&mut output, false).unwrap();

        assert_eq!(check, String::from_utf8_lossy(&output));
    }

    #[test]
    fn test_word2vec_text_write_unnormalized() {
        let mut reader = BufReader::new(File::open("testdata/similarity.nodims").unwrap());

        // Read unnormalized embeddings
        let embeddings_check = Embeddings::read_text_raw(&mut reader).unwrap();

        // Read normalized embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_text(&mut reader).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text(&mut output, true).unwrap();

        let embeddings = Embeddings::read_text_raw(&mut Cursor::new(&output)).unwrap();

        assert!(embeddings
            .storage()
            .0
            .all_close(&embeddings_check.storage().0, 1e-6));
    }

}
