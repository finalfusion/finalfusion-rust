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
use ndarray::{Array2, CowArray};

use crate::chunks::norms::NdNorms;
use crate::chunks::storage::{NdArray, Storage, StorageViewMut};
use crate::chunks::vocab::{SimpleVocab, Vocab};
use crate::embeddings::Embeddings;
use crate::error::{Error, Result};
use crate::util::{l2_normalize_array, read_number};

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

    /// Read the embeddings from the given buffered reader.
    ///
    /// In contrast to `read_text`, this constructor does not
    /// fail if a token contains invalid UTF-8. Instead, it will
    /// replace invalid UTF-8 characters by the replacement
    /// character.
    fn read_text_lossy(reader: &mut R) -> Result<Self>;
}

impl<R> ReadText<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_raw(reader, false)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms::new(norms)))
    }

    fn read_text_lossy(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_raw(reader, true)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms::new(norms)))
    }
}

pub(crate) trait ReadTextRaw<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the unnormalized embeddings from the given buffered reader.
    fn read_text_raw(reader: &mut R, lossy: bool) -> Result<Self>;
}

impl<R> ReadTextRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_raw(reader: &mut R, lossy: bool) -> Result<Self> {
        read_embeds(reader, None, lossy)
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

    /// Read the embeddings from the given buffered reader.
    ///
    /// In contrast to `read_text_dims`, this constructor does not
    /// fail if a token contains invalid UTF-8. Instead, it will
    /// replace invalid UTF-8 characters by the replacement
    /// character.
    fn read_text_dims_lossy(reader: &mut R) -> Result<Self>;
}

impl<R> ReadTextDims<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_dims(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_dims_raw(reader)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms::new(norms)))
    }

    fn read_text_dims_lossy(reader: &mut R) -> Result<Self> {
        let (_, vocab, mut storage, _) = Self::read_text_dims_raw_lossy(reader)?.into_parts();
        let norms = l2_normalize_array(storage.view_mut());

        Ok(Embeddings::new(None, vocab, storage, NdNorms::new(norms)))
    }
}

pub(crate) trait ReadTextDimsRaw<R>
where
    Self: Sized,
    R: BufRead,
{
    /// Read the unnormalized embeddings from the given buffered reader.
    fn read_text_dims_raw(reader: &mut R) -> Result<Self>;

    /// Read the unnormalized embeddings from the given buffered reader.
    ///
    /// This is the lossy variant of the method that accepts incorrect
    /// UTF-8.
    fn read_text_dims_raw_lossy(reader: &mut R) -> Result<Self>;
}

impl<R> ReadTextDimsRaw<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_dims_raw(reader: &mut R) -> Result<Self> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        read_embeds(reader, Some((n_words, embed_len)), false)
    }

    fn read_text_dims_raw_lossy(reader: &mut R) -> Result<Self> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        read_embeds(reader, Some((n_words, embed_len)), true)
    }
}

fn read_embeds<R>(
    reader: &mut R,
    shape: Option<(usize, usize)>,
    lossy: bool,
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

    loop {
        let mut buf = Vec::new();
        match reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| Error::io_error("Cannot read line from embedding file", e))?
        {
            0 => break,
            n => {
                if buf[n - 1] == b'\n' {
                    buf.pop();
                }
            }
        };

        let line = if lossy {
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::from_utf8(buf)
                .map_err(|e| Error::Format(format!("Token contains invalid UTF-8: {}", e)))?
        };

        let mut parts = line
            .split(|c: char| c.is_ascii_whitespace())
            .filter(|part| !part.is_empty());

        let word = parts
            .next()
            .ok_or_else(|| Error::Format(String::from("Spurious empty line")))?
            .trim_matches(|c: char| c.is_ascii_whitespace());
        words.push(word.to_owned());

        for part in parts {
            data.push(part.parse().map_err(|e| {
                Error::Format(format!("Cannot parse vector component '{}': {}", part, e))
            })?);
        }
    }

    let shape = if let Some((n_words, dims)) = shape {
        if words.len() != n_words {
            return Err(Error::Format(format!(
                "Incorrect vocabulary size, expected: {}, got: {}",
                n_words,
                words.len()
            )));
        }

        if data.len() / n_words != dims {
            return Err(Error::Format(format!(
                "Incorrect embedding dimensionality, expected: {}, got: {}",
                dims,
                data.len() / n_words,
            )));
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
        NdArray::new(matrix),
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
                CowArray::from(embed_norm.into_unnormalized())
            } else {
                embed_norm.embedding
            };

            let embed_str = embed.view().iter().map(ToString::to_string).join(" ");
            writeln!(write, "{} {}", word, embed_str)
                .map_err(|e| Error::io_error("Cannot write word embedding", e))?;
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
        writeln!(write, "{} {}", self.vocab().words_len(), self.dims())
            .map_err(|e| Error::io_error("Cannot write word embedding matrix shape", e))?;
        self.write_text(write, unnormalize)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

    use approx::AbsDiffEq;

    use crate::chunks::storage::{NdArray, StorageView};
    use crate::chunks::vocab::{SimpleVocab, Vocab};
    use crate::compat::word2vec::ReadWord2VecRaw;
    use crate::embeddings::Embeddings;

    use super::{ReadText, ReadTextDims, ReadTextDimsRaw, ReadTextRaw, WriteText, WriteTextDims};

    fn read_word2vec() -> Embeddings<SimpleVocab, NdArray> {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        Embeddings::read_word2vec_binary_raw(&mut reader, false).unwrap()
    }

    #[test]
    fn fails_on_invalid_utf8() {
        let f = File::open("testdata/utf8-incomplete.txt").unwrap();
        let mut reader = BufReader::new(f);
        assert!(Embeddings::read_text(&mut reader).is_err());
    }

    #[test]
    fn fails_on_invalid_utf8_dims() {
        let f = File::open("testdata/utf8-incomplete.dims").unwrap();
        let mut reader = BufReader::new(f);
        assert!(Embeddings::read_text_dims(&mut reader).is_err());
    }

    #[test]
    fn read_lossy() {
        let f = File::open("testdata/utf8-incomplete.txt").unwrap();
        let mut reader = BufReader::new(f);
        let embeds = Embeddings::read_text_lossy(&mut reader).unwrap();
        let words = embeds.vocab().words();
        assert_eq!(words, &["meren", "zee�n", "rivieren"]);
    }

    #[test]
    fn read_dims_lossy() {
        let f = File::open("testdata/utf8-incomplete.dims").unwrap();
        let mut reader = BufReader::new(f);
        let embeds = Embeddings::read_text_dims_lossy(&mut reader).unwrap();
        let words = embeds.vocab().words();
        assert_eq!(words, &["meren", "zee�n", "rivieren"]);
    }

    #[test]
    fn read_text() {
        let f = File::open("testdata/similarity.nodims").unwrap();
        let mut reader = BufReader::new(f);
        let text_embeddings = Embeddings::read_text_raw(&mut reader, false).unwrap();

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
        let embeddings = Embeddings::read_text_raw(&mut reader, false).unwrap();

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
        let embeddings_check = Embeddings::read_text_raw(&mut reader, false).unwrap();

        // Read normalized embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_text(&mut reader).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text(&mut output, true).unwrap();

        let embeddings = Embeddings::read_text_raw(&mut Cursor::new(&output), false).unwrap();

        assert!(embeddings
            .storage()
            .view()
            .abs_diff_eq(&embeddings_check.storage().view(), 1e-6));
    }
}
