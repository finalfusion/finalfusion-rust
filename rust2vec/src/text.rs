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
//! use rust2vec::prelude::*;
//!
//! let mut reader = BufReader::new(File::open("testdata/similarity.txt").unwrap());
//!
//! // Read the embeddings. The second arguments specifies whether
//! // the embeddings should be normalized to unit vectors.
//! let embeddings = Embeddings::read_text_dims(&mut reader, true)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("Berlin");
//! ```

use std::io::{BufRead, Write};

use failure::{ensure, err_msg, Error, ResultExt};
use itertools::Itertools;
use ndarray::Array2;

use crate::embeddings::Embeddings;
use crate::storage::{NdArray, Storage};
use crate::util::l2_normalize;
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
    fn read_text(reader: &mut R, normalize: bool) -> Result<Self, Error>;
}

impl<R> ReadText<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text(reader: &mut R, normalize: bool) -> Result<Self, Error> {
        read_embeds(reader, None, normalize)
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
    fn read_text_dims(reader: &mut R, normalize: bool) -> Result<Self, Error>;
}

impl<R> ReadTextDims<R> for Embeddings<SimpleVocab, NdArray>
where
    R: BufRead,
{
    fn read_text_dims(reader: &mut R, normalize: bool) -> Result<Self, Error> {
        let mut dims = String::new();
        reader.read_line(&mut dims)?;

        let mut dims_iter = dims.split_whitespace();
        let vocab_len = dims_iter
            .next()
            .ok_or(failure::err_msg("Missing vocabulary size"))?
            .parse::<usize>()
            .context("Cannot parse vocabulary size")?;
        let embed_len = dims_iter
            .next()
            .ok_or(failure::err_msg("Missing vocabulary size"))?
            .parse::<usize>()
            .context("Cannot parse vocabulary size")?;

        read_embeds(reader, Some((vocab_len, embed_len)), normalize)
    }
}

fn read_embeds<R>(
    reader: &mut R,
    shape: Option<(usize, usize)>,
    normalize: bool,
) -> Result<Embeddings<SimpleVocab, NdArray>, Error>
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

        let word = parts.next().ok_or(err_msg("Empty line"))?.trim();
        words.push(word.to_owned());

        for part in parts {
            data.push(part.parse()?);
        }
    }

    let shape = if let Some((n_words, dims)) = shape {
        ensure!(
            words.len() == n_words,
            "Expected {} words, got: {}",
            n_words,
            words.len()
        );
        ensure!(
            data.len() / n_words == dims,
            "Expected {} dimensions, got: {}",
            dims,
            data.len() / n_words
        );
        (n_words, dims)
    } else {
        let dims = data.len() / words.len();
        (words.len(), dims)
    };

    ensure!(
        data.len() % shape.1 == 0,
        "Number of dimensions per vector is not constant"
    );

    let mut matrix = Array2::from_shape_vec(shape, data)?;

    if normalize {
        for mut embedding in matrix.outer_iter_mut() {
            l2_normalize(embedding.view_mut());
        }
    }

    Ok(Embeddings::new(SimpleVocab::new(words), NdArray(matrix)))
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
    fn write_text(&self, writer: &mut W) -> Result<(), Error>;
}

impl<W, V, S> WriteText<W> for Embeddings<V, S>
where
    W: Write,
    V: Vocab,
    S: Storage,
{
    /// Write the embeddings to the given writer.
    fn write_text(&self, write: &mut W) -> Result<(), Error> {
        for (word, embed) in self.iter() {
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
    fn write_text_dims(&self, writer: &mut W) -> Result<(), Error>;
}

impl<W, V, S> WriteTextDims<W> for Embeddings<V, S>
where
    W: Write,
    V: Vocab,
    S: Storage,
{
    fn write_text_dims(&self, write: &mut W) -> Result<(), Error> {
        writeln!(write, "{} {}", self.vocab().len(), self.dims())?;
        self.write_text(write)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};

    use crate::embeddings::Embeddings;
    use crate::storage::{NdArray, StorageView};
    use crate::vocab::{SimpleVocab, Vocab};
    use crate::word2vec::ReadWord2Vec;

    use super::{ReadText, ReadTextDims, WriteText, WriteTextDims};

    fn read_word2vec() -> Embeddings<SimpleVocab, NdArray> {
        let f = File::open("testdata/similarity.bin").unwrap();
        let mut reader = BufReader::new(f);
        Embeddings::read_word2vec_binary(&mut reader, false).unwrap()
    }

    #[test]
    fn read_text() {
        let f = File::open("testdata/similarity.nodims").unwrap();
        let mut reader = BufReader::new(f);
        let text_embeddings = Embeddings::read_text(&mut reader, false).unwrap();

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
        let text_embeddings = Embeddings::read_text_dims(&mut reader, false).unwrap();

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
        let embeddings = Embeddings::read_text(&mut reader, false).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text(&mut output).unwrap();

        assert_eq!(check, String::from_utf8_lossy(&output));
    }

    #[test]
    fn test_word2vec_text_dims_roundtrip() {
        let mut reader = BufReader::new(File::open("testdata/similarity.txt").unwrap());
        let mut check = String::new();
        reader.read_to_string(&mut check).unwrap();

        // Read embeddings.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let embeddings = Embeddings::read_text_dims(&mut reader, false).unwrap();

        // Write embeddings to a byte vector.
        let mut output = Vec::new();
        embeddings.write_text_dims(&mut output).unwrap();

        assert_eq!(check, String::from_utf8_lossy(&output));
    }
}
