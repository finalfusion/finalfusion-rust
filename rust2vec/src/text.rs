//! Readers and writers for text formats.

use std::io::{BufRead, Seek, SeekFrom, Write};

use failure::{ensure, err_msg, Error, ResultExt};
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis};

use crate::storage::Storage;
use crate::util::l2_normalize;
use crate::vocab::Vocab;

use super::*;

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
    R: BufRead + Seek,
{
    /// Read the embeddings from the given buffered reader.
    fn read_text(reader: &mut R, normalize: bool) -> Result<Self, Error>;
}

impl<R> ReadText<R> for Embeddings
where
    R: BufRead + Seek,
{
    fn read_text(reader: &mut R, normalize: bool) -> Result<Self, Error> {
        let (vocab_len, embed_len) = text_vectors_dims(reader)?;
        reader.seek(SeekFrom::Start(0))?;
        read_embeds(reader, vocab_len, embed_len, normalize)
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
    R: BufRead + Seek,
{
    /// Read the embeddings from the given buffered reader.
    fn read_text_dims(reader: &mut R, normalize: bool) -> Result<Self, Error>;
}

impl<R> ReadTextDims<R> for Embeddings
where
    R: BufRead + Seek,
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

        read_embeds(reader, vocab_len, embed_len, normalize)
    }
}

fn read_embeds<R>(
    reader: &mut R,
    vocab_len: usize,
    embed_len: usize,
    normalize: bool,
) -> Result<Embeddings, Error>
where
    R: BufRead,
{
    let mut matrix = Array2::zeros((vocab_len, embed_len));
    let mut words = Vec::with_capacity(vocab_len);

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let mut parts = line.split_whitespace();

        let word = match parts.next() {
            Some(word) => word,
            None => return Err(err_msg("Empty line")),
        };

        let word = word.trim();
        words.push(word.to_owned());

        let embedding: Array1<f32> = r#try!(parts.map(str::parse).collect());
        ensure!(
            embedding.shape()[0] == embed_len,
            "Expected embedding size: {}, got: {}",
            embed_len,
            embedding.shape()[0]
        );

        matrix.index_axis_mut(Axis(0), idx).assign(&embedding);
    }

    ensure!(
        words.len() == vocab_len,
        "Vocabulary size: {}, expected: {}",
        words.len(),
        vocab_len
    );

    if normalize {
        for mut embedding in matrix.outer_iter_mut() {
            l2_normalize(embedding.view_mut());
        }
    }

    Ok(Embeddings::new(
        Vocab::new_simple_vocab(words),
        Storage::NdArray(matrix),
    ))
}

pub fn text_vectors_dims<R>(reader: &mut R) -> Result<(usize, usize), Error>
where
    R: BufRead + Seek,
{
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let embed_size = line.split_whitespace().count() - 1;

    let n_words = 1 + reader.lines().count();

    Ok((n_words, embed_size))
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

impl<W> WriteText<W> for Embeddings
where
    W: Write,
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

impl<W> WriteTextDims<W> for Embeddings
where
    W: Write,
{
    fn write_text_dims(&self, write: &mut W) -> Result<(), Error> {
        writeln!(write, "{} {}", self.vocab().len(), self.embed_len())?;
        self.write_text(write)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};

    use crate::word2vec::ReadWord2Vec;
    use crate::Embeddings;

    use super::{ReadText, ReadTextDims, WriteText, WriteTextDims};

    fn read_word2vec() -> Embeddings {
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
        assert_eq!(text_embeddings.data(), embeddings.data());
    }

    #[test]
    fn read_text_dims() {
        let f = File::open("testdata/similarity.txt").unwrap();
        let mut reader = BufReader::new(f);
        let text_embeddings = Embeddings::read_text_dims(&mut reader, false).unwrap();

        let embeddings = read_word2vec();
        assert_eq!(text_embeddings.vocab().words(), embeddings.vocab().words());
        assert_eq!(text_embeddings.data(), embeddings.data());
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
