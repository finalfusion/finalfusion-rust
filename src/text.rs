use std::collections::HashMap;
use std::io::{BufRead, Write, Seek, SeekFrom};

use itertools::Itertools;
use ndarray::{Array, Axis, Ix1};

use super::*;

/// Method to construct `Embeddings` from a text file.
///
/// This trait defines an extension to `Embeddings` to read the word embeddings
/// from a text stream. The text should contain one word embedding per line in
/// the following format:
///
/// *word0 component_1 component_2 ... component_n*
pub trait ReadText<R>
    where R: BufRead + Seek
{
    /// Read the embeddings from the given buffered reader.
    fn read_text(reader: &mut R) -> Result<Embeddings>;
}

impl<R> ReadText<R> for Embeddings
    where R: BufRead + Seek
{
    fn read_text(reader: &mut R) -> Result<Embeddings> {
        let (n_words, embed_size) = text_vectors_dims(reader)?;
        let mut matrix = Array::zeros((n_words, embed_size));
        let mut indices = HashMap::new();
        let mut words = Vec::with_capacity(1);

        reader.seek(SeekFrom::Start(0))?;

        for (idx, line) in (0..n_words).zip(reader.lines()) {
            let line = line?;
            let mut parts = line.split_whitespace();

            let word = match parts.next() {
                Some(word) => word,
                None => return Err("Empty line".into()),
            };

            let word = word.trim();
            words.push(word.to_owned());
            indices.insert(word.to_owned(), idx);

            let embedding: Array<f32, Ix1> = try!(parts.map(str::parse).collect());
            matrix.subview_mut(Axis(0), idx).assign(&embedding);
        }


        Ok(super::embeddings::new_embeddings(matrix, embed_size, indices, words))
    }
}

pub fn text_vectors_dims<R>(reader: &mut R) -> Result<(usize, usize)>
    where R: BufRead + Seek
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
    where W: Write
{
    /// Read the embeddings from the given buffered reader.
    fn write_text(&self, writer: &mut W) -> Result<()>;
}


impl<W> WriteText<W> for Embeddings
    where W: Write
{
    fn write_text(&self, write: &mut W) -> Result<()> {
        for (word, embed) in self.words().iter().zip(self.data().outer_iter()) {
            let embed_str = embed.iter().map(ToString::to_string).join(" ");
            writeln!(write, "{} {}", word, embed_str)?;
        }

        Ok(())
    }
}
