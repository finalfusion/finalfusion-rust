//! Reader and writer for the word2vec binary format.
//!
//! Embeddings in the word2vec binary format are these formats are
//! read as follows:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use rust2vec::Embeddings;
//! use rust2vec::word2vec::ReadWord2Vec;
//!
//! let mut reader = BufReader::new(File::open("testdata/similarity.bin").unwrap());
//!
//! // Read the embeddings. The second arguments specifies whether
//! // the embeddings should be normalized to unit vectors.
//! let embeddings = Embeddings::read_word2vec_binary(&mut reader, true)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("Berlin");
//! ```


use std::io::{BufRead, Write};
use std::mem;
use std::slice::from_raw_parts_mut;

use byteorder::{LittleEndian, WriteBytesExt};
use failure::{err_msg, Error};
use ndarray::{Array2, Axis};

use crate::storage::Storage;
use crate::util::l2_normalize;
use crate::vocab::Vocab;

use super::*;

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
    fn read_word2vec_binary(reader: &mut R, normalize: bool) -> Result<Self, Error>;
}

impl<R> ReadWord2Vec<R> for Embeddings
where
    R: BufRead,
{
    fn read_word2vec_binary(reader: &mut R, normalize: bool) -> Result<Self, Error> {
        let n_words = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b'\n')?;

        let mut matrix = Array2::zeros((n_words, embed_len));
        let mut words = Vec::with_capacity(n_words);

        for idx in 0..n_words {
            let word = read_string(reader, ' ' as u8)?;
            let word = word.trim();
            words.push(word.to_owned());

            let mut embedding = matrix.index_axis_mut(Axis(0), idx);

            {
                let mut embedding_raw = match embedding.as_slice_mut() {
                    Some(s) => unsafe { typed_to_bytes(s) },
                    None => return Err(err_msg("Matrix not contiguous")),
                };
                reader.read_exact(&mut embedding_raw)?;
            }
        }

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
}

fn read_number(reader: &mut BufRead, delim: u8) -> Result<usize, Error> {
    let field_str = read_string(reader, delim)?;
    Ok(field_str.parse()?)
}

fn read_string(reader: &mut BufRead, delim: u8) -> Result<String, Error> {
    let mut buf = Vec::new();
    reader.read_until(delim, &mut buf)?;
    buf.pop();
    Ok(String::from_utf8(buf)?)
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
    fn write_word2vec_binary(&self, w: &mut W) -> Result<(), Error>;
}

impl<W> WriteWord2Vec<W> for Embeddings
where
    W: Write,
{
    fn write_word2vec_binary(&self, w: &mut W) -> Result<(), Error>
    where
        W: Write,
    {
        write!(w, "{} {}\n", self.vocab().len(), self.embed_len())?;

        for (word, embed) in self.iter() {
            write!(w, "{} ", word)?;

            // Write embedding to a vector with little-endian encoding.
            for v in embed.as_view() {
                w.write_f32::<LittleEndian>(*v)?;
            }

            w.write(&[0x0a])?;
        }

        Ok(())
    }
}
