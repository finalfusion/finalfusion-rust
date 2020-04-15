//! Traits and error types for I/O.
//!
//! This module provides traits for reading embeddings
//! (`ReadEmbeddings`), memory mapping embeddings (`MmapEmbeddings`),
//! and writing embeddings (`WriteEmbeddings`). Moreover, the module
//! provides the `Error`, `ErrorKind`, and `Result` types that are
//! used for handling I/O errors throughout the crate.

use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};

use crate::error::Result;

/// Read finalfusion embeddings.
///
/// This trait is used to read embeddings in the finalfusion format.
/// Implementations are provided for the vocabulary and storage types
/// in this crate.
///
/// ```
/// use std::fs::File;
///
/// use finalfusion::prelude::*;
///
/// let mut f = File::open("testdata/similarity.fifu").unwrap();
/// let embeddings: Embeddings<VocabWrap, StorageWrap> =
///     Embeddings::read_embeddings(&mut f).unwrap();
/// ```
pub trait ReadEmbeddings
where
    Self: Sized,
{
    /// Read the embeddings.
    fn read_embeddings<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek;
}

/// Read finalfusion embeddings metadata.
///
/// This trait is used to read the metadata of embeddings in the
/// finalfusion format. This is typically faster than
/// `ReadEmbeddings::read_embeddings`.
///
/// ```
/// use std::fs::File;
///
/// use finalfusion::prelude::*;
/// use finalfusion::metadata::Metadata;
/// use finalfusion::io::ReadMetadata;
///
/// let mut f = File::open("testdata/similarity.fifu").unwrap();
/// let metadata: Option<Metadata> =
///     ReadMetadata::read_metadata(&mut f).unwrap();
/// ```
pub trait ReadMetadata
where
    Self: Sized,
{
    /// Read the metadata.
    fn read_metadata<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek;
}

/// Memory-map finalfusion embeddings.
///
/// This trait is used to read finalfusion embeddings while [memory
/// mapping](https://en.wikipedia.org/wiki/Mmap) the embedding matrix.
/// This leads to considerable memory savings, since the operating
/// system will load the relevant pages from disk on demand.
pub trait MmapEmbeddings
where
    Self: Sized,
{
    fn mmap_embeddings(read: &mut BufReader<File>) -> Result<Self>;
}

/// Write embeddings in finalfusion format.
///
/// This trait is used to write embeddings in finalfusion
/// format. Writing in finalfusion format is supported regardless of
/// the original format of the embeddings.
pub trait WriteEmbeddings {
    fn write_embeddings<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek;
}
