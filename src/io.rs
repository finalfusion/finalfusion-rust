//! Traits and error types for I/O.
//!
//! This module provides traits for reading embeddings
//! (`ReadEmbeddings`), memory mapping embeddings (`MmapEmbeddings`),
//! and writing embeddings (`WriteEmbeddings`). Moreover, the module
//! provides the `Error`, `ErrorKind`, and `Result` types that are
//! used for handling I/O errors throughout the crate.

use std::fmt;
use std::fs::File;
use std::io;
use std::io::{BufReader, Read, Seek, Write};

use ndarray::ShapeError;

/// `Result` type alias for operations that can lead to I/O errors.
pub type Result<T> = ::std::result::Result<T, Error>;

/// I/O errors in reading or writing embeddings.
#[derive(Debug)]
pub enum Error {
    /// finalfusion errors.
    FinalFusion(ErrorKind),

    /// `ndarray` shape error.
    Shape(ShapeError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Error::*;
        match *self {
            FinalFusion(ref kind) => kind.fmt(f),
            Shape(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        use self::Error::*;

        match *self {
            FinalFusion(ErrorKind::Format(ref desc)) => desc,
            FinalFusion(ErrorKind::Io { ref desc, .. }) => desc,
            Shape(ref err) => err.description(),
        }
    }
}

#[derive(Debug)]
pub enum ErrorKind {
    /// Invalid file format.
    Format(String),

    /// I/O error.
    Io { desc: String, error: io::Error },
}

impl ErrorKind {
    pub fn io_error(desc: impl Into<String>, error: io::Error) -> Self {
        ErrorKind::Io {
            desc: desc.into(),
            error,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ErrorKind::*;
        match *self {
            Format(ref desc) => write!(f, "{}", desc),
            Io {
                ref desc,
                ref error,
            } => write!(f, "{}: {}", desc, error),
        }
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        Error::FinalFusion(kind)
    }
}

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
///
/// Memory mapping is currently not implemented for quantized
/// matrices.
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
