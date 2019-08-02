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
/// let embeddings: Embeddings<SimpleVocab, NdArray> =
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

pub(crate) mod private {
    use std::fmt::{self, Display};
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, Write};

    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

    use super::{Error, ErrorKind, Result};

    const MODEL_VERSION: u32 = 0;

    const MAGIC: [u8; 4] = [b'F', b'i', b'F', b'u'];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(u32)]
    pub enum ChunkIdentifier {
        Header = 0,
        SimpleVocab = 1,
        NdArray = 2,
        FinalfusionSubwordVocab = 3,
        QuantizedArray = 4,
        Metadata = 5,
        NdNorms = 6,
        FastTextSubwordVocab = 7,
    }

    impl ChunkIdentifier {
        pub fn try_from(identifier: u32) -> Option<Self> {
            use self::ChunkIdentifier::*;

            match identifier {
                1 => Some(SimpleVocab),
                2 => Some(NdArray),
                3 => Some(FinalfusionSubwordVocab),
                4 => Some(QuantizedArray),
                5 => Some(Metadata),
                6 => Some(NdNorms),
                7 => Some(FastTextSubwordVocab),
                _ => None,
            }
        }

        /// Read and ensure that the chunk has the given identifier.
        pub fn ensure_chunk_type<R>(read: &mut R, identifier: ChunkIdentifier) -> Result<()>
        where
            R: Read,
        {
            let chunk_id = read
                .read_u32::<LittleEndian>()
                .map_err(|e| ErrorKind::io_error("Cannot read chunk identifier", e))?;
            let chunk_id = ChunkIdentifier::try_from(chunk_id)
                .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
                .map_err(Error::from)?;
            if chunk_id != identifier {
                return Err(ErrorKind::Format(format!(
                    "Invalid chunk identifier, expected: {}, got: {}",
                    identifier, chunk_id
                ))
                .into());
            }

            Ok(())
        }
    }

    impl Display for ChunkIdentifier {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            use self::ChunkIdentifier::*;

            match self {
                Header => write!(f, "Header"),
                SimpleVocab => write!(f, "SimpleVocab"),
                NdArray => write!(f, "NdArray"),
                FastTextSubwordVocab => write!(f, "FastTextSubwordVocab"),
                FinalfusionSubwordVocab => write!(f, "FinalfusionSubwordVocab"),
                QuantizedArray => write!(f, "QuantizedArray"),
                Metadata => write!(f, "Metadata"),
                NdNorms => write!(f, "NdNorms"),
            }
        }
    }

    pub trait TypeId {
        /// Read and ensure that the data type is equal to `Self`.
        fn ensure_data_type<R>(read: &mut R) -> Result<()>
        where
            R: Read;

        fn type_id() -> u32;
    }

    macro_rules! typeid_impl {
        ($type:ty, $id:expr) => {
            impl TypeId for $type {
                fn ensure_data_type<R>(read: &mut R) -> Result<()>
                where
                    R: Read,
                {
                    let type_id = read
                        .read_u32::<LittleEndian>()
                        .map_err(|e| ErrorKind::io_error("Cannot read type identifier", e))?;
                    if type_id != Self::type_id() {
                        return Err(ErrorKind::Format(format!(
                            "Invalid type, expected: {}, got: {}",
                            Self::type_id(),
                            type_id
                        ))
                        .into());
                    }

                    Ok(())
                }

                fn type_id() -> u32 {
                    $id
                }
            }
        };
    }

    typeid_impl!(f32, 10);
    typeid_impl!(u8, 1);

    pub trait ReadChunk
    where
        Self: Sized,
    {
        fn read_chunk<R>(read: &mut R) -> Result<Self>
        where
            R: Read + Seek;
    }

    /// Memory-mappable chunks.
    pub trait MmapChunk
    where
        Self: Sized,
    {
        /// Memory map a chunk.
        ///
        /// The given `File` object should be positioned at the start of the chunk.
        fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self>;
    }

    pub trait WriteChunk {
        /// Get the identifier of a chunk.
        fn chunk_identifier(&self) -> ChunkIdentifier;

        fn write_chunk<W>(&self, write: &mut W) -> Result<()>
        where
            W: Write + Seek;
    }

    #[derive(Debug, Eq, PartialEq)]
    pub(crate) struct Header {
        chunk_identifiers: Vec<ChunkIdentifier>,
    }

    impl Header {
        pub fn new(chunk_identifiers: impl Into<Vec<ChunkIdentifier>>) -> Self {
            Header {
                chunk_identifiers: chunk_identifiers.into(),
            }
        }

        pub fn chunk_identifiers(&self) -> &[ChunkIdentifier] {
            &self.chunk_identifiers
        }
    }

    impl WriteChunk for Header {
        fn chunk_identifier(&self) -> ChunkIdentifier {
            ChunkIdentifier::Header
        }

        fn write_chunk<W>(&self, write: &mut W) -> Result<()>
        where
            W: Write + Seek,
        {
            write
                .write_all(&MAGIC)
                .map_err(|e| ErrorKind::io_error("Cannot write magic", e))?;
            write
                .write_u32::<LittleEndian>(MODEL_VERSION)
                .map_err(|e| ErrorKind::io_error("Cannot write model version", e))?;
            write
                .write_u32::<LittleEndian>(self.chunk_identifiers.len() as u32)
                .map_err(|e| ErrorKind::io_error("Cannot write chunk identifiers length", e))?;

            for &identifier in &self.chunk_identifiers {
                write
                    .write_u32::<LittleEndian>(identifier as u32)
                    .map_err(|e| ErrorKind::io_error("Cannot write chunk identifier", e))?;
            }

            Ok(())
        }
    }

    impl ReadChunk for Header {
        fn read_chunk<R>(read: &mut R) -> Result<Self>
        where
            R: Read + Seek,
        {
            // Magic and version ceremony.
            let mut magic = [0u8; 4];
            read.read_exact(&mut magic)
                .map_err(|e| ErrorKind::io_error("Cannot read magic", e))?;

            if magic != MAGIC {
                return Err(ErrorKind::Format(format!(
                    "Expected 'FiFu' as magic, got: {}",
                    String::from_utf8_lossy(&magic).into_owned()
                ))
                .into());
            }

            let version = read
                .read_u32::<LittleEndian>()
                .map_err(|e| ErrorKind::io_error("Cannot read model version", e))?;
            if version != MODEL_VERSION {
                return Err(
                    ErrorKind::Format(format!("Unknown finalfusion version: {}", version)).into(),
                );
            }

            // Read chunk identifiers.
            let chunk_identifiers_len = read
                .read_u32::<LittleEndian>()
                .map_err(|e| ErrorKind::io_error("Cannot read chunk identifiers length", e))?
                as usize;
            let mut chunk_identifiers = Vec::with_capacity(chunk_identifiers_len);
            for _ in 0..chunk_identifiers_len {
                let identifier = read
                    .read_u32::<LittleEndian>()
                    .map_err(|e| ErrorKind::io_error("Cannot read chunk identifier", e))?;
                let chunk_identifier = ChunkIdentifier::try_from(identifier)
                    .ok_or_else(|| {
                        ErrorKind::Format(format!("Unknown chunk identifier: {}", identifier))
                    })
                    .map_err(Error::from)?;
                chunk_identifiers.push(chunk_identifier);
            }

            Ok(Header { chunk_identifiers })
        }
    }

}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use crate::io::private::{ChunkIdentifier, Header, ReadChunk, WriteChunk};

    #[test]
    fn header_write_read_roundtrip() {
        let check_header =
            Header::new(vec![ChunkIdentifier::SimpleVocab, ChunkIdentifier::NdArray]);
        let mut cursor = Cursor::new(Vec::new());
        check_header.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = Header::read_chunk(&mut cursor).unwrap();
        assert_eq!(header, check_header);
    }
}
