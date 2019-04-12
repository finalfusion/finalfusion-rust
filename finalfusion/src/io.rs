//! Traits for I/O.
//!
//! This module provides traits for reading embeddings
//! (`ReadEmbeddings`), memory mapping embeddings (`MmapEmbeddings`),
//! and writing embeddings (`WriteEmbeddings`).

use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};

use failure::Error;

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
    fn read_embeddings<R>(read: &mut R) -> Result<Self, Error>
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
    fn read_metadata<R>(read: &mut R) -> Result<Self, Error>
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
    fn mmap_embeddings(read: &mut BufReader<File>) -> Result<Self, Error>;
}

/// Write embeddings in finalfusion format.
///
/// This trait is used to write embeddings in finalfusion
/// format. Writing in finalfusion format is supported regardless of
/// the original format of the embeddings.
pub trait WriteEmbeddings {
    fn write_embeddings<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek;
}

pub(crate) mod private {
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, Write};

    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use failure::{ensure, format_err, Error, ResultExt};

    const MODEL_VERSION: u32 = 0;

    const MAGIC: [u8; 4] = [b'F', b'i', b'F', b'u'];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(u32)]
    pub enum ChunkIdentifier {
        Header = 0,
        SimpleVocab = 1,
        NdArray = 2,
        SubwordVocab = 3,
        QuantizedArray = 4,
        Metadata = 5,
        NdNorms = 6,
    }

    impl ChunkIdentifier {
        pub fn try_from(identifier: u32) -> Option<Self> {
            use self::ChunkIdentifier::*;

            match identifier {
                1 => Some(SimpleVocab),
                2 => Some(NdArray),
                3 => Some(SubwordVocab),
                4 => Some(QuantizedArray),
                5 => Some(Metadata),
                6 => Some(NdNorms),
                _ => None,
            }
        }
    }

    pub trait TypeId {
        fn type_id() -> u32;
    }

    macro_rules! typeid_impl {
        ($type:ty, $id:expr) => {
            impl TypeId for $type {
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
        fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
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
        fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self, Error>;
    }

    pub trait WriteChunk {
        /// Get the identifier of a chunk.
        fn chunk_identifier(&self) -> ChunkIdentifier;

        fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
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

        fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
        where
            W: Write + Seek,
        {
            write.write_all(&MAGIC)?;
            write.write_u32::<LittleEndian>(MODEL_VERSION)?;
            write.write_u32::<LittleEndian>(self.chunk_identifiers.len() as u32)?;

            for &identifier in &self.chunk_identifiers {
                write.write_u32::<LittleEndian>(identifier as u32)?
            }

            Ok(())
        }
    }

    impl ReadChunk for Header {
        fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
        where
            R: Read + Seek,
        {
            // Magic and version ceremony.
            let mut magic = [0u8; 4];
            read.read_exact(&mut magic)?;
            ensure!(
                magic == MAGIC,
                "File does not have finalfusion magic, expected: {}, was: {}",
                String::from_utf8_lossy(&MAGIC),
                String::from_utf8_lossy(&magic)
            );
            let version = read.read_u32::<LittleEndian>()?;
            ensure!(
                version == MODEL_VERSION,
                "Unknown model version, expected: {}, was: {}",
                MODEL_VERSION,
                version
            );

            // Read chunk identifiers.
            let chunk_identifiers_len = read.read_u32::<LittleEndian>()? as usize;
            let mut chunk_identifiers = Vec::with_capacity(chunk_identifiers_len);
            for _ in 0..chunk_identifiers_len {
                let identifier = read
                    .read_u32::<LittleEndian>()
                    .with_context(|e| format!("Cannot read chunk identifier: {}", e))?;
                let chunk_identifier = ChunkIdentifier::try_from(identifier)
                    .ok_or_else(|| format_err!("Unknown chunk identifier: {}", identifier))?;
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
