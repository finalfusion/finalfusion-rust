use std::fmt::{self, Display};
use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Error, Result};

const MODEL_VERSION: u32 = 0;

const MAGIC: [u8; 4] = [b'F', b'i', b'F', b'u'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ChunkIdentifier {
    Header = 0,
    SimpleVocab = 1,
    NdArray = 2,
    BucketSubwordVocab = 3,
    QuantizedArray = 4,
    Metadata = 5,
    NdNorms = 6,
    FastTextSubwordVocab = 7,
    ExplicitSubwordVocab = 8,
    FloretSubwordVocab = 9,
}

impl ChunkIdentifier {
    /// Read and ensure that the chunk has the given identifier.
    pub fn ensure_chunk_type<R>(read: &mut R, identifier: ChunkIdentifier) -> Result<()>
    where
        R: Read,
    {
        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;
        if chunk_id != identifier {
            return Err(Error::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                identifier, chunk_id
            )));
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
            ExplicitSubwordVocab => write!(f, "ExplicitSubwordVocab"),
            FloretSubwordVocab => write!(f, "FloretSubwordVocab"),
            BucketSubwordVocab => write!(f, "BucketSubwordVocab"),
            QuantizedArray => write!(f, "QuantizedArray"),
            Metadata => write!(f, "Metadata"),
            NdNorms => write!(f, "NdNorms"),
        }
    }
}

impl TryFrom<u32> for ChunkIdentifier {
    type Error = Error;

    fn try_from(identifier: u32) -> Result<Self> {
        use self::ChunkIdentifier::*;

        match identifier {
            1 => Ok(SimpleVocab),
            2 => Ok(NdArray),
            3 => Ok(BucketSubwordVocab),
            4 => Ok(QuantizedArray),
            5 => Ok(Metadata),
            6 => Ok(NdNorms),
            7 => Ok(FastTextSubwordVocab),
            8 => Ok(ExplicitSubwordVocab),
            9 => Ok(FloretSubwordVocab),
            unknown => Err(Error::UnknownChunkIdentifier(unknown)),
        }
    }
}

/// Trait defining identifiers for data types.
pub trait TypeId {
    /// Read and ensure that the data type is equal to `Self`.
    fn ensure_data_type<R>(read: &mut R) -> Result<()>
    where
        R: Read;

    /// Get this data type's identifier.
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
                    .map_err(|e| Error::read_error("Cannot read type identifier", e))?;
                if type_id != Self::type_id() {
                    return Err(Error::Format(format!(
                        "Invalid type, expected: {}, got: {}",
                        Self::type_id(),
                        type_id
                    )));
                }

                Ok(())
            }

            fn type_id() -> u32 {
                $id
            }
        }
    };
}

// floats starting at 10 to leave room for other integer types.
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

    /// Get the serialized length of a chunk.
    ///
    /// The `offset` of the chunk in the serialized data is required
    /// because some chunks store arrays aligned.
    fn chunk_len(&self, offset: u64) -> u64;

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

    fn chunk_len(&self, _offset: u64) -> u64 {
        // magic + model version (u32) + chunk ids len (u32), chunk ids (len * u32)
        (MAGIC.len()
            + mem::size_of_val(&MODEL_VERSION)
            + mem::size_of::<u32>()
            + self.chunk_identifiers.len() * mem::size_of::<u32>()) as u64
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_all(&MAGIC)
            .map_err(|e| Error::write_error("Cannot write magic", e))?;
        write
            .write_u32::<LittleEndian>(MODEL_VERSION)
            .map_err(|e| Error::write_error("Cannot write model version", e))?;
        write
            .write_u32::<LittleEndian>(self.chunk_identifiers.len() as u32)
            .map_err(|e| Error::write_error("Cannot write chunk identifiers length", e))?;

        for &identifier in &self.chunk_identifiers {
            write
                .write_u32::<LittleEndian>(identifier as u32)
                .map_err(|e| Error::write_error("Cannot write chunk identifier", e))?;
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
            .map_err(|e| Error::read_error("Cannot read magic", e))?;

        if magic != MAGIC {
            return Err(Error::Format(format!(
                "Expected 'FiFu' as magic, got: {}",
                String::from_utf8_lossy(&magic).into_owned()
            )));
        }

        let version = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read model version", e))?;
        if version != MODEL_VERSION {
            return Err(Error::Format(format!(
                "Unknown finalfusion version: {}",
                version
            )));
        }

        // Read chunk identifiers.
        let chunk_identifiers_len = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read chunk identifiers length", e))?
            as usize;
        let mut chunk_identifiers = Vec::with_capacity(chunk_identifiers_len);
        for _ in 0..chunk_identifiers_len {
            let identifier = read
                .read_u32::<LittleEndian>()
                .map_err(|e| Error::read_error("Cannot read chunk identifier", e))?;
            let chunk_identifier = ChunkIdentifier::try_from(identifier)?;
            chunk_identifiers.push(chunk_identifier);
        }

        Ok(Header { chunk_identifiers })
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use super::{ChunkIdentifier, Header, ReadChunk, WriteChunk};

    #[test]
    fn header_chunk_len_is_correct() {
        let check_header =
            Header::new(vec![ChunkIdentifier::SimpleVocab, ChunkIdentifier::NdArray]);
        let mut cursor = Cursor::new(Vec::new());
        check_header.write_chunk(&mut cursor).unwrap();

        let data = cursor.into_inner();
        assert_eq!(data.len() as u64, check_header.chunk_len(0));
    }

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
