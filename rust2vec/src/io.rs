use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error, ResultExt};

const MODEL_VERSION: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ChunkIdentifier {
    SimpleVocab = 1,
    NdArray = 2,
    SubwordVocab = 3,
    QuantizedArray = 4,
}

impl ChunkIdentifier {
    pub fn try_from(identifier: u32) -> Option<Self> {
        use ChunkIdentifier::*;

        match identifier {
            1 => Some(SimpleVocab),
            2 => Some(NdArray),
            3 => Some(SubwordVocab),
            4 => Some(QuantizedArray),
            _ => None,
        }
    }
}

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
    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek;
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

#[derive(Debug, Eq, PartialEq)]
pub struct Header {
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
    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        write.write_all(&[b'R', b'2', b'V'])?;
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
        let mut magic = [0u8; 3];
        read.read_exact(&mut magic)?;
        ensure!(
            magic == [b'R', b'2', b'V'],
            "File does not have rust2vec magic, expected: R2V, was: {}",
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
                .ok_or(format_err!("Unknown chunk identifier: {}", identifier))?;
            chunk_identifiers.push(chunk_identifier);
        }

        Ok(Header { chunk_identifiers })
    }
}

pub trait ReadEmbeddings
where
    Self: Sized,
{
    fn read_embeddings<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek;
}

pub trait MmapEmbeddings
where
    Self: Sized,
{
    fn mmap_embeddings(read: &mut BufReader<File>) -> Result<Self, Error>;
}

pub trait WriteEmbeddings {
    fn write_embeddings<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek;
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use super::{ChunkIdentifier, Header};
    use crate::io::{ReadChunk, WriteChunk};

    #[test]
    fn header_write_read_roundtrip() {
        let check_header = Header {
            chunk_identifiers: vec![ChunkIdentifier::SimpleVocab, ChunkIdentifier::NdArray],
        };
        let mut cursor = Cursor::new(Vec::new());
        check_header.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let header = Header::read_chunk(&mut cursor).unwrap();
        assert_eq!(header, check_header);
    }
}
