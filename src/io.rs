use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error, ResultExt};

const MODEL_VERSION: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ChunkIdentifier {
    SimpleVocab = 1,
    NdArray = 2,
}

impl ChunkIdentifier {
    fn try_from(identifier: u32) -> Option<Self> {
        use ChunkIdentifier::*;

        match identifier {
            1 => Some(SimpleVocab),
            2 => Some(NdArray),
            _ => None,
        }
    }
}

pub trait ReadChunk
where
    Self: Sized,
{
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error>;
}

pub trait WriteChunk {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error>;
}

pub trait TypeId {
    fn type_id() -> u32;
}

impl TypeId for f32 {
    fn type_id() -> u32 {
        10
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Header {
    chunk_identifiers: Vec<ChunkIdentifier>,
}

impl WriteChunk for Header {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error> {
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
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error> {
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

pub trait WriteModelBinary {
    fn write_model_binary(&self, write: &mut impl Write) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{ChunkIdentifier, Header};
    use crate::io::{ReadChunk, WriteChunk};

    #[test]
    fn header_write_read_roundtrip() {
        let check_header = Header {
            chunk_identifiers: vec![ChunkIdentifier::SimpleVocab, ChunkIdentifier::NdArray],
        };
        let mut serialized = Vec::new();
        check_header.write_chunk(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let header = Header::read_chunk(&mut cursor).unwrap();
        assert_eq!(header, check_header);
    }
}
