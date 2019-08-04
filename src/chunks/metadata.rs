//! Metadata chunks

use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use toml::Value;

use super::io::{ChunkIdentifier, Header, ReadChunk, WriteChunk};
use crate::io::{Error, ErrorKind, ReadMetadata, Result};

/// Embeddings metadata.
///
/// finalfusion metadata in TOML format.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata(pub Value);

impl ReadChunk for Metadata {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::Metadata)?;

        // Read chunk length.
        let chunk_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read chunk length", e))?
            as usize;

        // Read TOML data.
        let mut buf = vec![0; chunk_len];
        read.read_exact(&mut buf)
            .map_err(|e| ErrorKind::io_error("Cannot read TOML metadata", e))?;
        let buf_str = String::from_utf8(buf)
            .map_err(|e| ErrorKind::Format(format!("TOML metadata contains invalid UTF-8: {}", e)))
            .map_err(Error::from)?;

        Ok(Metadata(
            buf_str
                .parse::<Value>()
                .map_err(|e| ErrorKind::Format(format!("Cannot deserialize TOML metadata: {}", e)))
                .map_err(Error::from)?,
        ))
    }
}

impl WriteChunk for Metadata {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::Metadata
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        let metadata_str = self.0.to_string();

        write
            .write_u32::<LittleEndian>(self.chunk_identifier() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write metadata chunk identifier", e))?;
        write
            .write_u64::<LittleEndian>(metadata_str.len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write metadata length", e))?;
        write
            .write_all(metadata_str.as_bytes())
            .map_err(|e| ErrorKind::io_error("Cannot write metadata", e))?;

        Ok(())
    }
}

impl ReadMetadata for Option<Metadata> {
    fn read_metadata<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        let header = Header::read_chunk(read)?;
        let chunks = header.chunk_identifiers();

        if chunks.is_empty() {
            return Err(
                ErrorKind::Format(String::from("Embedding file does not contain chunks")).into(),
            );
        }

        if header.chunk_identifiers()[0] == ChunkIdentifier::Metadata {
            Ok(Some(Metadata::read_chunk(read)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use toml::toml;

    use super::Metadata;
    use crate::chunks::io::{ReadChunk, WriteChunk};

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    fn test_metadata() -> Metadata {
        Metadata(toml! {
            [hyperparameters]
            dims = 300
            ns = 5

            [description]
            description = "Test model"
            language = "de"
        })
    }

    #[test]
    fn metadata_correct_chunk_size() {
        let check_metadata = test_metadata();
        let mut cursor = Cursor::new(Vec::new());
        check_metadata.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }

    #[test]
    fn metadata_write_read_roundtrip() {
        let check_metadata = test_metadata();
        let mut cursor = Cursor::new(Vec::new());
        check_metadata.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let metadata = Metadata::read_chunk(&mut cursor).unwrap();
        assert_eq!(metadata, check_metadata);
    }
}
