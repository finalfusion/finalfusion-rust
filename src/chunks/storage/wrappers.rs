use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::ArrayView2;

use super::{CowArray1, MmapArray, NdArray, QuantizedArray, Storage, StorageView};
use crate::chunks::io::{ChunkIdentifier, MmapChunk, ReadChunk, WriteChunk};
use crate::io::{Error, ErrorKind, Result};

/// Storage types wrapper.
///
/// This crate makes it possible to create fine-grained embedding
/// types, such as `Embeddings<SimpleVocab, NdArray>` or
/// `Embeddings<SubwordVocab, QuantizedArray>`. However, in some cases
/// it is more pleasant to have a single type that covers all
/// vocabulary and storage types. `VocabWrap` and `StorageWrap` wrap
/// all the vocabularies and storage types known to this crate such
/// that the type `Embeddings<VocabWrap, StorageWrap>` covers all
/// variations.
pub enum StorageWrap {
    NdArray(NdArray),
    QuantizedArray(QuantizedArray),
    MmapArray(MmapArray),
}

impl Storage for StorageWrap {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            StorageWrap::MmapArray(inner) => inner.embedding(idx),
            StorageWrap::NdArray(inner) => inner.embedding(idx),
            StorageWrap::QuantizedArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            StorageWrap::MmapArray(inner) => inner.shape(),
            StorageWrap::NdArray(inner) => inner.shape(),
            StorageWrap::QuantizedArray(inner) => inner.shape(),
        }
    }
}

impl From<MmapArray> for StorageWrap {
    fn from(s: MmapArray) -> Self {
        StorageWrap::MmapArray(s)
    }
}

impl From<NdArray> for StorageWrap {
    fn from(s: NdArray) -> Self {
        StorageWrap::NdArray(s)
    }
}

impl From<QuantizedArray> for StorageWrap {
    fn from(s: QuantizedArray) -> Self {
        StorageWrap::QuantizedArray(s)
    }
}

impl ReadChunk for StorageWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read
            .seek(SeekFrom::Current(0))
            .map_err(|e| ErrorKind::io_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
            .map_err(Error::from)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| ErrorKind::io_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageWrap::NdArray),
            ChunkIdentifier::QuantizedArray => {
                QuantizedArray::read_chunk(read).map(StorageWrap::QuantizedArray)
            }
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected one of: {} or {}, got: {}",
                ChunkIdentifier::NdArray,
                ChunkIdentifier::QuantizedArray,
                chunk_id
            ))
            .into()),
        }
    }
}

impl MmapChunk for StorageWrap {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
        let chunk_start_pos = read
            .seek(SeekFrom::Current(0))
            .map_err(|e| ErrorKind::io_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
            .map_err(Error::from)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| ErrorKind::io_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageWrap::MmapArray),
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))
            .into()),
        }
    }
}

impl WriteChunk for StorageWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            StorageWrap::MmapArray(inner) => inner.chunk_identifier(),
            StorageWrap::NdArray(inner) => inner.chunk_identifier(),
            StorageWrap::QuantizedArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            StorageWrap::MmapArray(inner) => inner.write_chunk(write),
            StorageWrap::NdArray(inner) => inner.write_chunk(write),
            StorageWrap::QuantizedArray(inner) => inner.write_chunk(write),
        }
    }
}

/// Wrapper for storage types that implement views.
///
/// This type covers the subset of storage types that implement
/// `StorageView`. See the `StorageWrap` type for more information.
pub enum StorageViewWrap {
    MmapArray(MmapArray),
    NdArray(NdArray),
}

impl Storage for StorageViewWrap {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.embedding(idx),
            StorageViewWrap::NdArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.shape(),
            StorageViewWrap::NdArray(inner) => inner.shape(),
        }
    }
}

impl StorageView for StorageViewWrap {
    fn view(&self) -> ArrayView2<f32> {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.view(),
            StorageViewWrap::NdArray(inner) => inner.view(),
        }
    }
}

impl From<MmapArray> for StorageViewWrap {
    fn from(s: MmapArray) -> Self {
        StorageViewWrap::MmapArray(s)
    }
}

impl From<NdArray> for StorageViewWrap {
    fn from(s: NdArray) -> Self {
        StorageViewWrap::NdArray(s)
    }
}

impl ReadChunk for StorageViewWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read
            .seek(SeekFrom::Current(0))
            .map_err(|e| ErrorKind::io_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
            .map_err(Error::from)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| ErrorKind::io_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageViewWrap::NdArray),
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))
            .into()),
        }
    }
}

impl WriteChunk for StorageViewWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.chunk_identifier(),
            StorageViewWrap::NdArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.write_chunk(write),
            StorageViewWrap::NdArray(inner) => inner.write_chunk(write),
        }
    }
}

impl MmapChunk for StorageViewWrap {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
        let chunk_start_pos = read
            .seek(SeekFrom::Current(0))
            .map_err(|e| ErrorKind::io_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| ErrorKind::Format(format!("Unknown chunk identifier: {}", chunk_id)))
            .map_err(Error::from)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| ErrorKind::io_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageViewWrap::MmapArray),
            _ => Err(ErrorKind::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))
            .into()),
        }
    }
}
