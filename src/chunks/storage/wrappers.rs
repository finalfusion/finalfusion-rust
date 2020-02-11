#[cfg(feature = "memmap")]
use std::fs::File;
#[cfg(feature = "memmap")]
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom, Write};

#[cfg(feature = "memmap")]
use super::{MmapArray, MmapQuantizedArray};
use super::{NdArray, QuantizedArray, Storage, StorageView};
#[cfg(feature = "memmap")]
use crate::chunks::io::MmapChunk;
use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::io::{Error, ErrorKind, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{ArrayView2, CowArray, Ix1};

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
    // Boxed: clippy complains about large variant otherwise. Boxing
    // does not seem to have a noticable impact on performance.
    QuantizedArray(Box<QuantizedArray>),
    #[cfg(feature = "memmap")]
    MmapArray(MmapArray),
    #[cfg(feature = "memmap")]
    MmapQuantizedArray(MmapQuantizedArray),
}

impl Storage for StorageWrap {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        match self {
            #[cfg(feature = "memmap")]
            StorageWrap::MmapArray(inner) => inner.embedding(idx),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.embedding(idx),
            StorageWrap::NdArray(inner) => inner.embedding(idx),
            StorageWrap::QuantizedArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            #[cfg(feature = "memmap")]
            StorageWrap::MmapArray(inner) => inner.shape(),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.shape(),
            StorageWrap::NdArray(inner) => inner.shape(),
            StorageWrap::QuantizedArray(inner) => inner.shape(),
        }
    }
}

#[cfg(feature = "memmap")]
impl From<MmapArray> for StorageWrap {
    fn from(s: MmapArray) -> Self {
        StorageWrap::MmapArray(s)
    }
}

#[cfg(feature = "memmap")]
impl From<MmapQuantizedArray> for StorageWrap {
    fn from(s: MmapQuantizedArray) -> Self {
        StorageWrap::MmapQuantizedArray(s)
    }
}

impl From<NdArray> for StorageWrap {
    fn from(s: NdArray) -> Self {
        StorageWrap::NdArray(s)
    }
}

impl From<QuantizedArray> for StorageWrap {
    fn from(s: QuantizedArray) -> Self {
        StorageWrap::QuantizedArray(Box::new(s))
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
            ChunkIdentifier::QuantizedArray => QuantizedArray::read_chunk(read)
                .map(Box::new)
                .map(StorageWrap::QuantizedArray),
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

#[cfg(feature = "memmap")]
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
            ChunkIdentifier::QuantizedArray => {
                MmapQuantizedArray::mmap_chunk(read).map(StorageWrap::MmapQuantizedArray)
            }
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
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageWrap::MmapArray(inner) => inner.chunk_identifier(),
            #[cfg(target_endian = "big")]
            StorageWrap::MmapArray(_inner) => unimplemented!(),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.chunk_identifier(),
            StorageWrap::NdArray(inner) => inner.chunk_identifier(),
            StorageWrap::QuantizedArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageWrap::MmapArray(inner) => inner.write_chunk(write),
            #[cfg(target_endian = "big")]
            StorageWrap::MmapArray(_inner) => unimplemented!(),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.write_chunk(write),
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
    #[cfg(all(feature = "memmap", target_endian = "little"))]
    MmapArray(MmapArray),
    NdArray(NdArray),
}

impl Storage for StorageViewWrap {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.embedding(idx),
            StorageViewWrap::NdArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.shape(),
            StorageViewWrap::NdArray(inner) => inner.shape(),
        }
    }
}

impl StorageView for StorageViewWrap {
    fn view(&self) -> ArrayView2<f32> {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.view(),
            StorageViewWrap::NdArray(inner) => inner.view(),
        }
    }
}

#[cfg(all(feature = "memmap", target_endian = "little"))]
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
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.chunk_identifier(),
            StorageViewWrap::NdArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.write_chunk(write),
            StorageViewWrap::NdArray(inner) => inner.write_chunk(write),
        }
    }
}

#[cfg(feature = "memmap")]
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
            #[cfg(target_endian = "little")]
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
