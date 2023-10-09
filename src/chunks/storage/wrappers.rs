#[cfg(feature = "memmap")]
use std::fs::File;
#[cfg(feature = "memmap")]
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array2, ArrayView2, CowArray, Ix1};

#[cfg(feature = "memmap")]
use super::{MmapArray, MmapQuantizedArray};
use super::{NdArray, QuantizedArray, Storage, StorageView};
#[cfg(feature = "memmap")]
use crate::chunks::io::MmapChunk;
use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::storage::sealed::CloneFromMapping;
use crate::error::{Error, Result};

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

    fn embeddings(&self, indices: &[usize]) -> Array2<f32> {
        match self {
            #[cfg(feature = "memmap")]
            StorageWrap::MmapArray(inner) => inner.embeddings(indices),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.embeddings(indices),
            StorageWrap::NdArray(inner) => inner.embeddings(indices),
            StorageWrap::QuantizedArray(inner) => inner.embeddings(indices),
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

impl CloneFromMapping for StorageWrap {
    type Result = StorageWrap;

    fn clone_from_mapping(&self, mapping: &[usize]) -> Self::Result {
        match self {
            StorageWrap::QuantizedArray(quant) => {
                StorageWrap::QuantizedArray(Box::new(quant.clone_from_mapping(mapping)))
            }
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(quant) => {
                StorageWrap::QuantizedArray(Box::new(quant.clone_from_mapping(mapping)))
            }
            #[cfg(feature = "memmap")]
            StorageWrap::MmapArray(mmapped) => {
                StorageWrap::NdArray(mmapped.clone_from_mapping(mapping))
            }
            StorageWrap::NdArray(array) => StorageWrap::NdArray(array.clone_from_mapping(mapping)),
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
            .stream_position()
            .map_err(|e| Error::read_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| Error::read_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageWrap::NdArray),
            ChunkIdentifier::QuantizedArray => QuantizedArray::read_chunk(read)
                .map(Box::new)
                .map(StorageWrap::QuantizedArray),
            _ => Err(Error::Format(format!(
                "Invalid chunk identifier, expected one of: {} or {}, got: {}",
                ChunkIdentifier::NdArray,
                ChunkIdentifier::QuantizedArray,
                chunk_id
            ))),
        }
    }
}

#[cfg(feature = "memmap")]
impl MmapChunk for StorageWrap {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
        let chunk_start_pos = read
            .stream_position()
            .map_err(|e| Error::read_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| Error::read_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageWrap::MmapArray),
            ChunkIdentifier::QuantizedArray => {
                MmapQuantizedArray::mmap_chunk(read).map(StorageWrap::MmapQuantizedArray)
            }
            _ => Err(Error::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))),
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

    fn chunk_len(&self, offset: u64) -> u64 {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageWrap::MmapArray(inner) => inner.chunk_len(offset),
            #[cfg(target_endian = "big")]
            StorageWrap::MmapArray(_inner) => unimplemented!(),
            #[cfg(feature = "memmap")]
            StorageWrap::MmapQuantizedArray(inner) => inner.chunk_len(offset),
            StorageWrap::NdArray(inner) => inner.chunk_len(offset),
            StorageWrap::QuantizedArray(inner) => inner.chunk_len(offset),
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

    fn embeddings(&self, indices: &[usize]) -> Array2<f32> {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.embeddings(indices),
            StorageViewWrap::NdArray(inner) => inner.embeddings(indices),
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

impl CloneFromMapping for StorageViewWrap {
    type Result = StorageViewWrap;

    fn clone_from_mapping(&self, mapping: &[usize]) -> Self::Result {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(mmapped) => {
                StorageViewWrap::NdArray(mmapped.clone_from_mapping(mapping))
            }
            StorageViewWrap::NdArray(array) => {
                StorageViewWrap::NdArray(array.clone_from_mapping(mapping))
            }
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
            .stream_position()
            .map_err(|e| Error::read_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| Error::read_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageViewWrap::NdArray),
            _ => Err(Error::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))),
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

    fn chunk_len(&self, offset: u64) -> u64 {
        match self {
            #[cfg(all(feature = "memmap", target_endian = "little"))]
            StorageViewWrap::MmapArray(inner) => inner.chunk_len(offset),
            StorageViewWrap::NdArray(inner) => inner.chunk_len(offset),
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
            .stream_position()
            .map_err(|e| Error::read_error("Cannot get storage chunk start position", e))?;

        let chunk_id = read
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read storage chunk identifier", e))?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)?;

        read.seek(SeekFrom::Start(chunk_start_pos))
            .map_err(|e| Error::read_error("Cannot seek to storage chunk start position", e))?;

        match chunk_id {
            #[cfg(target_endian = "little")]
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageViewWrap::MmapArray),
            _ => Err(Error::Format(format!(
                "Invalid chunk identifier, expected: {}, got: {}",
                ChunkIdentifier::NdArray,
                chunk_id
            ))),
        }
    }
}
