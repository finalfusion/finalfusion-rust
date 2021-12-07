//! Embedding matrix representations.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, CowArray, Ix1};

mod array;
#[cfg(feature = "memmap")]
pub use self::array::MmapArray;
pub use self::array::NdArray;

mod quantized;
#[cfg(feature = "memmap")]
pub use self::quantized::MmapQuantizedArray;
pub use self::quantized::{Quantize, QuantizedArray, Reconstruct};

mod wrappers;
pub use self::wrappers::{StorageViewWrap, StorageWrap};

/// Embedding matrix storage.
///
/// To allow for embeddings to be stored in different manners (e.g.
/// regular *n x d* matrix or as quantized vectors), this trait
/// abstracts over concrete storage types.
pub trait Storage {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1>;

    /// Retrieve multiple embeddings.
    fn embeddings(&self, indices: &[usize]) -> Array2<f32>;

    fn shape(&self) -> (usize, usize);
}

pub(crate) mod sealed {
    use crate::chunks::storage::Storage;

    /// Return a new storage from an existing Storage based on a mapping.
    pub trait CloneFromMapping {
        type Result: Storage;

        /// Construct a new Storage based on a mapping.
        ///
        /// The `i`th entry in the returned storage is based on `self.embedding(mapping[i])`.
        fn clone_from_mapping(&self, mapping: &[usize]) -> Self::Result;
    }
}

/// Storage that provide a view of the embedding matrix.
pub trait StorageView: Storage {
    /// Get a view of the embedding matrix.
    fn view(&self) -> ArrayView2<f32>;
}

/// Storage that provide a mutable view of the embedding matrix.
pub(crate) trait StorageViewMut: Storage {
    /// Get a view of the embedding matrix.
    fn view_mut(&mut self) -> ArrayViewMut2<f32>;
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use crate::chunks::io::WriteChunk;
    use byteorder::{LittleEndian, ReadBytesExt};

    use crate::storage::StorageWrap;

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[cfg(test)]
    pub(crate) fn test_storage_chunk_len(check_storage: StorageWrap) {
        for offset in 0..16u64 {
            let mut cursor = Cursor::new(Vec::new());
            cursor.seek(SeekFrom::Start(offset)).unwrap();
            check_storage.write_chunk(&mut cursor).unwrap();
            cursor.seek(SeekFrom::Start(offset)).unwrap();

            let chunk_size = read_chunk_size(&mut cursor);
            assert_eq!(
                cursor.read_to_end(&mut Vec::new()).unwrap(),
                chunk_size as usize
            );

            let data = cursor.into_inner();
            assert_eq!(data.len() as u64 - offset, check_storage.chunk_len(offset));
        }
    }
}
