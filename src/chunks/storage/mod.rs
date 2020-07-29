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
