//! Embedding matrix representations.

use ndarray::{Array1, ArrayView2, ArrayViewMut2, CowArray, Ix1};

mod array;
pub use self::array::{MmapArray, NdArray};

mod quantized;
pub use self::quantized::{MmapQuantizedArray, Quantize, QuantizedArray};

mod wrappers;
pub use self::wrappers::{StorageViewWrap, StorageWrap};

/// Embedding matrix storage.
///
/// To allow for embeddings to be stored in different manners (e.g.
/// regular *n x d* matrix or as quantized vectors), this trait
/// abstracts over concrete storage types.
pub trait Storage {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1>;

    fn shape(&self) -> (usize, usize);
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

/// Storage that can be pruned.
pub trait StoragePrune: Storage {
    /// Prune a storage. Discard the vectors which need to be pruned off based on their indices.
    fn prune_storage(&self, toss_indices: &[usize]) -> StorageWrap;

    /// Find a nearest vector for each vector that need to be tossed.
    fn most_similar(
        &self,
        keep_indices: &[usize],
        toss_indices: &[usize],
        batch_size: usize,
    ) -> Array1<usize>;
}
