//! Embedding matrix representations.

use ndarray::{ArrayView2, ArrayViewMut2, CowArray, Ix1};

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
