//! Embedding matrix representations.

use ndarray::{Array, Array2, ArrayView, Axis, Dimension, Ix1};

pub enum CowArray<'a, A, D> {
    Borrowed(ArrayView<'a, A, D>),
    Owned(Array<A, D>),
}

impl<'a, A, D> CowArray<'a, A, D>
where
    D: Dimension,
{
    pub fn as_view(&self) -> ArrayView<A, D> {
        match self {
            CowArray::Borrowed(borrow) => borrow.view(),
            CowArray::Owned(owned) => owned.view(),
        }
    }
}

impl<'a, A, D> CowArray<'a, A, D>
where
    A: Clone,
    D: Dimension,
{
    pub fn into_owned(self) -> Array<A, D> {
        match self {
            CowArray::Borrowed(borrow) => borrow.to_owned(),
            CowArray::Owned(owned) => owned,
        }
    }
}

/// 1D copy-on-write array.
pub type CowArray1<'a, A> = CowArray<'a, A, Ix1>;

/// Embedding matrix storage.
///
/// To allow for embeddings to be stored in different manners (e.g.
/// regular *n x d* matrix or as quantized vectors), this trait
/// abstracts over concrete storage types.
pub trait Storage {
    /// Get the embedding directionality.
    fn dims(&self) -> usize;

    /// Get the embedding `idx`.
    fn embedding(&self, idx: usize) -> CowArray1<f32>;
}

#[derive(Debug, PartialEq)]
pub struct NdArray(pub Array2<f32>);

impl Storage for NdArray {
    fn dims(&self) -> usize {
        self.0.cols()
    }

    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        CowArray::Borrowed(self.0.index_axis(Axis(0), idx))
    }
}

/// Normalization of embeddings by their L2 norms.
pub trait Normalize {
    /// Normalize embeddings by their L2 norms.
    fn normalize(&mut self);
}

impl Normalize for NdArray {
    fn normalize(&mut self) {
        for mut embedding in self.0.outer_iter_mut() {
            let l2norm = embedding.dot(&embedding).sqrt();
            if l2norm != 0f32 {
                embedding /= l2norm;
            }
        }
    }
}
