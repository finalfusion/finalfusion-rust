use std::alloc::{alloc, Layout};
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use cfg_if::cfg_if;
use ndarray::{Array, Array1, Dimension, Ix1, Ix2, ShapeBuilder, ShapeError, StrideShape};
use num_traits::Zero;

cfg_if! {
    if #[cfg(feature = "align16")] {
        pub type DefaultAlignment = Align16;
    } else if #[cfg(feature = "align32")] {
        pub type DefaultAlignment = Align32;
    } else if #[cfg(feature = "align64")] {
        pub type DefaultAlignment = Align64;
    } else {
        pub type DefaultAlignment = f32;
    }
}

/// `AlignedArray` construction errors.
#[derive(Debug, PartialEq)]
pub enum AlignedArrayError {
    /// The provided data has an incorrect alignment.
    IncorrectAlignment,

    /// ndarray shape error.
    Shape(ShapeError),
}

impl fmt::Display for AlignedArrayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::AlignedArrayError::*;
        match *self {
            IncorrectAlignment => write!(f, "Input array has incorrect alignment"),
            Shape(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for AlignedArrayError {
    fn description(&self) -> &str {
        use self::AlignedArrayError::*;

        match *self {
            IncorrectAlignment => "Input array has incorrect alignment",
            Shape(ref err) => err.description(),
        }
    }
}

/// Matrix with alignment on a boundary that is a multiple of the size
/// of type `Align`.
pub type AlignedArray2<Align, A> = AlignedArray<Align, A, Ix2>;

/// Wrapper for ndarray's `Array` that guarantees alignment on a
/// boundary that is a multiple of the size of the type `Align`.
pub struct AlignedArray<Align, A, D>
where
    D: Dimension,
{
    inner: Array<A, D>,
    _phantom: PhantomData<Align>,
}

impl<Align, A> AlignedArray<Align, A, Ix1> {
    /// Construct an `AlignedArray` from a `Vec`.
    ///
    /// This constructor fails when `data` is not aligned on a
    /// boundary that is a multiple of the size of `Align`.
    pub fn from_vec(data: Vec<A>) -> Result<Self, AlignedArrayError> {
        if data.as_ptr().align_offset(mem::align_of::<Align>()) != 0 {
            return Err(AlignedArrayError::IncorrectAlignment);
        }

        Ok(AlignedArray {
            inner: Array1::from_vec(data),
            _phantom: PhantomData,
        })
    }
}

impl<Align, A, D> AlignedArray<Align, A, D>
where
    D: Dimension,
{
    /// Construct an `AlignedArray`, filled with an element.
    ///
    /// This constructor creates an `AlignedArray` of shape `shape`,
    /// filled with `elem`, aligned on a boundary that is a multiple
    /// of the size of `Align`.
    pub fn from_elem<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape();

        let mut v = Vec::with_capacity_aligned::<Align>(shape.size());
        v.extend(iter::repeat(elem).take(shape.size()));

        AlignedArray {
            inner: Array::from_shape_vec(shape, v)
                .map_err(AlignedArrayError::Shape)
                .expect("Shape mismatch"),
            _phantom: PhantomData,
        }
    }

    /// Construct an `AlignedArray` from a `Vec`.
    ///
    /// This constructor crates an `AlignedArray`, using `data` as
    /// the backing storage with shape `shape`.
    ///
    /// Construction failes fails when `data` is not aligned on a
    /// boundary that is a multiple of the size of `Align`.
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, AlignedArrayError>
    where
        Sh: Into<StrideShape<D>>,
    {
        if data.as_ptr().align_offset(mem::align_of::<Align>()) != 0 {
            return Err(AlignedArrayError::IncorrectAlignment);
        }

        Ok(AlignedArray {
            inner: Array::from_shape_vec(shape, data).map_err(AlignedArrayError::Shape)?,
            _phantom: PhantomData,
        })
    }

    /// Construct an `AlignedArray`, filled with zeros.
    ///
    /// This constructor creates an `AlignedArray` of shape `shape`,
    /// filled with zeros, aligned on a boundary that is a multiple of
    /// the size of `Align`.
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(shape, A::zero())
    }
}

impl<Align, A, D> Deref for AlignedArray<Align, A, D>
where
    D: Dimension,
{
    type Target = Array<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Align, A, D> DerefMut for AlignedArray<Align, A, D>
where
    D: Dimension,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Align, A, D> fmt::Debug for AlignedArray<Align, A, D>
where
    A: fmt::Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AlignedArrayError {{ inner: {:?}}}", self.inner)
    }
}

/// Allocate an array type with aligned memory.
pub trait WithCapacityAligned<T> {
    /// Allocate an array type with aligned memory.
    ///
    /// The type with the length `capacity` is aligned on a boundary
    /// that is a multiple of the size of `Align`.
    fn with_capacity_aligned<Align>(capacity: usize) -> Self;
}

impl<T> WithCapacityAligned<T> for Vec<T> {
    fn with_capacity_aligned<Align>(capacity: usize) -> Self {
        // alloc() beheviour is undefined for zero-sized layouts.
        if capacity == 0 {
            return Vec::with_capacity(0);
        }

        let align_size = mem::size_of::<Align>();
        let type_size = mem::size_of::<T>();

        let layout = Layout::from_size_align(capacity * type_size, align_size).unwrap();

        // Allocate and wrap.
        unsafe {
            let p = alloc(layout);
            Vec::from_raw_parts(p as *mut T, 0, capacity)
        }
    }
}

#[allow(dead_code)]
#[repr(align(16))]
pub struct Align16 {
    _data: [u8; 16],
}

#[cfg(feature = "align32")]
#[repr(align(32))]
pub struct Align32 {
    _data: [u8; 32],
}

#[allow(dead_code)]
#[repr(align(64))]
pub struct Align64 {
    _data: [u8; 64],
}

#[cfg(test)]
mod tests {
    use std::mem;

    use super::{Align64, AlignedArray2, AlignedArrayError, WithCapacityAligned};

    #[test]
    fn test_aligned_64() {
        // Test various capacity sizes.
        for cap in 0..100 {
            let t: Vec<f32> = Vec::with_capacity_aligned::<Align64>(cap);
            assert_eq!(t.capacity(), cap);
            if cap != 0 {
                assert_eq!(t.as_ptr().align_offset(mem::align_of::<Align64>()), 0);
            }
        }
    }

    #[test]
    fn test_aligned_array_from_vec() {
        // Assuming a probability of accidental correct alignment of
        // 1/64, do some iterations to make accidental alignment throughout
        // all iterations extremely unlikely (7.5e-37).
        //
        // Collect rather than panicking, so that we can check whether the
        // correct error is used.
        let result: Result<Vec<AlignedArray2<Align64, _>>, _> = (0..20)
            .map(|_| {
                let data = vec![0f32; 1];
                AlignedArray2::from_shape_vec((1, 1), data)
            })
            .collect();

        assert_eq!(result.unwrap_err(), AlignedArrayError::IncorrectAlignment);
    }
}
