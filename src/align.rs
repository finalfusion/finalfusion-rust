use std::alloc::{alloc, Layout};
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use cfg_if::cfg_if;
use ndarray::{Array, Ix2, ShapeBuilder, ShapeError};
use num_traits::Zero;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout<Align, A> {
    rows: usize,
    cols: usize,
    _align_phantom: PhantomData<Align>,
    _data_phantom: PhantomData<A>,
}

impl<Align, A> fmt::Debug for MatrixLayout<Align, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MatrixLayout {{ rows: {:?}, cols: {:?}}}",
            self.rows, self.cols
        )
    }
}

impl<Align, A> MatrixLayout<Align, A> {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(
            mem::size_of::<Align>() % mem::size_of::<A>() == 0,
            "Alignment should be multiple of data type"
        );

        MatrixLayout {
            rows,
            cols,
            _align_phantom: PhantomData,
            _data_phantom: PhantomData,
        }
    }

    /// Array shape with padding.
    pub fn shape_with_padding(&self) -> [usize; 2] {
        // The number of A that fits in an Align.
        let data_per_align = mem::size_of::<Align>() / mem::size_of::<A>();

        // Round up to get the padded length.
        let padded_len = (self.cols + data_per_align - 1) & !(data_per_align - 1);

        [self.rows, padded_len]
    }

    /// Shape without padding.
    pub fn shape_without_padding(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
}

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

/// `AlignedMatrix` construction errors.
#[derive(Debug, PartialEq)]
pub enum AlignedMatrixError {
    /// The provided data has an incorrect alignment.
    IncorrectAlignment,

    /// ndarray shape error.
    Shape(ShapeError),
}

impl fmt::Display for AlignedMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::AlignedMatrixError::*;
        match *self {
            IncorrectAlignment => write!(f, "Input array has incorrect alignment"),
            Shape(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for AlignedMatrixError {
    fn description(&self) -> &str {
        use self::AlignedMatrixError::*;

        match *self {
            IncorrectAlignment => "Input array has incorrect alignment",
            Shape(ref err) => err.description(),
        }
    }
}

/// Wrapper for ndarray's `Array` that guarantees alignment on a
/// boundary that is a multiple of the size of the type `Align`.
pub struct AlignedMatrix<Align, A> {
    inner: Array<A, Ix2>,
    _phantom: PhantomData<Align>,
}

impl<Align, A> AlignedMatrix<Align, A> {
    /// Construct an `AlignedMatrix`, filled with an element.
    ///
    /// This constructor creates an `AlignedMatrix` of shape `shape`,
    /// filled with `elem`, aligned on a boundary that is a multiple
    /// of the size of `Align`.
    pub fn from_elem(layout: MatrixLayout<Align, A>, elem: A) -> Self
    where
        A: Clone + Zero,
    {
        let shape = layout.shape_with_padding().into_shape();

        let mut v = Vec::with_capacity_aligned::<Align>(shape.size());
        v.extend(iter::repeat(elem).take(shape.size()));

        AlignedMatrix {
            inner: Array::from_shape_vec(shape, v)
                .map_err(AlignedMatrixError::Shape)
                .expect("Shape mismatch"),
            _phantom: PhantomData,
        }
    }

    /// Construct an `AlignedMatrix`, filled with zeros.
    ///
    /// This constructor creates an `AlignedMatrix` of shape `shape`,
    /// filled with zeros, aligned on a boundary that is a multiple of
    /// the size of `Align`.
    pub fn zeros(layout: MatrixLayout<Align, A>) -> Self
    where
        A: Clone + Zero,
    {
        Self::from_elem(layout, A::zero())
    }
}

impl<Align, A> Deref for AlignedMatrix<Align, A> {
    type Target = Array<A, Ix2>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Align, A> DerefMut for AlignedMatrix<Align, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Align, A> fmt::Debug for AlignedMatrix<Align, A>
where
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AlignedMatrixError {{ inner: {:?}}}", self.inner)
    }
}

/// Allocate an array type with aligned memory.
trait WithCapacityAligned<T> {
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
#[derive(Clone, Copy)]
pub struct Align16 {
    _data: [u8; 16],
}

#[cfg(feature = "align32")]
#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct Align32 {
    _data: [u8; 32],
}

#[allow(dead_code)]
#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Align64 {
    _data: [u8; 64],
}

#[cfg(test)]
mod tests {
    use std::mem;

    use super::{Align64, MatrixLayout, WithCapacityAligned};

    #[test]
    fn test_array_layout_4() {
        let layout: MatrixLayout<f32, f32> = MatrixLayout::new(10, 100);
        assert_eq!(layout.shape_with_padding(), [10, 100]);
    }

    #[test]
    fn test_array_layout_64() {
        let layout: MatrixLayout<Align64, f32> = MatrixLayout::new(10, 100);
        assert_eq!(layout.shape_with_padding(), [10, 112]);

        let layout: MatrixLayout<Align64, f32> = MatrixLayout::new(10, 32);
        assert_eq!(layout.shape_with_padding(), [10, 32]);
    }

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
}
