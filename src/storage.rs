//! Embedding matrix representations.

use std::io::{Read, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, Error};
use ndarray::{Array, Array2, ArrayView, Axis, Dimension, Ix1};

use crate::io::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};

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

impl ReadChunk for NdArray {
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error> {
        ensure!(
            read.read_u32::<LittleEndian>()? == ChunkIdentifier::NdArray as u32,
            "invalid chunk identifier for NdArray"
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let rows = read.read_u64::<LittleEndian>()? as usize;
        let cols = read.read_u32::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix for NdArray."
        );

        let mut data = vec![0f32; rows * cols];
        read.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(NdArray(Array2::from_shape_vec((rows, cols), data)?))
    }
}

impl WriteChunk for NdArray {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error> {
        // n_rows: 8 bytes, n_cols: 4 bytes, type_id: 4, matrix
        let chunk_len = 16 + (self.0.rows() * self.0.cols() * mem::size_of::<f32>());

        write.write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)?;
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.0.rows() as u64)?;
        write.write_u32::<LittleEndian>(self.0.cols() as u32)?;
        write.write_u32::<LittleEndian>(f32::type_id())?;

        for row in self.0.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col)?;
            }
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ndarray::Array2;

    use super::NdArray;
    use crate::io::{ReadChunk, WriteChunk};

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    #[test]
    fn ndarray_write_read_roundtrip() {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });
        let check_arr = NdArray(test_data);
        let mut serialized = Vec::new();
        check_arr.write_chunk(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let arr = NdArray::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr, check_arr);
    }
}
