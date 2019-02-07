//! Embedding matrix representations.

use std::io::{Read, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, Error};
use ndarray::{Array, Array2, ArrayView, ArrayView2, Axis, Dimension, Ix1};

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
#[derive(Debug, PartialEq)]
pub enum Storage {
    NdArray(Array2<f32>),
}

impl Storage {
    pub(crate) fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            Storage::NdArray(_) => ChunkIdentifier::NdArray,
        }
    }

    pub fn dims(&self) -> usize {
        match self {
            Storage::NdArray(ref data) => data.cols(),
        }
    }

    pub fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            Storage::NdArray(ref data) => CowArray::Borrowed(data.index_axis(Axis(0), idx)),
        }
    }

    pub fn view(&self) -> ArrayView2<f32> {
        match self {
            Storage::NdArray(ref data) => data.view(),
        }
    }
}

impl ReadChunk for Storage {
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

        Ok(Storage::NdArray(Array2::from_shape_vec(
            (rows, cols),
            data,
        )?))
    }
}

impl WriteChunk for Storage {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error> {
        match self {
            Storage::NdArray(ref data) => {
                // n_rows: 8 bytes, n_cols: 4 bytes, type_id: 4, matrix
                let chunk_len = 16 + (data.rows() * data.cols() * mem::size_of::<f32>());

                write.write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)?;
                write.write_u64::<LittleEndian>(chunk_len as u64)?;
                write.write_u64::<LittleEndian>(data.rows() as u64)?;
                write.write_u32::<LittleEndian>(data.cols() as u32)?;
                write.write_u32::<LittleEndian>(f32::type_id())?;

                for row in data.outer_iter() {
                    for col in row.iter() {
                        write.write_f32::<LittleEndian>(*col)?;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ndarray::Array2;

    use crate::io::{ReadChunk, WriteChunk};
    use crate::storage::Storage;

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    #[test]
    fn ndarray_write_read_roundtrip() {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });
        let check_arr = Storage::NdArray(test_data);
        let mut serialized = Vec::new();
        check_arr.write_chunk(&mut serialized).unwrap();
        let mut cursor = Cursor::new(serialized);
        let arr = Storage::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr, check_arr);
    }
}
