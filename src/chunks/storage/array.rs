use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, CowArray, Ix1};

use super::{sealed::CloneFromMapping, Storage, StorageView, StorageViewMut};
use crate::chunks::io::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};
use crate::error::{Error, Result};
use crate::util::padding;

#[cfg(feature = "memmap")]
mod mmap {
    use std::convert::TryInto;
    use std::fs::File;
    #[cfg(target_endian = "little")]
    use std::io::Write;
    use std::io::{BufReader, Seek, SeekFrom};
    use std::mem::size_of;

    use crate::chunks::io::MmapChunk;
    #[cfg(target_endian = "big")]
    use byteorder::ByteOrder;
    use byteorder::{LittleEndian, ReadBytesExt};
    use memmap2::{Mmap, MmapOptions};
    use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix1};
    use ndarray::{Dimension, Ix2};

    use super::NdArray;
    #[cfg(target_endian = "little")]
    use crate::chunks::io::WriteChunk;
    use crate::chunks::io::{ChunkIdentifier, TypeId};
    #[cfg(target_endian = "little")]
    use crate::chunks::storage::StorageView;
    use crate::chunks::storage::{sealed::CloneFromMapping, Storage};
    use crate::error::{Error, Result};
    use crate::util::padding;

    /// Memory-mapped matrix.
    #[derive(Debug)]
    pub struct MmapArray {
        map: Mmap,
        shape: Ix2,
    }

    impl Storage for MmapArray {
        fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
            #[allow(clippy::cast_ptr_alignment,unused_mut)]
        let mut embedding =
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
            unsafe { ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32) }
                .row(idx)
                .to_owned();

            #[cfg(target_endian = "big")]
            LittleEndian::from_slice_f32(
                embedding
                    .as_slice_mut()
                    .expect("Cannot borrow vector as mutable slice"),
            );

            CowArray::from(embedding)
        }

        #[allow(clippy::let_and_return)]
        fn embeddings(&self, indices: &[usize]) -> Array2<f32> {
            #[allow(clippy::cast_ptr_alignment,unused_mut)]
            let embeddings =
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
		unsafe { ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32) };

            #[allow(unused_mut)]
            let mut selected_embeddings = embeddings.select(Axis(0), indices);

            #[cfg(target_endian = "big")]
            LittleEndian::from_slice_f32(
                selected_embeddings
                    .as_slice_mut()
                    .expect("Cannot borrow matrix as mutable slice"),
            );

            selected_embeddings
        }

        fn shape(&self) -> (usize, usize) {
            self.shape.into_pattern()
        }
    }

    #[cfg(target_endian = "little")]
    impl StorageView for MmapArray {
        fn view(&self) -> ArrayView2<f32> {
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
            #[allow(clippy::cast_ptr_alignment)]
            unsafe {
                ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32)
            }
        }
    }

    impl CloneFromMapping for MmapArray {
        type Result = NdArray;

        fn clone_from_mapping(&self, mapping: &[usize]) -> Self::Result {
            NdArray::new(self.embeddings(mapping))
        }
    }

    impl MmapChunk for MmapArray {
        fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
            ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;

            // Read and discard chunk length.
            read.read_u64::<LittleEndian>()
                .map_err(|e| Error::read_error("Cannot read embedding matrix chunk length", e))?;

            let rows = read
                .read_u64::<LittleEndian>()
                .map_err(|e| {
                    Error::read_error("Cannot read number of rows of the embedding matrix", e)
                })?
                .try_into()
                .map_err(|_| Error::Overflow)?;
            let cols = read.read_u32::<LittleEndian>().map_err(|e| {
                Error::read_error("Cannot read number of columns of the embedding matrix", e)
            })? as usize;
            let shape = Ix2(rows, cols);

            // The components of the embedding matrix should be of type f32.
            f32::ensure_data_type(read)?;

            let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
                Error::read_error("Cannot get file position for computing padding", e)
            })?);
            read.seek(SeekFrom::Current(n_padding as i64))
                .map_err(|e| Error::read_error("Cannot skip padding", e))?;

            // Set up memory mapping.
            let matrix_len = shape.size() * size_of::<f32>();
            let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
                Error::read_error(
                    "Cannot get file position for memory mapping embedding matrix",
                    e,
                )
            })?;
            let mut mmap_opts = MmapOptions::new();
            let map = unsafe {
                mmap_opts
                    .offset(offset)
                    .len(matrix_len)
                    .map(&*read.get_ref())
                    .map_err(|e| Error::read_error("Cannot memory map embedding matrix", e))?
            };

            // Position the reader after the matrix.
            read.seek(SeekFrom::Current(matrix_len as i64))
                .map_err(|e| Error::read_error("Cannot skip embedding matrix", e))?;

            Ok(MmapArray { map, shape })
        }
    }

    #[cfg(target_endian = "little")]
    impl WriteChunk for MmapArray {
        fn chunk_identifier(&self) -> ChunkIdentifier {
            ChunkIdentifier::NdArray
        }

        fn chunk_len(&self, offset: u64) -> u64 {
            NdArray::chunk_len(self.view(), offset)
        }

        fn write_chunk<W>(&self, write: &mut W) -> Result<()>
        where
            W: Write + Seek,
        {
            NdArray::write_ndarray_chunk(self.view(), write)
        }
    }
}

#[cfg(feature = "memmap")]
pub use mmap::MmapArray;

/// In-memory `ndarray` matrix.
#[derive(Clone, Debug)]
pub struct NdArray {
    inner: Array2<f32>,
}

impl NdArray {
    pub fn new(arr: Array2<f32>) -> Self {
        NdArray { inner: arr }
    }

    fn chunk_len(data: ArrayView2<f32>, offset: u64) -> u64 {
        let n_padding = padding::<f32>(offset + mem::size_of::<u32>() as u64);

        // Chunk identifier (u32) + chunk len (u64) + rows (u64) + cols (u32) + type id (u32) + padding + matrix.
        (mem::size_of::<u32>()
            + mem::size_of::<u64>()
            + mem::size_of::<u64>()
            + mem::size_of::<u32>()
            + mem::size_of::<u32>()
            + data.len() * mem::size_of::<f32>()) as u64
            + n_padding
    }

    fn write_ndarray_chunk<W>(data: ArrayView2<f32>, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)
            .map_err(|e| Error::write_error("Cannot write embedding matrix chunk identifier", e))?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            Error::write_error("Cannot get file position for computing padding", e)
        })?);

        let remaining_chunk_len = Self::chunk_len(
            data.view(),
            write.seek(SeekFrom::Current(0)).map_err(|e| {
                Error::read_error("Cannot get file position for computing padding", e)
            })?,
        ) - (size_of::<u32>() + size_of::<u64>()) as u64;

        write
            .write_u64::<LittleEndian>(remaining_chunk_len)
            .map_err(|e| Error::write_error("Cannot write embedding matrix chunk length", e))?;
        write
            .write_u64::<LittleEndian>(data.nrows() as u64)
            .map_err(|e| {
                Error::write_error("Cannot write number of rows of the embedding matrix", e)
            })?;
        write
            .write_u32::<LittleEndian>(data.ncols() as u32)
            .map_err(|e| {
                Error::write_error("Cannot write number of columns of the embedding matrix", e)
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| Error::write_error("Cannot write embedding matrix type identifier", e))?;

        // Write padding, such that the embedding matrix starts on at
        // a multiple of the size of f32 (4 bytes). This is necessary
        // for memory mapping a matrix. Interpreting the raw u8 data
        // as a proper f32 array requires that the data is aligned in
        // memory. However, we cannot always memory map the starting
        // offset of the matrix directly, since mmap(2) requires a
        // file offset that is page-aligned. Since the page size is
        // always a larger power of 2 (e.g. 2^12), which is divisible
        // by 4, the offset of the matrix with regards to the page
        // boundary is also a multiple of 4.

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| Error::write_error("Cannot write padding", e))?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col).map_err(|e| {
                    Error::write_error("Cannot write embedding matrix component", e)
                })?;
            }
        }

        Ok(())
    }
}

impl From<Array2<f32>> for NdArray {
    fn from(arr: Array2<f32>) -> Self {
        NdArray::new(arr)
    }
}

impl From<NdArray> for Array2<f32> {
    fn from(arr: NdArray) -> Self {
        arr.inner
    }
}

impl Storage for NdArray {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        CowArray::from(self.inner.row(idx))
    }

    fn embeddings(&self, indices: &[usize]) -> Array2<f32> {
        self.inner.select(Axis(0), indices)
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.dim()
    }
}

impl StorageView for NdArray {
    fn view(&self) -> ArrayView2<f32> {
        self.inner.view()
    }
}

impl StorageViewMut for NdArray {
    fn view_mut(&mut self) -> ArrayViewMut2<f32> {
        self.inner.view_mut()
    }
}

impl CloneFromMapping for NdArray {
    type Result = NdArray;

    fn clone_from_mapping(&self, mapping: &[usize]) -> Self::Result {
        NdArray::new(self.embeddings(mapping))
    }
}

impl ReadChunk for NdArray {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read embedding matrix chunk length", e))?;

        let rows = read
            .read_u64::<LittleEndian>()
            .map_err(|e| {
                Error::read_error("Cannot read number of rows of the embedding matrix", e)
            })?
            .try_into()
            .map_err(|_| Error::Overflow)?;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            Error::read_error("Cannot read number of columns of the embedding matrix", e)
        })? as usize;

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read)?;

        let n_padding =
            padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
                Error::read_error("Cannot get file position for computing padding", e)
            })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| Error::read_error("Cannot skip padding", e))?;

        let mut data = Array2::zeros((rows, cols));
        read.read_f32_into::<LittleEndian>(data.as_slice_mut().unwrap())
            .map_err(|e| Error::read_error("Cannot read embedding matrix", e))?;

        Ok(NdArray { inner: data })
    }
}

impl WriteChunk for NdArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdArray
    }

    fn chunk_len(&self, offset: u64) -> u64 {
        Self::chunk_len(self.inner.view(), offset)
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        Self::write_ndarray_chunk(self.inner.view(), write)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek, SeekFrom};

    use ndarray::Array2;

    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::storage::{NdArray, Storage, StorageView};
    use crate::storage::tests::test_storage_chunk_len;

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    fn test_ndarray() -> NdArray {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });

        NdArray::new(test_data)
    }

    #[test]
    fn embeddings_returns_expected_embeddings() {
        const CHECK_INDICES: &[usize] = &[0, 50, 99, 0];

        let check_arr = test_ndarray();

        let embeddings = check_arr.embeddings(CHECK_INDICES);

        for (embedding, &idx) in embeddings.outer_iter().zip(CHECK_INDICES) {
            assert_eq!(embedding, check_arr.embedding(idx));
        }
    }

    #[test]
    fn ndarray_correct_chunk_size() {
        test_storage_chunk_len(test_ndarray().into());
    }

    #[test]
    fn ndarray_write_read_roundtrip() {
        let check_arr = test_ndarray();
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let arr = NdArray::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr.view(), check_arr.view());
    }

    #[cfg(feature = "memmap")]
    mod mmap {
        use std::io::{BufReader, BufWriter, Seek, SeekFrom};

        use tempfile::tempfile;

        use super::test_ndarray;
        use crate::chunks::io::{MmapChunk, WriteChunk};
        use crate::chunks::storage::{MmapArray, Storage};

        fn test_mmap_array() -> MmapArray {
            let array = test_ndarray();
            let mut tmp = tempfile().unwrap();
            array.write_chunk(&mut BufWriter::new(&mut tmp)).unwrap();
            tmp.seek(SeekFrom::Start(0)).unwrap();
            MmapArray::mmap_chunk(&mut BufReader::new(tmp)).unwrap()
        }

        #[test]
        fn embeddings_returns_expected_embeddings() {
            const CHECK_INDICES: &[usize] = &[0, 50, 99, 0];

            let check_arr = test_mmap_array();

            let embeddings = check_arr.embeddings(CHECK_INDICES);

            for (embedding, &idx) in embeddings.outer_iter().zip(CHECK_INDICES) {
                assert_eq!(embedding, check_arr.embedding(idx));
            }
        }
    }
}
