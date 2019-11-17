use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::iter::FromIterator;
use std::mem::size_of;

#[cfg(target_endian = "big")]
use byteorder::ByteOrder;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap::{Mmap, MmapOptions};
use ndarray::{s, Array1, Array2, ArrayView2, ArrayViewMut2, Axis, CowArray, Dimension, Ix1, Ix2};
use ordered_float::OrderedFloat;

use super::{Storage, StoragePrune, StorageView, StorageViewMut, StorageWrap};
use crate::chunks::io::{ChunkIdentifier, MmapChunk, ReadChunk, TypeId, WriteChunk};
use crate::io::{Error, ErrorKind, Result};
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

impl StorageViewMut for NdArray {
    fn view_mut(&mut self) -> ArrayViewMut2<f32> {
        self.inner.view_mut()
    }
}

impl MmapChunk for MmapArray {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read embedding matrix chunk length", e))?;

        let rows = read.read_u64::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read number of rows of the embedding matrix", e)
        })? as usize;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read number of columns of the embedding matrix", e)
        })? as usize;
        let shape = Ix2(rows, cols);

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read)?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip padding", e))?;

        // Set up memory mapping.
        let matrix_len = shape.size() * size_of::<f32>();
        let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error(
                "Cannot get file position for memory mapping embedding matrix",
                e,
            )
        })?;
        let mut mmap_opts = MmapOptions::new();
        let map = unsafe {
            mmap_opts
                .offset(offset)
                .len(matrix_len)
                .map(&read.get_ref())
                .map_err(|e| ErrorKind::io_error("Cannot memory map embedding matrix", e))?
        };

        // Position the reader after the matrix.
        read.seek(SeekFrom::Current(matrix_len as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip embedding matrix", e))?;

        Ok(MmapArray { map, shape })
    }
}

#[cfg(target_endian = "little")]
impl WriteChunk for MmapArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        NdArray::write_ndarray_chunk(self.view(), write)
    }
}

/// In-memory `ndarray` matrix.
#[derive(Clone, Debug)]
pub struct NdArray {
    inner: Array2<f32>,
}

impl NdArray {
    pub fn new(arr: Array2<f32>) -> Self {
        NdArray { inner: arr }
    }

    fn write_ndarray_chunk<W>(data: ArrayView2<f32>, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write embedding matrix chunk identifier", e)
            })?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        // Chunk size: rows (u64), columns (u32), type id (u32),
        //             padding ([0,4) bytes), matrix.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + n_padding as usize
            + (data.nrows() * data.ncols() * size_of::<f32>());
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write embedding matrix chunk length", e))?;
        write
            .write_u64::<LittleEndian>(data.nrows() as u64)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write number of rows of the embedding matrix", e)
            })?;
        write
            .write_u32::<LittleEndian>(data.ncols() as u32)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write number of columns of the embedding matrix", e)
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| ErrorKind::io_error("Cannot write embedding matrix type identifier", e))?;

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
            .map_err(|e| ErrorKind::io_error("Cannot write padding", e))?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col).map_err(|e| {
                    ErrorKind::io_error("Cannot write embedding matrix component", e)
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

impl Storage for NdArray {
    fn embedding(&self, idx: usize) -> CowArray<f32, Ix1> {
        CowArray::from(self.inner.row(idx))
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

impl ReadChunk for NdArray {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdArray)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read embedding matrix chunk length", e))?;

        let rows = read.read_u64::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read number of rows of the embedding matrix", e)
        })? as usize;
        let cols = read.read_u32::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read number of columns of the embedding matrix", e)
        })? as usize;

        // The components of the embedding matrix should be of type f32.
        f32::ensure_data_type(read)?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip padding", e))?;

        let mut data = vec![0f32; rows * cols];
        read.read_f32_into::<LittleEndian>(&mut data)
            .map_err(|e| ErrorKind::io_error("Cannot read embedding matrix", e))?;

        Ok(NdArray {
            inner: Array2::from_shape_vec((rows, cols), data).map_err(Error::Shape)?,
        })
    }
}

impl WriteChunk for NdArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        Self::write_ndarray_chunk(self.inner.view(), write)
    }
}

impl StoragePrune for NdArray {
    fn prune_storage(&self, toss_indices: &[usize]) -> StorageWrap {
        let toss_indices: HashSet<usize> = HashSet::from_iter(toss_indices.iter().cloned());
        let mut keep_indices_all = Vec::new();
        for idx in 0..self.shape().1 - 1 {
            if !toss_indices.contains(&idx) {
                keep_indices_all.push(idx);
            }
        }
        NdArray::new(self.inner.select(Axis(0), &keep_indices_all).to_owned()).into()
    }

    fn most_similar(
        &self,
        keep_indices: &[usize],
        toss_indices: &[usize],
        batch_size: usize,
    ) -> Array1<usize> {
        let toss = toss_indices.len();
        let mut remap_indices = Array1::zeros(toss);
        let keep_embeds = self.inner.select(Axis(0), keep_indices);
        let keep_embeds_t = &keep_embeds.t();
        let toss_embeds = self.inner.select(Axis(0), toss_indices);
        for n in (0..toss).step_by(batch_size) {
            let mut offset = n + batch_size;
            if offset > toss {
                offset = toss;
            }
            #[allow(clippy::deref_addrof)]
            let batch = toss_embeds.slice(s![n..offset, ..]);
            let similarity_scores = batch.dot(keep_embeds_t);
            for (i, row) in similarity_scores.axis_iter(Axis(0)).enumerate() {
                let dist = row
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| OrderedFloat(v))
                    .unwrap()
                    .0;
                remap_indices[n + i] = dist;
            }
        }
        remap_indices
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::{arr1, arr2, Array2, Axis};

    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::storage::{NdArray, Storage, StoragePrune, StorageView};

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    fn test_ndarray() -> NdArray {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });

        NdArray::new(test_data)
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[test]
    fn ndarray_correct_chunk_size() {
        let check_arr = test_ndarray();
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
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

    fn test_ndarray_for_pruning() -> NdArray {
        let test_storage: Array2<f32> = arr2(&[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0, 0.0],
        ]);
        NdArray::new(test_storage)
    }

    #[test]
    fn test_most_similar() {
        let storage = test_ndarray_for_pruning();
        let keep_indices = &[0, 1];
        let toss_indices = &[2, 3];
        let test_remap_indices = storage.most_similar(keep_indices, toss_indices, 1);
        assert_eq!(arr1(&[1, 0]), test_remap_indices);
    }

    #[test]
    fn test_prune_storage() {
        let pruned_storage: Array2<f32> =
            arr2(&[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]);
        let storage = test_ndarray_for_pruning();
        let toss_indices = &[2, 3];
        let test_pruned_storage = storage.prune_storage(toss_indices);
        assert_eq!(pruned_storage.dim(), test_pruned_storage.shape());
        for (idx, row) in pruned_storage.axis_iter(Axis(0)).enumerate() {
            assert_eq!(row, test_pruned_storage.embedding(idx));
        }
    }
}
