//! Embedding matrix representations.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error};
use memmap::{Mmap, MmapOptions};
use ndarray::{Array, Array2, ArrayBase, ArrayView, ArrayView2, Data, Dimension, Ix1, Ix2};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use reductive::pq::{QuantizeVector, ReconstructVector, TrainPQ, PQ};

use crate::io::{ChunkIdentifier, MmapChunk, ReadChunk, TypeId, WriteChunk};

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
#[derive(Debug)]
pub enum Storage {
    /// In-memory `ndarray` matrix.
    NdArray(Array2<f32>),

    /// Memory-mapped matrix.
    Mmap { map: Mmap, shape: Ix2 },

    /// Quantized matrix.
    Quantized {
        quantizer: PQ<f32>,
        quantized: Array2<u8>,
    },
}

impl Storage {
    pub(crate) fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            Storage::NdArray(_) => ChunkIdentifier::NdArray,
            Storage::Mmap { .. } => ChunkIdentifier::NdArray,
            Storage::Quantized { .. } => ChunkIdentifier::QuantizedArray,
        }
    }

    pub fn dims(&self) -> usize {
        match self {
            Storage::NdArray(ref data) => data.cols(),
            Storage::Mmap { shape, .. } => shape[1],
            Storage::Quantized { quantizer, .. } => quantizer.reconstructed_len(),
        }
    }

    pub fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            Storage::NdArray { .. } | Storage::Mmap { .. } => {
                CowArray::Owned(self.view().row(idx).to_owned())
            }
            Storage::Quantized {
                quantizer,
                quantized,
            } => CowArray::Owned(quantizer.reconstruct_vector(quantized.row(idx))),
        }
    }

    /// Quantize the embedding matrix.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    ///
    /// The xorshift PRNG is used for picking the initial quantizer
    /// centroids.
    pub fn quantize<T>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
    ) -> Self
    where
        T: TrainPQ<f32>,
    {
        self.quantize_using::<T, _>(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            &mut XorShiftRng::from_entropy(),
        )
    }

    /// Quantize the embedding matrix.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    pub fn quantize_using<T, R>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        rng: &mut R,
    ) -> Self
    where
        T: TrainPQ<f32>,
        R: Rng,
    {
        match self {
            Storage::NdArray { .. } | Storage::Mmap { .. } => {
                let quantizer = T::train_pq_using(
                    n_subquantizers,
                    n_subquantizer_bits,
                    n_iterations,
                    n_attempts,
                    self.view(),
                    rng,
                );
                let quantized = quantizer.quantize_batch(self.view());
                Storage::Quantized {
                    quantizer,
                    quantized,
                }
            }
            Storage::Quantized {
                quantizer,
                quantized,
            } => Storage::Quantized {
                quantizer: quantizer.clone(),
                quantized: quantized.clone(),
            },
        }
    }

    fn read_quantized_array_chunk<R>(read: &mut R) -> Result<Storage, Error>
    where
        R: Read + Seek,
    {
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let projection = read.read_u32::<LittleEndian>()? != 0;
        let quantized_len = read.read_u32::<LittleEndian>()? as usize;
        let reconstructed_len = read.read_u32::<LittleEndian>()? as usize;
        let n_centroids = read.read_u32::<LittleEndian>()? as usize;
        let n_embeddings = read.read_u64::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix quantizer matrices."
        );

        ensure!(
            read.read_u32::<LittleEndian>()? == u8::type_id(),
            "Expected unsigned byte quantized embedding matrices."
        );

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0))?);
        read.seek(SeekFrom::Current(n_padding as i64))?;

        let projection = if projection {
            let mut projection_vec = vec![0f32; reconstructed_len * reconstructed_len];
            read.read_f32_into::<LittleEndian>(&mut projection_vec)?;
            Some(Array2::from_shape_vec(
                (reconstructed_len, reconstructed_len),
                projection_vec,
            )?)
        } else {
            None
        };

        let mut quantizers = Vec::with_capacity(quantized_len);
        for _ in 0..quantized_len {
            let mut subquantizer_vec =
                vec![0f32; n_centroids * (reconstructed_len / quantized_len)];
            read.read_f32_into::<LittleEndian>(&mut subquantizer_vec)?;
            let subquantizer = Array2::from_shape_vec(
                (n_centroids, reconstructed_len / quantized_len),
                subquantizer_vec,
            )?;
            quantizers.push(subquantizer);
        }

        let mut quantized_embeddings_vec = vec![0u8; n_embeddings * quantized_len];
        read.read_exact(&mut quantized_embeddings_vec)?;
        let quantized =
            Array2::from_shape_vec((n_embeddings, quantized_len), quantized_embeddings_vec)?;

        Ok(Storage::Quantized {
            quantizer: PQ::new(projection, quantizers),
            quantized,
        })
    }

    pub fn read_ndarray_chunk<R>(read: &mut R) -> Result<Storage, Error>
    where
        R: Read + Seek,
    {
        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let rows = read.read_u64::<LittleEndian>()? as usize;
        let cols = read.read_u32::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix for NdArray."
        );

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0))?);
        read.seek(SeekFrom::Current(n_padding as i64))?;

        let mut data = vec![0f32; rows * cols];
        read.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(Storage::NdArray(Array2::from_shape_vec(
            (rows, cols),
            data,
        )?))
    }

    pub fn view(&self) -> ArrayView2<f32> {
        match self {
            Storage::NdArray(ref data) => data.view(),
            Storage::Mmap { map, shape } => unsafe {
                ArrayView2::from_shape_ptr(*shape, map.as_ptr() as *const f32)
            },
            Storage::Quantized { .. } => unimplemented!(),
        }
    }

    fn write_ndarray_chunk<S, W>(data: ArrayBase<S, Ix2>, mut write: W) -> Result<(), Error>
    where
        S: Data<Elem = f32>,
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);
        // Chunk size: rows (8 bytes), columns (4 bytes), type id (4 bytes),
        //             padding ([0,4) bytes), matrix.
        let chunk_len =
            16 + n_padding as usize + (data.rows() * data.cols() * mem::size_of::<f32>());
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(data.rows() as u64)?;
        write.write_u32::<LittleEndian>(data.cols() as u32)?;
        write.write_u32::<LittleEndian>(f32::type_id())?;

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
        write.write(&padding)?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col)?;
            }
        }

        Ok(())
    }

    fn write_quantized_array_chunk<S, W>(
        quantizer: &PQ<f32>,
        quantized: ArrayBase<S, Ix2>,
        mut write: W,
    ) -> Result<(), Error>
    where
        S: Data<Elem = u8>,
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::QuantizedArray as u32)?;

        // projection (4 bytes), quantized_len (4 bytes), reconstuced_len (4 bytes),
        // n_centroids (4 bytes), rows (8 bytes), types (2 x 4 bytes), padding,
        // projection matrix, centroids, quantized data.
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);
        let chunk_size = 32
            + n_padding as usize
            + quantizer.projection().is_some() as usize
                * quantizer.reconstructed_len()
                * quantizer.reconstructed_len()
                * mem::size_of::<f32>()
            + quantizer.quantized_len()
                * quantizer.n_quantizer_centroids()
                * (quantizer.reconstructed_len() / quantizer.quantized_len())
                * mem::size_of::<f32>()
            + quantized.rows() * quantizer.quantized_len();
        write.write_u64::<LittleEndian>(chunk_size as u64)?;

        write.write_u32::<LittleEndian>(quantizer.projection().is_some() as u32)?;
        write.write_u32::<LittleEndian>(quantizer.quantized_len() as u32)?;
        write.write_u32::<LittleEndian>(quantizer.reconstructed_len() as u32)?;
        write.write_u32::<LittleEndian>(quantizer.n_quantizer_centroids() as u32)?;
        write.write_u64::<LittleEndian>(quantized.rows() as u64)?;

        // Reconstruction and quantized types.
        write.write_u32::<LittleEndian>(f32::type_id())?;
        write.write_u32::<LittleEndian>(u8::type_id())?;

        let padding = vec![0u8; n_padding as usize];
        write.write(&padding)?;

        // Write projection matrix.
        if let Some(projection) = quantizer.projection() {
            for row in projection.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col)?;
                }
            }
        }

        // Write subquantizers.
        for subquantizer in quantizer.subquantizers() {
            for row in subquantizer.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col)?;
                }
            }
        }

        // Write quantized embedding matrix.
        for row in quantized.outer_iter() {
            for &col in row {
                write.write_u8(col)?;
            }
        }

        Ok(())
    }
}

impl MmapChunk for Storage {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self, Error> {
        ensure!(
            read.read_u32::<LittleEndian>()? == ChunkIdentifier::NdArray as u32,
            "invalid chunk identifier for NdArray"
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let rows = read.read_u64::<LittleEndian>()? as usize;
        let cols = read.read_u32::<LittleEndian>()? as usize;
        let shape = Ix2(rows, cols);

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix for NdArray."
        );

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0))?);
        read.seek(SeekFrom::Current(n_padding as i64))?;

        // Set up memory mapping.
        let matrix_len = shape.size() * mem::size_of::<f32>();
        let offset = read.seek(SeekFrom::Current(0))?;
        let mut mmap_opts = MmapOptions::new();
        let map = unsafe {
            mmap_opts
                .offset(offset)
                .len(matrix_len)
                .map(&read.get_ref())?
        };

        // Position the reader after the matrix.
        read.seek(SeekFrom::Current(matrix_len as i64))?;

        Ok(Storage::Mmap { map, shape })
    }
}

impl ReadChunk for Storage {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or(format_err!("Unknown chunk identifier: {}", chunk_id))?;

        match chunk_id {
            ChunkIdentifier::NdArray => Self::read_ndarray_chunk(read),
            ChunkIdentifier::QuantizedArray => Self::read_quantized_array_chunk(read),
            unknown => Err(format_err!("Not a storage chunk: {:?}", unknown)),
        }
    }
}

impl WriteChunk for Storage {
    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        match self {
            Storage::NdArray(ref data) => Self::write_ndarray_chunk(data.view(), write),
            Storage::Mmap { .. } => Self::write_ndarray_chunk(self.view(), write),
            Storage::Quantized {
                quantizer,
                quantized,
            } => Self::write_quantized_array_chunk(quantizer, quantized.view(), write),
        }
    }
}

fn padding<T>(pos: u64) -> u64 {
    let size = std::mem::size_of::<T>() as u64;
    size - (pos % size)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::{Array2, ArrayView2};
    use reductive::pq::PQ;

    use crate::io::{ReadChunk, WriteChunk};
    use crate::storage::Storage;

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    fn test_ndarray() -> Storage {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });

        Storage::NdArray(test_data)
    }

    fn test_quantized_array() -> Storage {
        let ndarray = test_ndarray();
        ndarray.quantize::<PQ<f32>>(10, 4, 5, 1)
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    fn quantizer_quantized(storage: &Storage) -> Option<(&PQ<f32>, ArrayView2<u8>)> {
        match storage {
            Storage::Quantized {
                quantizer,
                quantized,
            } => Some((quantizer, quantized.view())),
            _ => None,
        }
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
        let arr = Storage::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr.view(), check_arr.view());
    }

    #[test]
    fn quantized_array_correct_chunk_size() {
        let check_arr = test_quantized_array();
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
    fn quantized_array_read_write_roundtrip() {
        let check_arr = test_quantized_array();
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let arr = Storage::read_chunk(&mut cursor).unwrap();
        let (quantizer, quantized) = quantizer_quantized(&arr).unwrap();
        let (check_quantizer, check_quantized) = quantizer_quantized(&check_arr).unwrap();
        assert_eq!(quantizer, check_quantizer);
        assert_eq!(quantized, check_quantized);
    }
}
