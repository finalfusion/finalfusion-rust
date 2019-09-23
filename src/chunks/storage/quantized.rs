use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap::{Mmap, MmapOptions};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Dimension, IntoDimension};
use rand::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use reductive::pq::{QuantizeVector, ReconstructVector, TrainPQ, PQ};

use super::{CowArray, CowArray1, Storage, StorageView};
use crate::chunks::io::{ChunkIdentifier, MmapChunk, ReadChunk, TypeId, WriteChunk};
use crate::io::{Error, ErrorKind, Result};
use crate::util::padding;

/// Quantized embedding matrix.
pub struct QuantizedArray {
    quantizer: PQ<f32>,
    quantized_embeddings: Array2<u8>,
    norms: Option<Array1<f32>>,
}

struct PQRead {
    n_embeddings: usize,
    quantizer: PQ<f32>,
    read_norms: bool,
}

impl QuantizedArray {
    fn check_quantizer_invariants(quantized_len: usize, reconstructed_len: usize) -> Result<()> {
        if reconstructed_len % quantized_len != 0 {
            return Err(ErrorKind::Format(format!("Reconstructed embedding length ({}) not a multiple of the quantized embedding length: ({})", quantized_len, reconstructed_len)).into());
        }

        Ok(())
    }

    /// Get the quantizer.
    pub fn quantizer(&self) -> &PQ<f32> {
        &self.quantizer
    }

    fn read_product_quantizer<R>(read: &mut R) -> Result<PQRead>
    where
        R: Read + Seek,
    {
        let projection = read.read_u32::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read quantized embedding matrix projection", e)
        })? != 0;
        let read_norms = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read quantized embedding matrix norms", e))?
            != 0;
        let quantized_len = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read quantized embedding length", e))?
            as usize;
        let reconstructed_len = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read reconstructed embedding length", e))?
            as usize;
        let n_centroids = read
            .read_u32::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of subquantizers", e))?
            as usize;
        let n_embeddings = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read number of quantized embeddings", e))?
            as usize;

        Self::check_quantizer_invariants(quantized_len, reconstructed_len)?;

        // Quantized storage type.
        u8::ensure_data_type(read)?;

        // Reconstructed embedding type.
        f32::ensure_data_type(read)?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip padding", e))?;

        let projection = if projection {
            let mut projection_vec = vec![0f32; reconstructed_len * reconstructed_len];
            read.read_f32_into::<LittleEndian>(&mut projection_vec)
                .map_err(|e| ErrorKind::io_error("Cannot read projection matrix", e))?;
            Some(
                Array2::from_shape_vec((reconstructed_len, reconstructed_len), projection_vec)
                    .map_err(Error::Shape)?,
            )
        } else {
            None
        };

        let quantizer_shape = (
            quantized_len,
            n_centroids,
            reconstructed_len / quantized_len,
        )
            .into_dimension();
        let mut quantizers = vec![0f32; quantizer_shape.size()];
        read.read_f32_into::<LittleEndian>(&mut quantizers)
            .map_err(|e| ErrorKind::io_error("Cannot read subquantizer", e))?;

        Ok(PQRead {
            n_embeddings,
            quantizer: PQ::new(
                projection,
                Array::from_shape_vec(quantizer_shape, quantizers)
                    .expect("Incorrect quantizer shape"),
            ),
            read_norms,
        })
    }

    fn write_chunk<W>(
        write: &mut W,
        quantizer: &PQ<f32>,
        quantized: ArrayView2<u8>,
        norms: Option<ArrayView1<f32>>,
    ) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::QuantizedArray as u32)
            .map_err(|e| {
                ErrorKind::io_error(
                    "Cannot write quantized embedding matrix chunk identifier",
                    e,
                )
            })?;

        // projection (u32), use_norms (u32), quantized_len (u32),
        // reconstructed_len (u32), n_centroids (u32), rows (u64),
        // types (2 x u32 bytes), padding, projection matrix,
        // centroids, norms, quantized data.
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        let chunk_size = size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u64>()
            + 2 * size_of::<u32>()
            + n_padding as usize
            + quantizer.projection().is_some() as usize
                * quantizer.reconstructed_len()
                * quantizer.reconstructed_len()
                * size_of::<f32>()
            + quantizer.quantized_len()
                * quantizer.n_quantizer_centroids()
                * (quantizer.reconstructed_len() / quantizer.quantized_len())
                * size_of::<f32>()
            + norms.is_some() as usize * quantized.rows() * size_of::<f32>()
            + quantized.rows() * quantizer.quantized_len();

        write
            .write_u64::<LittleEndian>(chunk_size as u64)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write quantized embedding matrix chunk length", e)
            })?;

        write
            .write_u32::<LittleEndian>(quantizer.projection().is_some() as u32)
            .map_err(|e| {
                ErrorKind::io_error("Cannot write quantized embedding matrix projection", e)
            })?;
        write
            .write_u32::<LittleEndian>(norms.is_some() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write quantized embedding matrix norms", e))?;
        write
            .write_u32::<LittleEndian>(quantizer.quantized_len() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write quantized embedding length", e))?;
        write
            .write_u32::<LittleEndian>(quantizer.reconstructed_len() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write reconstructed embedding length", e))?;
        write
            .write_u32::<LittleEndian>(quantizer.n_quantizer_centroids() as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write number of subquantizers", e))?;
        write
            .write_u64::<LittleEndian>(quantized.rows() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write number of quantized embeddings", e))?;

        // Quantized and reconstruction types.
        write
            .write_u32::<LittleEndian>(u8::type_id())
            .map_err(|e| {
                ErrorKind::io_error("Cannot write quantized embedding type identifier", e)
            })?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| {
                ErrorKind::io_error("Cannot write reconstructed embedding type identifier", e)
            })?;

        let padding = vec![0u8; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| ErrorKind::io_error("Cannot write padding", e))?;

        // Write projection matrix.
        if let Some(projection) = quantizer.projection() {
            for row in projection.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        ErrorKind::io_error("Cannot write projection matrix component", e)
                    })?;
                }
            }
        }

        // Write subquantizers.
        for subquantizer in quantizer.subquantizers().outer_iter() {
            for row in subquantizer.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        ErrorKind::io_error("Cannot write subquantizer component", e)
                    })?;
                }
            }
        }

        // Write norms.
        if let Some(ref norms) = norms {
            for row in norms.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col).map_err(|e| {
                        ErrorKind::io_error("Cannot write norm vector component", e)
                    })?;
                }
            }
        }

        // Write quantized embedding matrix.
        for row in quantized.outer_iter() {
            for &col in row {
                write.write_u8(col).map_err(|e| {
                    ErrorKind::io_error("Cannot write quantized embedding matrix component", e)
                })?;
            }
        }

        Ok(())
    }
}

impl Storage for QuantizedArray {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        let mut reconstructed = self
            .quantizer
            .reconstruct_vector(self.quantized_embeddings.row(idx));
        if let Some(ref norms) = self.norms {
            reconstructed *= norms[idx];
        }

        CowArray::Owned(reconstructed)
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.quantized_embeddings.rows(),
            self.quantizer.reconstructed_len(),
        )
    }
}

impl ReadChunk for QuantizedArray {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::QuantizedArray)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read quantized embedding matrix chunk length", e)
        })?;

        let PQRead {
            n_embeddings,
            quantizer,
            read_norms,
        } = Self::read_product_quantizer(read)?;

        let norms = if read_norms {
            let mut norms_vec = vec![0f32; n_embeddings];
            read.read_f32_into::<LittleEndian>(&mut norms_vec)
                .map_err(|e| ErrorKind::io_error("Cannot read norms", e))?;
            Some(Array1::from_vec(norms_vec))
        } else {
            None
        };

        let mut quantized_embeddings_vec = vec![0u8; n_embeddings * quantizer.quantized_len()];
        read.read_exact(&mut quantized_embeddings_vec)
            .map_err(|e| ErrorKind::io_error("Cannot read quantized embeddings", e))?;
        let quantized_embeddings = Array2::from_shape_vec(
            (n_embeddings, quantizer.quantized_len()),
            quantized_embeddings_vec,
        )
        .map_err(Error::Shape)?;

        Ok(QuantizedArray {
            quantizer,
            quantized_embeddings,
            norms,
        })
    }
}

impl WriteChunk for QuantizedArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::QuantizedArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        Self::write_chunk(
            write,
            &self.quantizer,
            self.quantized_embeddings.view(),
            self.norms.as_ref().map(Array1::view),
        )
    }
}

/// Quantizable embedding matrix.
pub trait Quantize {
    /// Quantize the embedding matrix.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    ///
    /// The xorshift PRNG is used for picking the initial quantizer
    /// centroids.
    fn quantize<T>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
    ) -> QuantizedArray
    where
        T: TrainPQ<f32>,
    {
        self.quantize_using::<T, _>(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            normalize,
            XorShiftRng::from_entropy(),
        )
    }

    /// Quantize the embedding matrix using the provided RNG.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    fn quantize_using<T, R>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
        rng: R,
    ) -> QuantizedArray
    where
        T: TrainPQ<f32>,
        R: RngCore + SeedableRng + Send;
}

impl<S> Quantize for S
where
    S: StorageView,
{
    /// Quantize the embedding matrix.
    ///
    /// This method trains a quantizer for the embedding matrix and
    /// then quantizes the matrix using this quantizer.
    fn quantize_using<T, R>(
        &self,
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        normalize: bool,
        rng: R,
    ) -> QuantizedArray
    where
        T: TrainPQ<f32>,
        R: RngCore + SeedableRng + Send,
    {
        let (embeds, norms) = if normalize {
            let norms = self.view().outer_iter().map(|e| e.dot(&e).sqrt()).collect();
            let mut normalized = self.view().to_owned();
            for (mut embedding, &norm) in normalized.outer_iter_mut().zip(&norms) {
                embedding /= norm;
            }
            (CowArray::Owned(normalized), Some(norms))
        } else {
            (CowArray::Borrowed(self.view()), None)
        };

        let quantizer = T::train_pq_using(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            embeds.as_view(),
            rng,
        );

        let quantized_embeddings = quantizer.quantize_batch(embeds.as_view());

        QuantizedArray {
            quantizer,
            quantized_embeddings,
            norms,
        }
    }
}

/// Memory-mapped quantized embedding matrix.
pub struct MmapQuantizedArray {
    quantizer: PQ<f32>,
    quantized_embeddings: Mmap,
    norms: Option<Mmap>,
}

impl MmapQuantizedArray {
    unsafe fn norms(&self) -> Option<ArrayView1<f32>> {
        let n_embeddings = self.shape().0;

        #[allow(clippy::cast_ptr_alignment)]
        self.norms.as_ref().map(|norms|
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
                ArrayView1::from_shape_ptr((n_embeddings,), norms.as_ptr() as *const f32))
    }

    unsafe fn quantized_embeddings(&self) -> ArrayView2<u8> {
        let n_embeddings = self.shape().0;

        ArrayView2::from_shape_ptr(
            (n_embeddings, self.quantizer.quantized_len()),
            self.quantized_embeddings.as_ptr(),
        )
    }
}

impl MmapQuantizedArray {
    fn mmap_norms(read: &mut BufReader<File>, n_embeddings: usize) -> Result<Mmap> {
        let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error(
                "Cannot get file position for memory mapping the norms vector",
                e,
            )
        })?;

        let norms_len = n_embeddings * size_of::<f32>();
        let mut mmap_opts = MmapOptions::new();
        let norms = unsafe {
            mmap_opts
                .offset(offset)
                .len(norms_len)
                .map(&read.get_ref())
                .map_err(|e| ErrorKind::io_error("Cannot memory map norms vector", e))?
        };

        // Position the reader after the vector.
        read.seek(SeekFrom::Current(norms_len as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip norms vector", e))?;

        Ok(norms)
    }

    fn mmap_quantized_embeddings(
        read: &mut BufReader<File>,
        n_embeddings: usize,
        quantized_len: usize,
    ) -> Result<Mmap> {
        let offset = read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error(
                "Cannot get file position for memory mapping embedding matrix",
                e,
            )
        })?;
        let matrix_len = n_embeddings * quantized_len;
        let mut mmap_opts = MmapOptions::new();
        let quantized = unsafe {
            mmap_opts
                .offset(offset)
                .len(matrix_len)
                .map(&read.get_ref())
                .map_err(|e| {
                    ErrorKind::io_error("Cannot memory map quantized embedding matrix", e)
                })?
        };

        // Position the reader after the matrix.
        read.seek(SeekFrom::Current(matrix_len as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip quantized embedding matrix", e))?;

        Ok(quantized)
    }

    /// Get the quantizer.
    pub fn quantizer(&self) -> &PQ<f32> {
        &self.quantizer
    }
}

impl Storage for MmapQuantizedArray {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        let quantized = unsafe { self.quantized_embeddings() };

        let mut reconstructed = self.quantizer.reconstruct_vector(quantized.row(idx));
        if let Some(norms) = unsafe { self.norms() } {
            reconstructed *= norms[idx];
        }

        CowArray::Owned(reconstructed)
    }

    fn shape(&self) -> (usize, usize) {
        (
            self.quantized_embeddings.len() / self.quantizer.quantized_len(),
            self.quantizer.reconstructed_len(),
        )
    }
}

impl MmapChunk for MmapQuantizedArray {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self> {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::QuantizedArray)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>().map_err(|e| {
            ErrorKind::io_error("Cannot read quantized embedding matrix chunk length", e)
        })?;

        let PQRead {
            n_embeddings,
            quantizer,
            read_norms,
        } = QuantizedArray::read_product_quantizer(read)?;

        let norms = if read_norms {
            Some(Self::mmap_norms(read, n_embeddings)?)
        } else {
            None
        };

        let quantized_embeddings =
            Self::mmap_quantized_embeddings(read, n_embeddings, quantizer.quantized_len())?;

        Ok(MmapQuantizedArray {
            quantizer,
            quantized_embeddings,
            norms,
        })
    }
}

impl WriteChunk for MmapQuantizedArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::QuantizedArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        QuantizedArray::write_chunk(
            write,
            &self.quantizer,
            unsafe { self.quantized_embeddings() },
            unsafe { self.norms() },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::Array2;
    use reductive::pq::PQ;

    use crate::chunks::io::{MmapChunk, ReadChunk, WriteChunk};
    use crate::chunks::storage::{MmapQuantizedArray, NdArray, Quantize, QuantizedArray, Storage};

    const N_ROWS: usize = 100;
    const N_COLS: usize = 100;

    fn test_ndarray() -> NdArray {
        let test_data = Array2::from_shape_fn((N_ROWS, N_COLS), |(r, c)| {
            r as f32 * N_COLS as f32 + c as f32
        });

        NdArray(test_data)
    }

    fn test_quantized_array(norms: bool) -> QuantizedArray {
        let ndarray = test_ndarray();
        ndarray.quantize::<PQ<f32>>(10, 4, 5, 1, norms)
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    // Compare storage for which Eq is not implemented.
    fn storage_eq(arr: &impl Storage, check_arr: &impl Storage) {
        assert_eq!(arr.shape(), check_arr.shape());
        for idx in 0..check_arr.shape().0 {
            assert_eq!(
                arr.embedding(idx).as_view(),
                check_arr.embedding(idx).as_view()
            );
        }
    }

    #[test]
    fn quantized_array_correct_chunk_size() {
        let check_arr = test_quantized_array(false);
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
    fn quantized_array_norms_correct_chunk_size() {
        let check_arr = test_quantized_array(true);
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
        let check_arr = test_quantized_array(true);
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let arr = QuantizedArray::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr.quantizer, check_arr.quantizer);
        assert_eq!(arr.quantized_embeddings, check_arr.quantized_embeddings);
    }

    #[test]
    fn mmap_quantized_array() {
        let mut storage_read =
            BufReader::new(File::open("testdata/quantized_storage.bin").unwrap());
        let check_arr = QuantizedArray::read_chunk(&mut storage_read).unwrap();

        // Memory map matrix.
        storage_read.seek(SeekFrom::Start(0)).unwrap();
        let arr = MmapQuantizedArray::mmap_chunk(&mut storage_read).unwrap();

        // Check
        storage_eq(&arr, &check_arr);
    }

    #[test]
    fn write_mmap_quantized_array() {
        // Memory map matrix.
        let mut storage_read =
            BufReader::new(File::open("testdata/quantized_storage.bin").unwrap());
        let check_arr = MmapQuantizedArray::mmap_chunk(&mut storage_read).unwrap();

        // Write matrix
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();

        // Read using non-mmap'ed reader.
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let arr = QuantizedArray::read_chunk(&mut cursor).unwrap();

        // Check
        storage_eq(&arr, &check_arr);
    }
}
