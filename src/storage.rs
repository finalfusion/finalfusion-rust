//! Embedding matrix representations.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error};
use memmap::{Mmap, MmapOptions};
use ndarray::{Array, Array1, Array2, ArrayView, ArrayView2, ArrayViewMut2, Dimension, Ix1, Ix2};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use reductive::pq::{QuantizeVector, ReconstructVector, TrainPQ, PQ};

use crate::io::private::{ChunkIdentifier, MmapChunk, ReadChunk, TypeId, WriteChunk};
use crate::util::padding;

/// Copy-on-write wrapper for `Array`/`ArrayView`.
///
/// The `CowArray` type stores an owned array or an array view. In
/// both cases a view (`as_view`) or an owned array (`into_owned`) can
/// be obtained. If the wrapped array is a view, retrieving an owned
/// array will copy the underlying data.
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

/// Memory-mapped matrix.
pub struct MmapArray {
    map: Mmap,
    shape: Ix2,
}

impl MmapChunk for MmapArray {
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
        let matrix_len = shape.size() * size_of::<f32>();
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

        Ok(MmapArray { map, shape })
    }
}

impl WriteChunk for MmapArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        NdArray::write_ndarray_chunk(self.view(), write)
    }
}

/// In-memory `ndarray` matrix.
#[derive(Debug)]
pub struct NdArray(pub Array2<f32>);

impl NdArray {
    fn write_ndarray_chunk<W>(data: ArrayView2<f32>, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::NdArray as u32)?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);
        // Chunk size: rows (u64), columns (u32), type id (u32),
        //             padding ([0,4) bytes), matrix.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + size_of::<u32>()
            + n_padding as usize
            + (data.rows() * data.cols() * size_of::<f32>());
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
        write.write_all(&padding)?;

        for row in data.outer_iter() {
            for col in row.iter() {
                write.write_f32::<LittleEndian>(*col)?;
            }
        }

        Ok(())
    }
}

impl ReadChunk for NdArray {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;
        ensure!(
            chunk_id == ChunkIdentifier::NdArray,
            "Cannot read chunk {:?} as NdArray",
            chunk_id
        );

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

        Ok(NdArray(Array2::from_shape_vec((rows, cols), data)?))
    }
}

impl WriteChunk for NdArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        Self::write_ndarray_chunk(self.0.view(), write)
    }
}

/// Quantized embedding matrix.
pub struct QuantizedArray {
    quantizer: PQ<f32>,
    quantized: Array2<u8>,
    norms: Option<Array1<f32>>,
}

impl ReadChunk for QuantizedArray {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;
        ensure!(
            chunk_id == ChunkIdentifier::QuantizedArray,
            "Cannot read chunk {:?} as QuantizedArray",
            chunk_id
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let projection = read.read_u32::<LittleEndian>()? != 0;
        let read_norms = read.read_u32::<LittleEndian>()? != 0;
        let quantized_len = read.read_u32::<LittleEndian>()? as usize;
        let reconstructed_len = read.read_u32::<LittleEndian>()? as usize;
        let n_centroids = read.read_u32::<LittleEndian>()? as usize;
        let n_embeddings = read.read_u64::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == u8::type_id(),
            "Expected unsigned byte quantized embedding matrices."
        );

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix quantizer matrices."
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

        let norms = if read_norms {
            let mut norms_vec = vec![0f32; n_embeddings];
            read.read_f32_into::<LittleEndian>(&mut norms_vec)?;
            Some(Array1::from_vec(norms_vec))
        } else {
            None
        };

        let mut quantized_embeddings_vec = vec![0u8; n_embeddings * quantized_len];
        read.read_exact(&mut quantized_embeddings_vec)?;
        let quantized =
            Array2::from_shape_vec((n_embeddings, quantized_len), quantized_embeddings_vec)?;

        Ok(QuantizedArray {
            quantizer: PQ::new(projection, quantizers),
            quantized,
            norms,
        })
    }
}

impl WriteChunk for QuantizedArray {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::QuantizedArray
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::QuantizedArray as u32)?;

        // projection (u32), use_norms (u32), quantized_len (u32),
        // reconstructed_len (u32), n_centroids (u32), rows (u64),
        // types (2 x u32 bytes), padding, projection matrix,
        // centroids, norms, quantized data.
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);
        let chunk_size = size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u32>()
            + size_of::<u64>()
            + 2 * size_of::<u32>()
            + n_padding as usize
            + self.quantizer.projection().is_some() as usize
                * self.quantizer.reconstructed_len()
                * self.quantizer.reconstructed_len()
                * size_of::<f32>()
            + self.quantizer.quantized_len()
                * self.quantizer.n_quantizer_centroids()
                * (self.quantizer.reconstructed_len() / self.quantizer.quantized_len())
                * size_of::<f32>()
            + self.norms.is_some() as usize * self.quantized.rows() * size_of::<f32>()
            + self.quantized.rows() * self.quantizer.quantized_len();

        write.write_u64::<LittleEndian>(chunk_size as u64)?;

        write.write_u32::<LittleEndian>(self.quantizer.projection().is_some() as u32)?;
        write.write_u32::<LittleEndian>(self.norms.is_some() as u32)?;
        write.write_u32::<LittleEndian>(self.quantizer.quantized_len() as u32)?;
        write.write_u32::<LittleEndian>(self.quantizer.reconstructed_len() as u32)?;
        write.write_u32::<LittleEndian>(self.quantizer.n_quantizer_centroids() as u32)?;
        write.write_u64::<LittleEndian>(self.quantized.rows() as u64)?;

        // Quantized and reconstruction types.
        write.write_u32::<LittleEndian>(u8::type_id())?;
        write.write_u32::<LittleEndian>(f32::type_id())?;

        let padding = vec![0u8; n_padding as usize];
        write.write_all(&padding)?;

        // Write projection matrix.
        if let Some(projection) = self.quantizer.projection() {
            for row in projection.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col)?;
                }
            }
        }

        // Write subquantizers.
        for subquantizer in self.quantizer.subquantizers() {
            for row in subquantizer.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col)?;
                }
            }
        }

        // Write norms.
        if let Some(ref norms) = self.norms {
            for row in norms.outer_iter() {
                for &col in row {
                    write.write_f32::<LittleEndian>(col)?;
                }
            }
        }

        // Write quantized embedding matrix.
        for row in self.quantized.outer_iter() {
            for &col in row {
                write.write_u8(col)?;
            }
        }

        Ok(())
    }
}

/// Storage types wrapper.
///
/// This crate makes it possible to create fine-grained embedding
/// types, such as `Embeddings<SimpleVocab, NdArray>` or
/// `Embeddings<SubwordVocab, QuantizedArray>`. However, in some cases
/// it is more pleasant to have a single type that covers all
/// vocabulary and storage types. `VocabWrap` and `StorageWrap` wrap
/// all the vocabularies and storage types known to this crate such
/// that the type `Embeddings<VocabWrap, StorageWrap>` covers all
/// variations.
pub enum StorageWrap {
    NdArray(NdArray),
    QuantizedArray(QuantizedArray),
    MmapArray(MmapArray),
}

impl From<MmapArray> for StorageWrap {
    fn from(s: MmapArray) -> Self {
        StorageWrap::MmapArray(s)
    }
}

impl From<NdArray> for StorageWrap {
    fn from(s: NdArray) -> Self {
        StorageWrap::NdArray(s)
    }
}

impl From<QuantizedArray> for StorageWrap {
    fn from(s: QuantizedArray) -> Self {
        StorageWrap::QuantizedArray(s)
    }
}

impl ReadChunk for StorageWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read.seek(SeekFrom::Current(0))?;

        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;

        read.seek(SeekFrom::Start(chunk_start_pos))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageWrap::NdArray),
            ChunkIdentifier::QuantizedArray => {
                QuantizedArray::read_chunk(read).map(StorageWrap::QuantizedArray)
            }
            _ => Err(format_err!(
                "Chunk type {:?} cannot be read as storage",
                chunk_id
            )),
        }
    }
}

impl MmapChunk for StorageWrap {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self, Error> {
        let chunk_start_pos = read.seek(SeekFrom::Current(0))?;

        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;

        read.seek(SeekFrom::Start(chunk_start_pos))?;

        match chunk_id {
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageWrap::MmapArray),
            _ => Err(format_err!(
                "Chunk type {:?} cannot be memory mapped as viewable storage",
                chunk_id
            )),
        }
    }
}

impl WriteChunk for StorageWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            StorageWrap::MmapArray(inner) => inner.chunk_identifier(),
            StorageWrap::NdArray(inner) => inner.chunk_identifier(),
            StorageWrap::QuantizedArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        match self {
            StorageWrap::MmapArray(inner) => inner.write_chunk(write),
            StorageWrap::NdArray(inner) => inner.write_chunk(write),
            StorageWrap::QuantizedArray(inner) => inner.write_chunk(write),
        }
    }
}

/// Wrapper for storage types that implement views.
///
/// This type covers the subset of storage types that implement
/// `StorageView`. See the `StorageWrap` type for more information.
pub enum StorageViewWrap {
    MmapArray(MmapArray),
    NdArray(NdArray),
}

impl From<MmapArray> for StorageViewWrap {
    fn from(s: MmapArray) -> Self {
        StorageViewWrap::MmapArray(s)
    }
}

impl From<NdArray> for StorageViewWrap {
    fn from(s: NdArray) -> Self {
        StorageViewWrap::NdArray(s)
    }
}

impl ReadChunk for StorageViewWrap {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_start_pos = read.seek(SeekFrom::Current(0))?;

        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;

        read.seek(SeekFrom::Start(chunk_start_pos))?;

        match chunk_id {
            ChunkIdentifier::NdArray => NdArray::read_chunk(read).map(StorageViewWrap::NdArray),
            _ => Err(format_err!(
                "Chunk type {:?} cannot be read as viewable storage",
                chunk_id
            )),
        }
    }
}

impl WriteChunk for StorageViewWrap {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.chunk_identifier(),
            StorageViewWrap::NdArray(inner) => inner.chunk_identifier(),
        }
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.write_chunk(write),
            StorageViewWrap::NdArray(inner) => inner.write_chunk(write),
        }
    }
}

impl MmapChunk for StorageViewWrap {
    fn mmap_chunk(read: &mut BufReader<File>) -> Result<Self, Error> {
        let chunk_start_pos = read.seek(SeekFrom::Current(0))?;

        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;

        read.seek(SeekFrom::Start(chunk_start_pos))?;

        match chunk_id {
            ChunkIdentifier::NdArray => MmapArray::mmap_chunk(read).map(StorageViewWrap::MmapArray),
            _ => Err(format_err!(
                "Chunk type {:?} cannot be memory mapped as viewable storage",
                chunk_id
            )),
        }
    }
}

/// Embedding matrix storage.
///
/// To allow for embeddings to be stored in different manners (e.g.
/// regular *n x d* matrix or as quantized vectors), this trait
/// abstracts over concrete storage types.
pub trait Storage {
    fn embedding(&self, idx: usize) -> CowArray1<f32>;

    fn shape(&self) -> (usize, usize);
}

impl Storage for MmapArray {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        CowArray::Owned(
            // Alignment is ok, padding guarantees that the pointer is at
            // a multiple of 4.
            #[allow(clippy::cast_ptr_alignment)]
            unsafe { ArrayView2::from_shape_ptr(self.shape, self.map.as_ptr() as *const f32) }
                .row(idx)
                .to_owned(),
        )
    }

    fn shape(&self) -> (usize, usize) {
        self.shape.into_pattern()
    }
}

impl Storage for NdArray {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        CowArray::Borrowed(self.0.row(idx))
    }

    fn shape(&self) -> (usize, usize) {
        self.0.dim()
    }
}

impl Storage for QuantizedArray {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        let mut reconstructed = self.quantizer.reconstruct_vector(self.quantized.row(idx));
        if let Some(ref norms) = self.norms {
            reconstructed *= norms[idx];
        }

        CowArray::Owned(reconstructed)
    }

    fn shape(&self) -> (usize, usize) {
        (self.quantized.rows(), self.quantizer.reconstructed_len())
    }
}

impl Storage for StorageWrap {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            StorageWrap::MmapArray(inner) => inner.embedding(idx),
            StorageWrap::NdArray(inner) => inner.embedding(idx),
            StorageWrap::QuantizedArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            StorageWrap::MmapArray(inner) => inner.shape(),
            StorageWrap::NdArray(inner) => inner.shape(),
            StorageWrap::QuantizedArray(inner) => inner.shape(),
        }
    }
}

impl Storage for StorageViewWrap {
    fn embedding(&self, idx: usize) -> CowArray1<f32> {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.embedding(idx),
            StorageViewWrap::NdArray(inner) => inner.embedding(idx),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.shape(),
            StorageViewWrap::NdArray(inner) => inner.shape(),
        }
    }
}

/// Storage that provide a view of the embedding matrix.
pub trait StorageView: Storage {
    /// Get a view of the embedding matrix.
    fn view(&self) -> ArrayView2<f32>;
}

impl StorageView for NdArray {
    fn view(&self) -> ArrayView2<f32> {
        self.0.view()
    }
}

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

impl StorageView for StorageViewWrap {
    fn view(&self) -> ArrayView2<f32> {
        match self {
            StorageViewWrap::MmapArray(inner) => inner.view(),
            StorageViewWrap::NdArray(inner) => inner.view(),
        }
    }
}

/// Storage that provide a mutable view of the embedding matrix.
pub(crate) trait StorageViewMut: Storage {
    /// Get a view of the embedding matrix.
    fn view_mut(&mut self) -> ArrayViewMut2<f32>;
}

impl StorageViewMut for NdArray {
    fn view_mut(&mut self) -> ArrayViewMut2<f32> {
        self.0.view_mut()
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
            &mut XorShiftRng::from_entropy(),
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
        rng: &mut R,
    ) -> QuantizedArray
    where
        T: TrainPQ<f32>,
        R: Rng;
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
        rng: &mut R,
    ) -> QuantizedArray
    where
        T: TrainPQ<f32>,
        R: Rng,
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

        let quantized = quantizer.quantize_batch(embeds.as_view());

        QuantizedArray {
            quantizer,
            quantized,
            norms,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::Array2;
    use reductive::pq::PQ;

    use crate::io::private::{ReadChunk, WriteChunk};
    use crate::storage::{NdArray, Quantize, QuantizedArray, StorageView};

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
        assert_eq!(arr.quantized, check_arr.quantized);
    }
}
