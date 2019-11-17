//! Norms chunk

use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::ops::Deref;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array1;

use super::io::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};
use crate::io::{ErrorKind, Result};
use crate::util::padding;

/// Chunk for storing embedding l2 norms.
///
/// Word embeddings are always l2-normalized in finalfusion. Sometimes
/// it is useful to get the original unnormalized embeddings. The
/// norms chunk is used for storing norms of in-vocabulary embeddings.
/// The unnormalized embedding can be reconstructed by multiplying the
/// normalized embedding by its orginal l2 norm.
#[derive(Clone, Debug)]
pub struct NdNorms {
    inner: Array1<f32>,
}

impl NdNorms {
    /// Construct new `NdNorms`.
    pub fn new(norms: impl Into<Array1<f32>>) -> Self {
        NdNorms {
            inner: norms.into(),
        }
    }
}

impl Deref for NdNorms {
    type Target = Array1<f32>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<V> From<V> for NdNorms
where
    V: Into<Array1<f32>>,
{
    fn from(array: V) -> NdNorms {
        NdNorms::new(array)
    }
}

impl ReadChunk for NdNorms {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::NdNorms)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read norms chunk length", e))?;

        let len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read norms vector length", e))?
            as usize;

        f32::ensure_data_type(read)?;

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| ErrorKind::io_error("Cannot skip padding", e))?;

        let mut data = vec![0f32; len];
        read.read_f32_into::<LittleEndian>(&mut data)
            .map_err(|e| ErrorKind::io_error("Cannot read norms", e))?;

        Ok(NdNorms::new(data))
    }
}

impl WriteChunk for NdNorms {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdNorms
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdNorms as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write norms chunk identifier", e))?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0)).map_err(|e| {
            ErrorKind::io_error("Cannot get file position for computing padding", e)
        })?);

        // Chunk size: len (u64), type id (u32), padding ([0,4) bytes), vector.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + n_padding as usize
            + (self.len() * size_of::<f32>());
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write norms chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write norms vector length", e))?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| ErrorKind::io_error("Cannot write norms vector type identifier", e))?;

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| ErrorKind::io_error("Cannot write padding", e))?;

        for &val in self.iter() {
            write
                .write_f32::<LittleEndian>(val)
                .map_err(|e| ErrorKind::io_error("Cannot write norm", e))?;
        }

        Ok(())
    }
}

/// Prune the embedding norms.
pub trait PruneNorms {
    /// Prune the embedding norms. Remap the norms of the words whose original vectors need to be
    /// tossed to their nearest remaining vectors' norms.
    fn prune_norms(&self, toss_indices: &[usize], most_similar_indices: &Array1<usize>) -> NdNorms;
}

impl PruneNorms for NdNorms {
    fn prune_norms(&self, toss_indices: &[usize], most_similar_indices: &Array1<usize>) -> NdNorms {
        let mut pruned_norms = self.inner.clone();
        for (toss_idx, remapped_idx) in toss_indices.iter().zip(most_similar_indices) {
            pruned_norms[*toss_idx] = pruned_norms[*remapped_idx];
        }
        NdNorms::new(pruned_norms)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::ops::Deref;

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::{arr1, Array1};

    use super::{NdNorms, PruneNorms};
    use crate::chunks::io::{ReadChunk, WriteChunk};

    const LEN: usize = 100;

    fn test_ndnorms() -> NdNorms {
        NdNorms::new(Array1::range(0., LEN as f32, 1.))
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[test]
    fn ndnorms_correct_chunk_size() {
        let check_arr = test_ndnorms();
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
    fn ndnorms_write_read_roundtrip() {
        let check_arr = test_ndnorms();
        let mut cursor = Cursor::new(Vec::new());
        check_arr.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let arr = NdNorms::read_chunk(&mut cursor).unwrap();
        assert_eq!(arr.view(), check_arr.view());
    }

    #[test]
    fn test_prune_norms() {
        let original_norms = test_ndnorms();
        let toss_indices = &[1, 5, 7];
        let most_similar_indices = arr1(&[2, 6, 8]);
        let test_ndnorms = original_norms.prune_norms(toss_indices, &most_similar_indices);
        for (toss_idx, remap_idx) in toss_indices.iter().zip(most_similar_indices.iter()) {
            assert_eq!(
                test_ndnorms.deref()[*toss_idx],
                test_ndnorms.deref()[*remap_idx]
            );
            assert_eq!(
                original_norms.deref()[*remap_idx],
                test_ndnorms.deref()[*toss_idx]
            );
        }
    }
}
