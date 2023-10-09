//! Norms chunk

use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::mem::size_of;
use std::ops::Deref;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array1;

use crate::chunks::io::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};
use crate::error::{Error, Result};
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
            .map_err(|e| Error::read_error("Cannot read norms chunk length", e))?;

        let len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::read_error("Cannot read norms vector length", e))?
            .try_into()
            .map_err(|_| Error::Overflow)?;

        f32::ensure_data_type(read)?;

        let n_padding =
            padding::<f32>(read.stream_position().map_err(|e| {
                Error::read_error("Cannot get file position for computing padding", e)
            })?);
        read.seek(SeekFrom::Current(n_padding as i64))
            .map_err(|e| Error::read_error("Cannot skip padding", e))?;

        let mut data = Array1::zeros((len,));
        read.read_f32_into::<LittleEndian>(data.as_slice_mut().unwrap())
            .map_err(|e| Error::read_error("Cannot read norms", e))?;

        Ok(NdNorms::new(data))
    }
}

impl WriteChunk for NdNorms {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdNorms
    }

    fn chunk_len(&self, offset: u64) -> u64 {
        let n_padding = padding::<f32>(offset + mem::size_of::<u32>() as u64);

        // Chunk identifier (u32) + chunk len (u64) + len (u64) + type id (u32) + padding + vector.
        (mem::size_of::<u32>()
            + mem::size_of::<u64>()
            + mem::size_of::<u64>()
            + mem::size_of::<u32>()
            + self.len() * mem::size_of::<f32>()) as u64
            + n_padding
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        write
            .write_u32::<LittleEndian>(ChunkIdentifier::NdNorms as u32)
            .map_err(|e| Error::write_error("Cannot write norms chunk identifier", e))?;
        let n_padding = padding::<f32>(write.stream_position().map_err(|e| {
            Error::write_error("Cannot get file position for computing padding", e)
        })?);

        let remaining_chunk_len =
            self.chunk_len(write.stream_position().map_err(|e| {
                Error::read_error("Cannot get file position for computing padding", e)
            })?) - (size_of::<u32>() + size_of::<u64>()) as u64;

        write
            .write_u64::<LittleEndian>(remaining_chunk_len)
            .map_err(|e| Error::write_error("Cannot write norms chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.len() as u64)
            .map_err(|e| Error::write_error("Cannot write norms vector length", e))?;
        write
            .write_u32::<LittleEndian>(f32::type_id())
            .map_err(|e| Error::write_error("Cannot write norms vector type identifier", e))?;

        let padding = vec![0; n_padding as usize];
        write
            .write_all(&padding)
            .map_err(|e| Error::write_error("Cannot write padding", e))?;

        for &val in self.iter() {
            write
                .write_f32::<LittleEndian>(val)
                .map_err(|e| Error::write_error("Cannot write norm", e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::Array1;

    use super::NdNorms;
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
        for offset in 0..16u64 {
            let check_arr = test_ndnorms();
            let mut cursor = Cursor::new(Vec::new());
            cursor.seek(SeekFrom::Start(offset)).unwrap();
            check_arr.write_chunk(&mut cursor).unwrap();
            cursor.seek(SeekFrom::Start(offset)).unwrap();

            // Check size remained chunk against embedded chunk size.
            let chunk_size = read_chunk_size(&mut cursor);
            assert_eq!(
                cursor.read_to_end(&mut Vec::new()).unwrap(),
                chunk_size as usize
            );

            // Check overall chunk size.
            assert_eq!(
                cursor.into_inner().len() as u64 - offset,
                check_arr.chunk_len(offset)
            );
        }
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
}
