use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error};
use ndarray::Array1;

use crate::io::private::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};
use crate::util::padding;

/// Trait for norm chunks.
pub trait Norms {
    /// Return the norm for the word at the given index.
    fn norm(&self, idx: usize) -> f32;
}

/// Chunk for storing embedding l2 norms.
///
/// Word embeddings are always l2-normalized in finalfusion. Sometimes
/// it is useful to get the original unnormalized embeddings. The
/// norms chunk is used for storing norms of in-vocabulary embeddings.
/// The unnormalized embedding can be reconstructed by multiplying the
/// normalized embedding by its orginal l2 norm.
#[derive(Clone, Debug)]
pub struct NdNorms(pub Array1<f32>);

impl Norms for NdNorms {
    fn norm(&self, idx: usize) -> f32 {
        self.0[idx]
    }
}

impl ReadChunk for NdNorms {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;
        ensure!(
            chunk_id == ChunkIdentifier::NdNorms,
            "Cannot read chunk {:?} as NdNorms",
            chunk_id
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let len = read.read_u64::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix for NdNorms."
        );

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0))?);
        read.seek(SeekFrom::Current(n_padding as i64))?;

        let mut data = vec![0f32; len];
        read.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(NdNorms(Array1::from_vec(data)))
    }
}

impl WriteChunk for NdNorms {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::NdNorms
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::NdNorms as u32)?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);
        // Chunk size: len (u64), type id (u32), padding ([0,4) bytes), vector.
        let chunk_len = size_of::<u64>()
            + size_of::<u32>()
            + n_padding as usize
            + (self.0.len() * size_of::<f32>());
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.0.len() as u64)?;
        write.write_u32::<LittleEndian>(f32::type_id())?;

        let padding = vec![0; n_padding as usize];
        write.write_all(&padding)?;

        for &val in self.0.iter() {
            write.write_f32::<LittleEndian>(val)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::Array1;

    use crate::io::private::{ReadChunk, WriteChunk};
    use crate::norms::NdNorms;

    const LEN: usize = 100;

    fn test_ndnorms() -> NdNorms {
        NdNorms(Array1::range(0., LEN as f32, 1.))
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
        assert_eq!(arr.0, check_arr.0);
    }
}
