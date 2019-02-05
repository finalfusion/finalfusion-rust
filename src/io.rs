use std::io::{Read, Write};

use failure::Error;

pub trait ReadChunk
where
    Self: Sized,
{
    fn read_chunk(read: &mut impl Read) -> Result<Self, Error>;
}

pub trait WriteChunk {
    fn write_chunk(&self, write: &mut impl Write) -> Result<(), Error>;
}

pub trait TypeId {
    fn type_id() -> u32;
}

impl TypeId for f32 {
    fn type_id() -> u32 {
        10
    }
}
