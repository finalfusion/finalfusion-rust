use std::collections::VecDeque;
use std::io::BufRead;
use std::mem::size_of;

use crate::io::{Error, ErrorKind, Result};
use ndarray::{Array1, ArrayViewMut1, ArrayViewMut2};

/// Conversion from an `Iterator` into a collection with a given
/// capacity.
pub trait FromIteratorWithCapacity<T> {
    /// Construct a collection with the given capacity from an iterator.
    fn from_iter_with_capacity<I>(iter: I, capacity: usize) -> Self
    where
        I: IntoIterator<Item = T>;
}

impl<T> FromIteratorWithCapacity<T> for Vec<T> {
    fn from_iter_with_capacity<I>(iter: I, capacity: usize) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut v = Vec::with_capacity(capacity);
        v.extend(iter.into_iter());
        v
    }
}

impl<T> FromIteratorWithCapacity<T> for VecDeque<T> {
    fn from_iter_with_capacity<I>(iter: I, capacity: usize) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut v = VecDeque::with_capacity(capacity);
        v.extend(iter.into_iter());
        v
    }
}

/// Collect iterms from an `Iterator` into a collection with a
/// capacity.
pub trait CollectWithCapacity {
    type Item;

    /// Transform an iterator into a collection with the given capacity.
    fn collect_with_capacity<B>(self, capacity: usize) -> B
    where
        B: FromIteratorWithCapacity<Self::Item>;
}

impl<I, T> CollectWithCapacity for I
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn collect_with_capacity<B>(self, capacity: usize) -> B
    where
        B: FromIteratorWithCapacity<Self::Item>,
    {
        B::from_iter_with_capacity(self, capacity)
    }
}

pub fn padding<T>(pos: u64) -> u64 {
    let size = size_of::<T>() as u64;
    size - (pos % size)
}

pub fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

pub fn l2_normalize_array(mut v: ArrayViewMut2<f32>) -> Array1<f32> {
    let mut norms = Vec::with_capacity(v.rows());
    for embedding in v.outer_iter_mut() {
        norms.push(l2_normalize(embedding));
    }

    norms.into()
}

pub fn read_number(reader: &mut dyn BufRead, delim: u8) -> Result<usize> {
    let field_str = read_string(reader, delim, false)?;
    field_str
        .parse()
        .map_err(|e| {
            ErrorKind::Format(format!(
                "Cannot parse shape component '{}': {}",
                field_str, e
            ))
        })
        .map_err(Error::from)
}

pub fn read_string(reader: &mut dyn BufRead, delim: u8, lossy: bool) -> Result<String> {
    let mut buf = Vec::new();
    reader
        .read_until(delim, &mut buf)
        .map_err(|e| ErrorKind::io_error("Cannot read string", e))?;
    buf.pop();

    let s = if lossy {
        String::from_utf8_lossy(&buf).into_owned()
    } else {
        String::from_utf8(buf)
            .map_err(|e| ErrorKind::Format(format!("Token contains invalid UTF-8: {}", e)))?
    };

    Ok(s)
}
