use std::io::BufRead;
use std::mem::size_of;

use crate::io::{Error, ErrorKind, Result};
use ndarray::{Array1, ArrayViewMut1, ArrayViewMut2};

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

pub fn read_number(reader: &mut BufRead, delim: u8) -> Result<usize> {
    let field_str = read_string(reader, delim)?;
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

pub fn read_string(reader: &mut BufRead, delim: u8) -> Result<String> {
    let mut buf = Vec::new();
    reader.read_until(delim, &mut buf)?;
    buf.pop();
    Ok(String::from_utf8(buf)
        .map_err(|e| ErrorKind::Format(format!("Token contains invalid UTF-8: {}", e)))?)
}
