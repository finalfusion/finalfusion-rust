use std::mem::size_of;

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
