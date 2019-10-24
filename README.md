## Introduction

[![crates.io](https://img.shields.io/crates/v/finalfusion.svg)](https://crates.io/crates/finalfusion)
[![docs.rs](https://docs.rs/finalfusion/badge.svg)](https://docs.rs/finalfusion/)
[![Travis CI](https://img.shields.io/travis/finalfusion/finalfusion-rust.svg)](https://travis-ci.org/finalfusion/finalfusion-rust)

`finalfusion` is a crate for reading, writing, and using embeddings in
Rust. `finalfusion` primarily works with
[its own format](https://finalfusion.github.io/spec) which supports a large
variety of features. Additionally, the fastText, word2vec and GloVe file
formats are also supported.

`finalfusion` is API stable since 0.11.0. However, we cannot tag
version 1 yet, because several dependencies that are exposed through
the API have not reached version 1 (particularly `ndarray` and
`rand`). Future 0.x releases of `finalfusion` will be used to accomodate
updates of these dependencies.

## Usage

To make `finalfusion` available in your crate, simply place the following
in your `Cargo.toml`

~~~
finalfusion = 0.10
~~~

Loading embeddings and querying it is as simple as:

~~~Rust
use std::fs::File;
use std::io::BufReader;

import finalfusion::prelude::*;

fn main() {
    let mut reader = BufReader::new(File::open("embeddings.fifu").unwrap());
    let embeds = Embeddings::<VocabWrap, StorageWrap>::read_embeddings(&mut reader).unwrap();
    embeds.embedding("Query").unwrap();
}
~~~

## Features

`finalfusion` supports a variety of formats:

* Vocabulary
    * Subwords
    * No subwords
* Storage
    * Array
    * Memory-mapped
    * Quantized
* Format
    * [finalfusion](https://finalfusion.github.io/spec)
    * fastText
    * word2vec
    * GloVe
    
Moreover, `finalfusion` provides: 

* Similarity queries
* Analogy queries
* Quantizing embeddings through [reductive](https://github.com/finalfusion/reductive)
* Conversion to the following formats:
    * finalfusion
    * word2vec
    * GloVe

For more information, please consult the [API documentation](http://docs.rs/finalfusion/).

## Getting embeddings

Embeddings trained with [finalfrontier](https://finalfusion.github.io/finalfrontier) starting
with version `0.4` are in `finalfusion` format and compatible with his crate. A growing set
of [pretrained embeddings](https://finalfusion.github.io/pretrained) is offered on our website
and we have converted the fastText Wikipedia and Common Crawl embeddings to `finalfusion`.
More information can also be found at https://finalfusion.github.io.

## Which type of storage should I use?

### Quantized embeddings

Quantized embeddings store embeddings as discrete
representations. Imagine that for a given embeddings space, you would
find 256 prototypical embeddings. Each embedding could then be stored
as a 1-byte pointer to one of these prototypical embeddings. Of
course, having only 256 possible representations, this quantized
embedding space would be very coarse-grained.

**product quantizers** (**pq**) solve this problem by splitting each
embedding evenly into *q* subvectors and finding prototypical vectors
for each set of subvectors. If we use 256 prototypical representations
for each subspace, *256^q* different word embeddings can be
represented. For instance, if *q = 150*, we could represent *250^150*
different embeddings. Each embedding would then be stored as 150
byte-sized pointers.

**optimized product quantizers** (**opq**) additionally applies a
linear map to the embedding space to distribute variance across
embedding dimensions.

By quantizing an embedding matrix, its size can be reduced both on
disk and in memory.

### Memory mapped embeddings

Normally, we read embeddings into memory. However, as an alternative
the embeddings can be **memory mapped**. [Memory
mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) makes the
on-disk embedding matrix available as pages in virtual memory. The
operating system will then (transparently) load these pages into
physical memory as necessary.

Memory mapping speeds up the initial loading time of word embeddings,
since only the vocabulary needs to be read. The operating system will
then load (part of the) embedding matrix a by-need basis. The
operating system can additionally free up the memory again when no
embeddings are looked up and other processes require memory.

### Empirical comparison

The following empirical comparison of embedding types uses an
embedding matrix with 2,807,440 embeddings (710,288 word, 2,097,152
subword) of dimensionality 300. The embedding lookup timings were done
on an Intel Core i5-8259U CPU, 2.30GHz.

*Known lookup* and *Unknown lookup* time lookups of words that are
inside/outside the vocabulary. *Lookup* contains a mixture of known
and unknown words.

| Storage    | Lookup | Known lookup | Unknown lookup |   Memory |     Disk |
|:-----------|-------:|-------------:|---------------:|---------:|---------:|
| array      | 449 ns |       232 ns |          18 μs | 3213 MiB | 3213 MiB |
| array mmap | 833 ns |       494 ns |          23 μs | Variable | 3213 MiB |
| opq        |  40 μs |        21 μs |         962 μs |  402 MiB |  402 MiB |
| opq mmap   |  41 μs |        21 μs |         960 μs | Variable |  402 MiB |

**Note:** two units are used: nanoseconds (ns) and microseconds (μs).

## Using a BLAS or LAPACK library

If you are using `finalfusion` in an application, you can choose to
enable the use of a BLAS/LAPACK libraries in `reductive` using the
features:

* `reductive/netblas`: Use reference BLAS/LAPACK (slow, not recommended)
* `reductive/openblas`: Use OpenBLAS
* `reductive/intel-mkl`: Use Intel Math Kernel Library

Compiling against a LAPACK library is required to quantize embedding
matrices using an *optimized product quantizer*. You do **not** need
such a library to use quantized embeddings.

Embedding lookups in embedding matrices that were quantized using the
optimized product quantizer can be speeded up using a good BLAS
implementation. The following table compares lookup times on an
Intel Core i5-8259U CPU, 2.30GHz with finalfusion compiled with and
without MKL/OpenBLAS support:

| Storage             | Lookup | Known lookup | Unknown lookup |
|:--------------------|-------:|-------------:|---------------:|
| opq                 |  40 μs |        21 μs |         962 μs |
| opq mmap            |  41 μs |        21 μs |         960 μs |
| opq (MKL)           |  14 μs |         7 μs |         309 μs |
| opq mmap (MKL)      |  14 μs |         7 μs |         309 μs |
| opq (OpenBLAS)      |  15 μs |         7 μs |         336 μs |
| opq mmap (OpenBLAS) |  15 μs |         7 μs |         342 μs |


## Where to go from here

  * [finalfusion](https://finalfusion.github.io/)
  * [pretrained embeddings](https://finalfusion.github.io/pretrained)
  * [finalfrontier](https://finalfusion.github.io/finalfrontier)
  * [finalfusion-python](https://finalfusion.github.io/python)
  * [reductive](https://github.com/finalfusion/reductive)
