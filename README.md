## Introduction

[![crates.io](https://img.shields.io/crates/v/finalfusion.svg)](https://crates.io/crates/finalfusion)
[![docs.rs](https://docs.rs/finalfusion/badge.svg)](https://docs.rs/finalfusion/)
[![Travis CI](https://img.shields.io/travis/finalfusion/finalfusion-rust.svg)](https://travis-ci.org/finalfusion/finalfusion-rust)

`finalfusion` is a crate for reading, writing, and using embeddings in
Rust. `finalfusion` primarily works with
[its own format](https://finalfusion.github.io/spec) which supports a large
variety of features. Additionally, the fastText, word2vec and GloVe file
formats are also supported.

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

## Where to go from here

  * [finalfusion](https://finalfusion.github.io/)
  * [pretrained embeddings](https://finalfusion.github.io/pretrained)
  * [finalfrontier](https://finalfusion.github.io/finalfrontier)
  * [finalfusion-python](https://finalfusion.github.io/python)
  * [reductive](https://github.com/finalfusion/reductive)
