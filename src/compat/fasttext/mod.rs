//! Reader for the fastText format.
//!
//! This module provides support for reading non-quantized/pruned
//! fastText embeddings. Embeddings in the fastText format are read
//! as follows:
//!
//! ```
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! use finalfusion::prelude::*;
//!
//! let mut reader = BufReader::new(File::open("testdata/fasttext.bin").unwrap());
//!
//! // Read the embeddings.
//! let embeddings = Embeddings::read_fasttext(&mut reader)
//!     .unwrap();
//!
//! // Look up an embedding.
//! let embedding = embeddings.embedding("zwei");
//! ```

mod indexer;
pub use self::indexer::FastTextIndexer;

mod io;
pub use self::io::{ReadFastText, WriteFastText};
