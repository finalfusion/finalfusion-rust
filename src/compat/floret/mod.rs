//! Support for the floret embedding format.
//!
//! More information about floret can be found at:
//! https://github.com/explosion/floret#how-floret-works
//!
//! floret differs from finalfusion/fasttext embeddings in the
//! following ways:
//!
//! * No separate embeddings are stored for words.
//! * The word and its n-grams are mapped to 1-4 buckets.

mod io;
pub use io::ReadFloretText;

mod indexer;
pub use indexer::FloretIndexer;
