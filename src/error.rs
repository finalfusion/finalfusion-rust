//! Error/result types

use std::io;

use ndarray::ShapeError;
use reductive::error::ReductiveError;
use thiserror::Error;

/// `Result` type alias for operations that can lead to I/O errors.
pub type Result<T> = ::std::result::Result<T, Error>;

/// finalfusion errors
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// Add more context to an error.
    #[error("{context}")]
    Context {
        context: String,
        #[source]
        error: Box<Error>,
    },

    /// Invalid file format.
    #[error("Invalid file format {0}")]
    Format(String),

    /// Conversion of n-grams from implicit to explicit.
    #[error("{0}")]
    NGramConversion(String),

    /// Matrix construction shape error
    #[error("Cannot construct matrix with the given shape")]
    MatrixShape(#[source] ShapeError),

    #[error("{desc}")]
    Read {
        desc: String,
        #[source]
        error: io::Error,
    },

    #[error("Unknown chunk identifier {0}")]
    UnknownChunkIdentifier(u32),

    #[error("Data cannot be represented using native word size")]
    Overflow,

    #[error("Can't convert {from} to {to}")]
    Conversion { from: String, to: String },

    #[error("Cannot quantize embeddings")]
    Quantization(#[source] ReductiveError),

    #[error("{desc}")]
    Write {
        desc: String,
        #[source]
        error: io::Error,
    },
}

impl Error {
    pub fn context(self, context: impl Into<String>) -> Self {
        Error::Context {
            context: context.into(),
            error: self.into(),
        }
    }

    pub fn ngram_conversion_error(desc: impl Into<String>) -> Self {
        Error::NGramConversion(desc.into())
    }

    pub fn conversion_error(from: impl Into<String>, to: impl Into<String>) -> Self {
        Error::Conversion {
            from: from.into(),
            to: to.into(),
        }
    }

    pub fn read_error(desc: impl Into<String>, error: io::Error) -> Self {
        Error::Read {
            desc: desc.into(),
            error,
        }
    }

    pub fn write_error(desc: impl Into<String>, error: io::Error) -> Self {
        Error::Write {
            desc: desc.into(),
            error,
        }
    }
}
