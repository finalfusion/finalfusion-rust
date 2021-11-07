//! Error/result types

use std::io;

use ndarray::ShapeError;
use rand::Error as RandError;
use thiserror::Error;

/// `Result` type alias for operations that can lead to I/O errors.
pub type Result<T> = ::std::result::Result<T, Error>;

/// finalfusion errors
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid file format.
    #[error("Invalid file format {0}")]
    Format(String),

    /// Conversion of n-grams from implicit to explicit.
    #[error("{0}")]
    NGramConversionError(String),

    /// Random number generation error.
    #[error(transparent)]
    RandError(#[from] RandError),

    /// `ndarray` shape error.
    #[error(transparent)]
    Shape(#[from] ShapeError),

    #[error("{desc:?}: {error:?}")]
    Io { desc: String, error: io::Error },

    #[error("Unknown chunk identifier {0}")]
    UnknownChunkIdentifier(u32),

    #[error("Data cannot be represented using native word size")]
    Overflow,

    #[error("Can't convert {from:?} to {to:?}")]
    ConversionError { from: String, to: String },
}

impl Error {
    pub fn ngram_conversion_error(desc: impl Into<String>) -> Self {
        Error::NGramConversionError(desc.into())
    }

    pub fn io_error(desc: impl Into<String>, error: io::Error) -> Self {
        Error::Io {
            desc: desc.into(),
            error,
        }
    }

    pub fn conversion_error(from: impl Into<String>, to: impl Into<String>) -> Self {
        Error::ConversionError {
            from: from.into(),
            to: to.into(),
        }
    }
}
