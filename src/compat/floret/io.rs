use std::io::{BufRead, Write};

use itertools::Itertools;
use ndarray::{s, Array2};

use crate::chunks::storage::NdArray;
use crate::compat::floret::FloretIndexer;
use crate::embeddings::Embeddings;
use crate::error::{Error, Result};
use crate::storage::StorageView;
use crate::util::{read_number, read_string};
use crate::vocab::{FloretSubwordVocab, IndicesScope, Vocab};

/// Read embeddings in the floret format.
///
/// More information about how floret embeddings can be found at:
///
/// https://github.com/explosion/floret#how-floret-works
pub trait ReadFloretText
where
    Self: Sized,
{
    /// Read embeddings in the floret format.
    fn read_floret_text(reader: &mut impl BufRead) -> Result<Self>;
}

impl ReadFloretText for Embeddings<FloretSubwordVocab, NdArray> {
    fn read_floret_text(reader: &mut impl BufRead) -> Result<Self> {
        let n_buckets = read_number(reader, b' ')?;
        let embed_len = read_number(reader, b' ')?;
        let min_n = read_number(reader, b' ')? as u32;
        let max_n = read_number(reader, b' ')? as u32;
        let n_hashes = read_number(reader, b' ')? as u32;
        let hash_seed = read_number(reader, b' ')?;
        let bow = read_string(reader, b' ', false)?;
        let eow = read_string(reader, b'\n', false)?;

        if n_buckets == 0 {
            return Err(Error::Format("Expected at least 1 bucket".to_string()));
        }

        if embed_len == 0 {
            return Err(Error::Format(
                "Embeddings should have at least 1 dimension".to_string(),
            ));
        }

        if min_n > max_n {
            return Err(Error::Format(format!(
                "The minimum n-gram length ({}) must not be larger than the maximum length ({})",
                min_n, max_n
            )));
        }

        if !(1..=4).contains(&n_hashes) {
            return Err(Error::Format(format!(
                "The number of hashes should be between 1 and 4 (inclusive), was: {}",
                n_hashes
            )));
        }

        let mut data = Vec::with_capacity(n_buckets * embed_len);

        let mut prev_len = 0;
        for line in reader.lines() {
            let line = line.map_err(|err| Error::io_error("Cannot read line", err))?;

            let parts = line
                .split(|c: char| c.is_ascii_whitespace())
                .filter(|part| !part.is_empty());

            // Skip the first column, which is the embedding index.
            for part in parts.skip(1) {
                data.push(part.parse().map_err(|e| {
                    Error::Format(format!("Cannot parse vector component '{}': {}", part, e))
                })?);
            }

            if data.len() - prev_len != embed_len {
                return Err(Error::Format(format!(
                    "Incorrect number of embedding components, expected: {}, got: {}",
                    embed_len,
                    data.len() - prev_len
                )));
            }

            prev_len += embed_len;
        }

        let matrix = Array2::from_shape_vec((n_buckets, embed_len), data).map_err(Error::Shape)?;

        let indexer = FloretIndexer::new(n_buckets as u64, n_hashes, hash_seed as u32);

        Ok(Embeddings::new_with_maybe_norms(
            None,
            FloretSubwordVocab::new_with_boundaries(
                Vec::new(),
                min_n,
                max_n,
                indexer,
                IndicesScope::WordAndSubword,
                bow,
                eow,
            ),
            NdArray::new(matrix),
            None,
        ))
    }
}

/// Write embeddings in the floret format.
pub trait WriteFloretText
where
    Self: Sized,
{
    /// Read embeddings in the floret format.
    fn write_floret_text(&self, write: &mut dyn Write) -> Result<()>;
}

impl WriteFloretText for Embeddings<FloretSubwordVocab, NdArray> {
    fn write_floret_text(&self, write: &mut dyn Write) -> Result<()> {
        writeln!(
            write,
            "{} {} {} {} {} {} {} {}",
            self.vocab().vocab_len(),
            self.dims(),
            self.vocab().min_n(),
            self.vocab().max_n(),
            self.vocab().indexer().n_hashes(),
            self.vocab().indexer().seed(),
            self.vocab().bow(),
            self.vocab().eow()
        )
        .map_err(|e| Error::io_error("Cannot write floret embeddings metadata", e))?;

        // Ensure that we are only writing part of the embedding matrix which is
        // used for floret embeddings. The storage may have word embeddings through
        // other means (e.g. when used as an input vocab for training).
        let storage_view = self.storage().view();
        let hash_matrix = storage_view.slice(s![self.vocab().words_len().., ..]);

        for (idx, embed) in hash_matrix.outer_iter().enumerate() {
            let embed_str = embed.view().iter().map(ToString::to_string).join(" ");
            writeln!(write, "{} {}", idx, embed_str)
                .map_err(|e| Error::io_error("Cannot write embedding", e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::compat::floret::WriteFloretText;
    use approx::assert_abs_diff_eq;

    use super::ReadFloretText;
    use crate::compat::text::ReadTextDims;
    use crate::embeddings::Embeddings;

    fn floret_embeds_small_text() -> &'static str {
        // Example from spaCy tests.
        "10 10 2 3 2 2166136261 < >
0 -2.2611 3.9302 2.6676 -11.233 0.093715 -10.52 -9.6463 -0.11853 2.101 -0.10145
1 -3.12 -1.7981 10.7 -6.171 4.4527 10.967 9.073 6.2056 -6.1199 -2.0402
2 9.5689 5.6721 -8.4832 -1.2249 2.1871 -3.0264 -2.391 -5.3308 -3.2847 -4.0382
3 3.6268 4.2759 -1.7007 1.5002 5.5266 1.8716 -12.063 0.26314 2.7645 2.4929
4 -11.683 -7.7068 2.1102 2.214 7.2202 0.69799 3.2173 -5.382 -2.0838 5.0314
5 -4.3024 8.0241 2.0714 -1.0174 -0.28369 1.7622 7.8797 -1.7795 6.7541 5.6703
6 8.3574 -5.225 8.6529 8.5605 -8.9465 3.767 -5.4636 -1.4635 -0.98947 -0.58025
7 -10.01 3.3894 -4.4487 1.1669 -11.904 6.5158 4.3681 0.79913 -6.9131 -8.687
8 -5.4576 7.1019 -8.8259 1.7189 4.955 -8.9157 -3.8905 -0.60086 -2.1233 5.892
9 8.0678 -4.4142 3.6236 4.5889 -2.7611 2.4455 0.67096 -4.2822 2.0875 4.6274
"
    }

    fn floret_embeds_square_brackets() -> &'static str {
        "10 10 2 3 2 2166136261 [ ]
0 -2.2611 3.9302 2.6676 -11.233 0.093715 -10.52 -9.6463 -0.11853 2.101 -0.10145
1 -3.12 -1.7981 10.7 -6.171 4.4527 10.967 9.073 6.2056 -6.1199 -2.0402
2 9.5689 5.6721 -8.4832 -1.2249 2.1871 -3.0264 -2.391 -5.3308 -3.2847 -4.0382
3 3.6268 4.2759 -1.7007 1.5002 5.5266 1.8716 -12.063 0.26314 2.7645 2.4929
4 -11.683 -7.7068 2.1102 2.214 7.2202 0.69799 3.2173 -5.382 -2.0838 5.0314
5 -4.3024 8.0241 2.0714 -1.0174 -0.28369 1.7622 7.8797 -1.7795 6.7541 5.6703
6 8.3574 -5.225 8.6529 8.5605 -8.9465 3.767 -5.4636 -1.4635 -0.98947 -0.58025
7 -10.01 3.3894 -4.4487 1.1669 -11.904 6.5158 4.3681 0.79913 -6.9131 -8.687
8 -5.4576 7.1019 -8.8259 1.7189 4.955 -8.9157 -3.8905 -0.60086 -2.1233 5.892
9 8.0678 -4.4142 3.6236 4.5889 -2.7611 2.4455 0.67096 -4.2822 2.0875 4.6274"
    }

    fn check_embeds() -> &'static str {
        "10 10
, -5.7814 2.6918 0.57029 -3.6985 -2.7079 1.4406 1.0084 1.7463 -3.8625 -3.0565
. 3.8016 -1.759 0.59118 3.3044 -0.72975 0.45221 -2.1412 -3.8933 -2.1238 -0.47409
der 0.08224 2.6601 -1.173 1.1549 -0.42821 -0.097268 -2.5589 -1.609 -0.16968 0.84687
die -2.8781 0.082576 1.9286 -0.33279 0.79488 3.36 3.5609 -0.64328 -2.4152 0.17266
und 2.1558 1.8606 -1.382 0.45424 -0.65889 1.2706 0.5929 -2.0592 -2.6949 -1.6015
\" -1.1242 1.4588 -1.6263 1.0382 -2.7609 -0.99794 -0.83478 -1.5711 -1.2137 1.0239
in -0.87635 2.0958 4.0018 -2.2473 -1.2429 2.3474 1.8846 0.46521 -0.506 -0.26653
von -0.10589 1.196 1.1143 -0.40907 -1.0848 -0.054756 -2.5016 -1.0381 -0.41598 0.36982
( 0.59263 2.1856 0.67346 1.0769 1.0701 1.2151 1.718 -3.0441 2.7291 3.719
) 0.13812 3.3267 1.657 0.34729 -3.5459 0.72372 0.63034 -1.6145 1.2733 0.37798"
    }

    fn check_embeds_square_brackets() -> &'static str {
        "10 10
, 1.3844874 2.3464875 1.2599748 -0.6150249 -2.7724452 -0.79785013 -4.0532503 -1.1515112 0.19298255 -0.7406751
. 5.3217626 2.0444875 -2.7715 2.684125 -0.52285004 -2.1163874 -3.1512802 -3.050415 -1.7490175 0.39203754
der -0.30920622 3.038363 -0.68778753 -0.563806 1.5502453 -0.06880643 -1.2151338 -0.047910027 -1.5533295 0.95536256
die -2.2371624 0.43071893 1.4706686 -0.67453104 1.2487259 2.3045924 0.17036629 0.23663676 -2.5797918 0.075331196
und 0.24818125 2.7903755 0.3526249 0.09495008 -2.0179207 0.23948678 -0.71544373 -1.95366 0.3281494 -0.07667194
\" -2.6215875 3.4702003 -0.0053626 -3.4728873 2.9032319 -4.2840385 -5.7323003 -0.9489587 1.7973748 2.659394
in 1.3580999 2.151925 0.028333127 -1.8494834 1.0040858 0.08962494 -2.3529234 -0.5272758 -1.1616334 -0.076175064
von -1.6226685 -0.16304988 2.0203125 -0.8460312 -0.5596545 2.6361988 -0.19833624 -0.30120564 -2.102173 -0.793722
( -5.6849623 -0.17611253 -1.6755875 2.0539 -4.991875 3.5249727 2.00655 -1.7952224 -4.5117707 -3.6629562
) -3.4659 3.5374627 -2.0717626 0.44365007 -1.2155627 3.4016874 -2.2377748 1.0989438 -2.586125 -1.8413126"
    }

    #[test]
    fn test_floret_against_known() {
        let check_embeds_text = check_embeds();
        let check_embeds = Embeddings::read_text_dims(&mut Cursor::new(check_embeds_text)).unwrap();

        let floret_embeds_text = floret_embeds_small_text();
        let floret_embeds =
            Embeddings::read_floret_text(&mut Cursor::new(floret_embeds_text)).unwrap();

        for (word, check_embedding) in check_embeds.iter() {
            let floret_embedding = floret_embeds.embedding(word).unwrap();

            assert_abs_diff_eq!(floret_embedding, check_embedding, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_floret_non_standard_brackets() {
        let check_embeds_text = check_embeds_square_brackets();
        let check_embeds = Embeddings::read_text_dims(&mut Cursor::new(check_embeds_text)).unwrap();

        let floret_embeds_text = floret_embeds_square_brackets();
        let floret_embeds =
            Embeddings::read_floret_text(&mut Cursor::new(floret_embeds_text)).unwrap();

        for (word, check_embedding) in check_embeds.iter() {
            let floret_embedding = floret_embeds.embedding(word).unwrap();

            assert_abs_diff_eq!(floret_embedding, check_embedding, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_floret_read_write_roundtrip() {
        let floret_embeds_text = floret_embeds_small_text();
        let floret_embeds =
            Embeddings::read_floret_text(&mut Cursor::new(floret_embeds_text)).unwrap();

        let mut output = Vec::new();
        floret_embeds.write_floret_text(&mut output).unwrap();

        assert_eq!(output, floret_embeds_text.as_bytes());
    }
}
