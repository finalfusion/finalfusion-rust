use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array1;

use crate::chunks::io::{ChunkIdentifier, ReadChunk, WriteChunk};
use crate::chunks::vocab::{
    create_indices, read_vocab_items, write_vocab_items, Vocab, VocabPrune, VocabPruneIndices,
    VocabWrap, WordIndex,
};
use crate::io::{ErrorKind, Result};

/// Vocabulary without subword units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl SimpleVocab {
    /// Construct a new simple vocabulary.
    ///
    /// Words are assigned indices in the given order.
    ///
    /// Panics when there are duplicate words.
    pub fn new(words: impl Into<Vec<String>>) -> Self {
        let words = words.into();
        let indices = create_indices(&words);
        assert_eq!(
            words.len(),
            indices.len(),
            "words contained duplicate entries."
        );
        SimpleVocab { words, indices }
    }
}

impl Vocab for SimpleVocab {
    fn idx(&self, word: &str) -> Option<WordIndex> {
        self.indices.get(word).cloned().map(WordIndex::Word)
    }

    fn words_len(&self) -> usize {
        self.indices.len()
    }

    fn vocab_len(&self) -> usize {
        self.words_len()
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}

impl ReadChunk for SimpleVocab {
    fn read_chunk<R>(read: &mut R) -> Result<Self>
    where
        R: Read + Seek,
    {
        ChunkIdentifier::ensure_chunk_type(read, ChunkIdentifier::SimpleVocab)?;

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary chunk length", e))?;

        let vocab_len = read
            .read_u64::<LittleEndian>()
            .map_err(|e| ErrorKind::io_error("Cannot read vocabulary length", e))?
            as usize;

        let words = read_vocab_items(read, vocab_len)?;

        Ok(SimpleVocab::new(words))
    }
}

impl WriteChunk for SimpleVocab {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::SimpleVocab
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        // Chunk size: vocabulary size (u64), for each word:
        // word length in bytes (4 bytes), word bytes (variable-length).
        let chunk_len = size_of::<u64>()
            + self
                .words
                .iter()
                .map(|w| w.len() + size_of::<u32>())
                .sum::<usize>();

        write
            .write_u32::<LittleEndian>(ChunkIdentifier::SimpleVocab as u32)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary chunk identifier", e))?;
        write
            .write_u64::<LittleEndian>(chunk_len as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary chunk length", e))?;
        write
            .write_u64::<LittleEndian>(self.words.len() as u64)
            .map_err(|e| ErrorKind::io_error("Cannot write vocabulary length", e))?;

        write_vocab_items(write, self.words())?;

        Ok(())
    }
}

impl VocabPrune for SimpleVocab {
    fn prune_vocab(&self, remapped_indices: HashMap<String, usize>) -> VocabWrap {
        let new_vocab = SimpleVocab {
            words: self.words.clone(),
            indices: remapped_indices,
        };
        new_vocab.into()
    }
}

impl VocabPruneIndices for SimpleVocab {
    fn part_indices(&self, n_keep: usize) -> (Vec<usize>, Vec<usize>) {
        let keep_indices = self
            .words()
            .iter()
            .take(n_keep)
            .map(|w| *self.indices.get(w).unwrap())
            .collect();
        let toss_indices = self.words()[n_keep..]
            .iter()
            .map(|w| *self.indices.get(w).unwrap())
            .collect();
        (keep_indices, toss_indices)
    }

    fn create_remapped_indices(
        &self,
        most_similar_indices: &Array1<usize>,
    ) -> HashMap<String, usize> {
        let mut remapped_indices = self.indices.clone();
        for (toss_word, remapped_idx) in self.words()
            [self.words_len() - most_similar_indices.len()..]
            .iter()
            .zip(most_similar_indices)
        {
            remapped_indices.insert(toss_word.to_owned(), *remapped_idx);
        }
        remapped_indices
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use ndarray::arr1;

    use super::SimpleVocab;
    use crate::chunks::io::{ReadChunk, WriteChunk};
    use crate::chunks::vocab::{read_chunk_size, VocabPruneIndices};

    fn test_simple_vocab() -> SimpleVocab {
        let words = vec![
            "this".to_owned(),
            "is".to_owned(),
            "a".to_owned(),
            "test".to_owned(),
        ];

        SimpleVocab::new(words)
    }

    #[test]
    fn simple_vocab_write_read_roundtrip() {
        let check_vocab = test_simple_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let vocab = SimpleVocab::read_chunk(&mut cursor).unwrap();
        assert_eq!(vocab, check_vocab);
    }

    #[test]
    fn simple_vocab_correct_chunk_size() {
        let check_vocab = test_simple_vocab();
        let mut cursor = Cursor::new(Vec::new());
        check_vocab.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }

    #[test]
    fn test_part_indices() {
        let vocab = test_simple_vocab();
        let (test_keep_indices, test_toss_indices) = vocab.part_indices(2);
        assert_eq!(vec![0, 1], test_keep_indices);
        assert_eq!(vec![2, 3], test_toss_indices);
    }

    #[test]
    fn test_create_remapped_indices() {
        let vocab = test_simple_vocab();
        let test_remapped_indices = vocab.create_remapped_indices(&arr1(&[1, 0]));
        let remapped_indices: HashMap<String, usize> = [
            ("this".to_owned(), 0),
            ("is".to_owned(), 1),
            ("a".to_owned(), 1),
            ("test".to_owned(), 0),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(remapped_indices, test_remapped_indices);
    }
}
