use std::collections::HashMap;

pub trait Vocab {
    /// Get the index of a token.
    fn idx(&self, word: &str) -> Option<usize>;

    /// Get the vocabulary size.
    fn len(&self) -> usize;

    /// Get the words in the vocabulary.
    fn words(&self) -> &[String];
}

pub struct SimpleVocab {
    indices: HashMap<String, usize>,
    words: Vec<String>,
}

impl SimpleVocab {
    pub fn new(words: impl Into<Vec<String>>) -> Self {
        let words = words.into();

        let mut indices = HashMap::new();
        for (idx, word) in words.iter().enumerate() {
            indices.insert(word.to_owned(), idx);
        }

        SimpleVocab { words, indices }
    }
}

impl Vocab for SimpleVocab {
    fn idx(&self, word: &str) -> Option<usize> {
        self.indices.get(word).cloned()
    }

    fn len(&self) -> usize {
        self.words.len()
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}
