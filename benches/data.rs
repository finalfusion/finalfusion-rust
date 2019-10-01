use std::fs::File;
use std::io::{BufRead, BufReader};

const TEST_CORPUS: &str = "benches/dewiki-aa-00.txt";

pub fn load_corpus() -> Vec<String> {
    let f = File::open(TEST_CORPUS).expect("Corpus file missing, run fetch-data.sh");
    BufReader::new(f)
        .lines()
        .map(|line| line.unwrap())
        .collect()
}
