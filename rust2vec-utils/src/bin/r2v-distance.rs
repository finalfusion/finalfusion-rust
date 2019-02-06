use std::env::args;
use std::fmt;
use std::fs::File;
use std::io::{stdin, BufRead, BufReader};
use std::process;

use getopts::Options;
use rust2vec::{
    similarity::Similarity, storage::NdArray, vocab::SimpleVocab, word2vec::ReadWord2Vec,
    Embeddings,
};

pub fn or_exit<T, E: fmt::Display>(r: Result<T, E>) -> T {
    r.unwrap_or_else(|e: E| -> T {
        println!("Error: {}", e);
        process::exit(1)
    })
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] EMBEDDINGS", program);
    print!("{}", opts.usage(&brief));
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = or_exit(opts.parse(&args[1..]));

    if matches.opt_present("h") {
        print_usage(&program, opts);
        process::exit(1)
    }

    if matches.free.is_empty() || matches.free.len() > 1 {
        print_usage(&program, opts);
        process::exit(1)
    }

    let embeddings = read_embeddings(&matches.free[0]);

    let stdin = stdin();
    for line in stdin.lock().lines() {
        match embeddings.similarity(&line.unwrap(), 10) {
            Some(results) => {
                for result in results {
                    println!("{}\t{}", result.word, result.similarity);
                }
            }
            None => (),
        }
    }
}

fn read_embeddings(filename: &str) -> Embeddings<SimpleVocab, NdArray> {
    let f = or_exit(File::open(filename));
    let mut reader = BufReader::new(f);
    let mut embeds = or_exit(Embeddings::read_word2vec_binary(&mut reader));
    embeds.normalize();
    embeds
}
