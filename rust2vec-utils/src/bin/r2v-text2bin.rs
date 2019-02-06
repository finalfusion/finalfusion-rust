use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::process;

use getopts::Options;
use rust2vec::{text::ReadText, word2vec::WriteWord2Vec, Embeddings};
use stdinout::OrExit;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] input output", program);
    print!("{}", opts.usage(&brief));
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse arguments", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        process::exit(1)
    }

    if matches.free.len() != 2 {
        print_usage(&program, opts);
        process::exit(1)
    }

    let mut reader = BufReader::new(
        File::open(&matches.free[0])
            .or_exit(format!("Cannot read embedding file {}", matches.free[0]), 1),
    );
    let embeddings = Embeddings::read_text(&mut reader).or_exit("Cannot read embeddings", 1);

    let mut writer = BufWriter::new(File::create(&matches.free[1]).or_exit(
        format!("Cannot write embedding file {}", matches.free[1]),
        1,
    ));
    embeddings
        .write_word2vec_binary(&mut writer)
        .or_exit("Cannot write embeddings", 1);
}
