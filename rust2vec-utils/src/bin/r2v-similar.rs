use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::{App, AppSettings, Arg, ArgMatches};
use rust2vec::{
    io::ReadEmbeddings, similarity::Similarity, text::ReadText, text::ReadTextDims,
    word2vec::ReadWord2Vec, Embeddings,
};
use rust2vec_utils::EmbeddingFormat;
use stdinout::{Input, OrExit};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

fn parse_args() -> ArgMatches<'static> {
    App::new("r2v-similar")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("format")
                .short("f")
                .value_name("FORMAT")
                .help("Embedding format: rust2vec, word2vec, text, or textdims (default: rust2vec)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("neighbors")
                .short("k")
                .value_name("K")
                .help("Return K nearest neighbors (default: 10)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("EMBEDDINGS")
                .help("Embeddings file")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name("INPUT").help("Input words").index(2))
        .get_matches()
}

struct Config {
    embeddings_filename: String,
    embedding_format: EmbeddingFormat,
    k: usize,
}

fn config_from_matches<'a>(matches: &ArgMatches<'a>) -> Config {
    let embeddings_filename = matches.value_of("EMBEDDINGS").unwrap().to_owned();

    let embedding_format = matches
        .value_of("format")
        .map(|f| EmbeddingFormat::try_from(f).or_exit("Cannot parse embedding format", 1))
        .unwrap_or(EmbeddingFormat::Rust2Vec);

    let k = matches
        .value_of("neighbors")
        .map(|v| v.parse().or_exit("Cannot parse k", 1))
        .unwrap_or(10);

    Config {
        embeddings_filename,
        embedding_format,
        k,
    }
}

fn read_embeddings(filename: &str, embedding_format: EmbeddingFormat) -> Embeddings {
    let f = File::open(filename).or_exit("Cannot open embeddings file", 1);
    let mut reader = BufReader::new(f);

    use EmbeddingFormat::*;
    match embedding_format {
        Rust2Vec => ReadEmbeddings::read_embeddings(&mut reader),
        Word2Vec => ReadWord2Vec::read_word2vec_binary(&mut reader, true),
        Text => ReadText::read_text(&mut reader, true),
        TextDims => ReadTextDims::read_text_dims(&mut reader, true),
    }
    .or_exit("Cannot read embeddings", 1)
}

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let embeddings = read_embeddings(&config.embeddings_filename, config.embedding_format);

    let input = Input::from(matches.value_of("INPUT"));
    let reader = input.buf_read().or_exit("Cannot open input for reading", 1);

    for line in reader.lines() {
        let line = line.or_exit("Cannot read line", 1).trim().to_owned();
        if line.is_empty() {
            continue;
        }

        let results = match embeddings.similarity(&line, config.k) {
            Some(results) => results,
            None => continue,
        };

        for similar in results {
            println!("{}\t{}", similar.word, similar.similarity);
        }
    }
}
