use std::fs::File;
use std::io::{BufReader, BufWriter};

use clap::{App, AppSettings, Arg, ArgMatches};
use rust2vec::{
    io::{ReadEmbeddings, WriteEmbeddings},
    text::{ReadText, ReadTextDims, WriteText, WriteTextDims},
    word2vec::{ReadWord2Vec, WriteWord2Vec},
    Embeddings,
};
use rust2vec_utils::EmbeddingFormat;
use stdinout::OrExit;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

struct Config {
    input_filename: String,
    output_filename: String,
    input_format: EmbeddingFormat,
    output_format: EmbeddingFormat,
}

// Option constants
static INPUT_FORMAT: &str = "input_format";
static OUTPUT_FORMAT: &str = "output_format";

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

fn parse_args() -> ArgMatches<'static> {
    App::new("r2v-convert")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(INPUT)
                .help("Finalfrontier model")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name(OUTPUT).help("Output file").index(2))
        .arg(
            Arg::with_name(INPUT_FORMAT)
                .short("f")
                .long("from")
                .value_name("FORMAT")
                .help("Input format: rust2vec, text, textdims, word2vec (default: word2vec)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(OUTPUT_FORMAT)
                .short("t")
                .long("to")
                .value_name("FORMAT")
                .help("Output format: rust2vec, text, textdims, word2vec (default: rust2vec)")
                .takes_value(true),
        )
        .get_matches()
}

fn config_from_matches(matches: &ArgMatches) -> Config {
    let input_filename = matches.value_of(INPUT).unwrap().to_owned();
    let input_format = matches
        .value_of(INPUT_FORMAT)
        .map(|v| EmbeddingFormat::try_from(v).or_exit("Cannot parse input format", 1))
        .unwrap_or(EmbeddingFormat::Word2Vec);
    let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();
    let output_format = matches
        .value_of(OUTPUT_FORMAT)
        .map(|v| EmbeddingFormat::try_from(v).or_exit("Cannot parse output format", 1))
        .unwrap_or(EmbeddingFormat::Rust2Vec);

    Config {
        input_filename,
        output_filename,
        input_format,
        output_format,
    }
}

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let embeddings = read_embeddings(&config.input_filename, config.input_format);
    write_embeddings(embeddings, &config.output_filename, config.output_format);
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

fn write_embeddings(embeddings: Embeddings, filename: &str, embedding_format: EmbeddingFormat) {
    let f = File::create(filename).or_exit("Cannot create embeddings file", 1);
    let mut writer = BufWriter::new(f);

    use EmbeddingFormat::*;
    match embedding_format {
        Rust2Vec => embeddings.write_embeddings(&mut writer),
        Word2Vec => embeddings.write_word2vec_binary(&mut writer),
        Text => embeddings.write_text(&mut writer),
        TextDims => embeddings.write_text_dims(&mut writer),
    }
    .or_exit("Cannot write embeddings", 1)
}
