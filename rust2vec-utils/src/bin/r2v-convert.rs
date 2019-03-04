use std::fs::File;
use std::io::{BufReader, BufWriter, Read};

use clap::{App, AppSettings, Arg, ArgMatches};
use failure::err_msg;
use rust2vec::prelude::*;
use rust2vec_utils::EmbeddingFormat;
use stdinout::OrExit;
use toml::Value;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

struct Config {
    input_filename: String,
    output_filename: String,
    metadata_filename: Option<String>,
    input_format: EmbeddingFormat,
    output_format: EmbeddingFormat,
    normalization: bool,
}

// Option constants
static INPUT_FORMAT: &str = "input_format";
static METADATA_FILENAME: &str = "metadata_filename";
static NO_NORMALIZATION: &str = "no_normalization";
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
                .help("Input format: finalfusion, text, textdims, word2vec (default: word2vec)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(METADATA_FILENAME)
                .short("m")
                .long("metadata")
                .value_name("FILENAME")
                .help("TOML metadata add to the embeddings")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(NO_NORMALIZATION)
                .short("n")
                .long("no-normalization")
                .help("Do not normalize embeddings during conversion."),
        )
        .arg(
            Arg::with_name(OUTPUT_FORMAT)
                .short("t")
                .long("to")
                .value_name("FORMAT")
                .help("Output format: finalfusion, text, textdims, word2vec (default: finalfusion)")
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
        .unwrap_or(EmbeddingFormat::FinalFusion);

    let metadata_filename = matches.value_of(METADATA_FILENAME).map(ToOwned::to_owned);

    let normalization = !matches.is_present(NO_NORMALIZATION);

    Config {
        input_filename,
        output_filename,
        input_format,
        output_format,
        metadata_filename,
        normalization,
    }
}

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let metadata = config.metadata_filename.map(read_metadata).map(Metadata);

    let mut embeddings = read_embeddings(
        &config.input_filename,
        config.input_format,
        config.normalization,
    );

    // Overwrite metadata if provided, otherwise retain existing metadata.
    if metadata.is_some() {
        embeddings.set_metadata(metadata);
    }

    write_embeddings(embeddings, &config.output_filename, config.output_format);
}

fn read_metadata(filename: impl AsRef<str>) -> Value {
    let f = File::open(filename.as_ref()).or_exit("Cannot open metadata file", 1);
    let mut reader = BufReader::new(f);
    let mut buf = String::new();
    reader
        .read_to_string(&mut buf)
        .or_exit("Cannot read metadata", 1);
    buf.parse::<Value>()
        .or_exit("Cannot parse metadata TOML", 1)
}

fn read_embeddings(
    filename: &str,
    embedding_format: EmbeddingFormat,
    normalization: bool,
) -> Embeddings<VocabWrap, StorageWrap> {
    let f = File::open(filename).or_exit("Cannot open embeddings file", 1);
    let mut reader = BufReader::new(f);

    use EmbeddingFormat::*;
    match embedding_format {
        FinalFusion => ReadEmbeddings::read_embeddings(&mut reader),
        FinalFusionMmap => MmapEmbeddings::mmap_embeddings(&mut reader),
        Word2Vec => {
            ReadWord2Vec::read_word2vec_binary(&mut reader, normalization).map(Embeddings::into)
        }
        Text => ReadText::read_text(&mut reader, normalization).map(Embeddings::into),
        TextDims => ReadTextDims::read_text_dims(&mut reader, normalization).map(Embeddings::into),
    }
    .or_exit("Cannot read embeddings", 1)
}

fn write_embeddings(
    embeddings: Embeddings<VocabWrap, StorageWrap>,
    filename: &str,
    embedding_format: EmbeddingFormat,
) {
    let f = File::create(filename).or_exit("Cannot create embeddings file", 1);
    let mut writer = BufWriter::new(f);

    use EmbeddingFormat::*;
    match embedding_format {
        FinalFusion => embeddings.write_embeddings(&mut writer),
        FinalFusionMmap => Err(err_msg("Writing to this format is not supported")),
        Word2Vec => embeddings.write_word2vec_binary(&mut writer),
        Text => embeddings.write_text(&mut writer),
        TextDims => embeddings.write_text_dims(&mut writer),
    }
    .or_exit("Cannot write embeddings", 1)
}
