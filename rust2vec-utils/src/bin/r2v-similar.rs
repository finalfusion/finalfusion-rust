use std::io::BufRead;

use clap::{App, AppSettings, Arg, ArgMatches};
use rust2vec::similarity::Similarity;
use rust2vec_utils::{read_embeddings_view, EmbeddingFormat};
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
                .help("Embedding format: finalfusion, finalfusion_mmap, word2vec, text, or textdims (default: finalfusion)")
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
        .unwrap_or(EmbeddingFormat::FinalFusion);

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

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let embeddings = read_embeddings_view(&config.embeddings_filename, config.embedding_format)
        .or_exit("Cannot read embeddings", 1);

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
