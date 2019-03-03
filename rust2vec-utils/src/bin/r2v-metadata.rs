use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use clap::{App, AppSettings, Arg, ArgMatches};
use rust2vec::prelude::*;
use stdinout::{OrExit, Output};
use toml::ser::to_string_pretty;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

struct Config {
    input_filename: String,
    output_filename: Option<String>,
}

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

fn parse_args() -> ArgMatches<'static> {
    App::new("r2v-metadata")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(INPUT)
                .help("finalfusion model")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name(OUTPUT).help("Output file").index(2))
        .get_matches()
}

fn config_from_matches(matches: &ArgMatches) -> Config {
    let input_filename = matches.value_of(INPUT).unwrap().to_owned();
    let output_filename = matches.value_of(OUTPUT).map(ToOwned::to_owned);

    Config {
        input_filename,
        output_filename,
    }
}

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let output = Output::from(config.output_filename);
    let mut writer = BufWriter::new(output.write().or_exit("Cannot open output for writing", 1));

    if let Some(metadata) = read_metadata(&config.input_filename) {
        writer
            .write_all(
                to_string_pretty(&metadata.0)
                    .or_exit("Cannot serialize metadata to TOML", 1)
                    .as_bytes(),
            )
            .or_exit("Cannot write metadata", 1);
    }
}

fn read_metadata(filename: &str) -> Option<Metadata> {
    let f = File::open(filename).or_exit("Cannot open embeddings file", 1);
    let mut reader = BufReader::new(f);
    ReadMetadata::read_metadata(&mut reader).or_exit("Cannot read metadata", 1)
}
