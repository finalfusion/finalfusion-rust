use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::{Arc, Mutex};

use clap::{App, AppSettings, Arg, ArgMatches};
use finalfusion::prelude::*;
use finalfusion::similarity::Analogy;
use finalfusion_utils::{read_embeddings_view, EmbeddingFormat};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use stdinout::{Input, OrExit};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    ThreadPoolBuilder::new()
        .num_threads(config.n_threads)
        .build_global()
        .unwrap();

    let embeddings =
        read_embeddings_view(&config.embeddings_filename, EmbeddingFormat::FinalFusion)
            .or_exit("Cannot read embeddings", 1);

    let analogies_file = Input::from(config.analogies_filename);
    let reader = analogies_file
        .buf_read()
        .or_exit("Cannot open analogy file for reading", 1);

    let instances = read_analogies(reader);
    process_analogies(&embeddings, &instances);
}

// Option constants
static EMBEDDINGS: &str = "EMBEDDINGS";
static ANALOGIES: &str = "ANALOGIES";
static THREADS: &str = "threads";

fn parse_args() -> ArgMatches<'static> {
    App::new("ff-compute-accuracy")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(THREADS)
                .long("threads")
                .value_name("N")
                .help("Number of threads (default: logical_cpus / 2)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(EMBEDDINGS)
                .help("Embedding file")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name(ANALOGIES).help("Analogy file").index(2))
        .get_matches()
}

struct Config {
    analogies_filename: Option<String>,
    embeddings_filename: String,
    n_threads: usize,
}

fn config_from_matches(matches: &ArgMatches) -> Config {
    let embeddings_filename = matches.value_of(EMBEDDINGS).unwrap().to_owned();
    let analogies_filename = matches.value_of(ANALOGIES).map(ToOwned::to_owned);
    let n_threads = matches
        .value_of("threads")
        .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
        .unwrap_or(num_cpus::get() / 2);

    Config {
        analogies_filename,
        embeddings_filename,
        n_threads,
    }
}

struct Counts {
    n_correct: usize,
    n_instances: usize,
    n_skipped: usize,
}

impl Default for Counts {
    fn default() -> Self {
        Counts {
            n_correct: 0,
            n_instances: 0,
            n_skipped: 0,
        }
    }
}

#[derive(Clone)]
struct Eval<'a> {
    embeddings: &'a Embeddings<VocabWrap, StorageViewWrap>,
    section_counts: Arc<Mutex<BTreeMap<String, Counts>>>,
}

impl<'a> Eval<'a> {
    fn new(embeddings: &'a Embeddings<VocabWrap, StorageViewWrap>) -> Self {
        Eval {
            embeddings,
            section_counts: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Evaluate an analogy.
    fn eval_analogy(&self, instance: &Instance) {
        // Skip instances where the to-be-predicted word is not in the
        // vocab. This is a shortcoming of the vocab size and not of the
        // embedding model itself.
        if self.embeddings.vocab().idx(&instance.answer).is_none() {
            let mut section_counts = self.section_counts.lock().unwrap();
            let counts = section_counts.entry(instance.section.clone()).or_default();
            counts.n_skipped += 1;
            return;
        }

        // If the model is not able to provide a query result, it is counted
        // as an error.
        let is_correct = self
            .embeddings
            .analogy(&instance.query.0, &instance.query.1, &instance.query.2, 1)
            .map(|r| r.first().unwrap().word == instance.answer)
            .unwrap_or(false);

        let mut section_counts = self.section_counts.lock().unwrap();
        let counts = section_counts.entry(instance.section.clone()).or_default();
        counts.n_instances += 1;
        if is_correct {
            counts.n_correct += 1;
        }
    }

    /// Print the accuracy for a section.
    fn print_section_accuracy(&self, section: &str, counts: &Counts) {
        if counts.n_instances == 0 {
            eprintln!("{}: no evaluation instances", section);
            return;
        }

        println!(
            "{}: {}/{} correct, accuracy: {:.2}, skipped: {}",
            section,
            counts.n_correct,
            counts.n_instances,
            (counts.n_correct as f64 / counts.n_instances as f64) * 100.,
            counts.n_skipped,
        );
    }
}

impl<'a> Drop for Eval<'a> {
    fn drop(&mut self) {
        let section_counts = self.section_counts.lock().unwrap();

        // Print out counts for all sections.
        for (section, counts) in section_counts.iter() {
            self.print_section_accuracy(section, counts);
        }

        let n_correct = section_counts.values().map(|c| c.n_correct).sum::<usize>();
        let n_instances = section_counts
            .values()
            .map(|c| c.n_instances)
            .sum::<usize>();
        let n_skipped = section_counts.values().map(|c| c.n_skipped).sum::<usize>();
        let n_instances_with_skipped = n_instances + n_skipped;

        // Print out overall counts.
        println!(
            "Total: {}/{} correct, accuracy: {:.2}",
            n_correct,
            n_instances,
            (n_correct as f64 / n_instances as f64) * 100.
        );

        // Print skip counts.
        println!(
            "Skipped: {}/{} ({}%)",
            n_skipped,
            n_instances_with_skipped,
            (n_skipped as f64 / n_instances_with_skipped as f64) * 100.
        );
    }
}

struct Instance {
    section: String,
    query: (String, String, String),
    answer: String,
}

fn read_analogies(reader: impl BufRead) -> Vec<Instance> {
    let mut section = String::new();

    let mut instances = Vec::new();

    for line in reader.lines() {
        let line = line.or_exit("Cannot read line.", 1);

        if line.starts_with(": ") {
            section = line.chars().skip(2).collect::<String>();
            continue;
        }

        let quadruple: Vec<_> = line.split_whitespace().collect();

        instances.push(Instance {
            section: section.clone(),
            query: (
                quadruple[0].to_owned(),
                quadruple[1].to_owned(),
                quadruple[2].to_owned(),
            ),
            answer: quadruple[3].to_owned(),
        });
    }

    instances
}

fn process_analogies(embeddings: &Embeddings<VocabWrap, StorageViewWrap>, instances: &[Instance]) {
    let eval = Eval::new(&embeddings);
    instances
        .par_iter()
        .for_each(|instance| eval.eval_analogy(instance));
}
