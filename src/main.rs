use anyhow::Result;
use clap::{Parser, Subcommand};
use railog::commands::{ingest, retrain, test_patterns, train};
use railog::preprocessing::LogPreprocessor;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Path to the regex patterns file
    #[arg(short, long, global = true, default_value = "patterns.txt")]
    patterns_file: String,
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model on a log file to create initial centroids
    Train {
        /// Path to the log file to train on
        #[arg(short, long, default_value = "example.txt")]
        input_file: String,
        /// Path to save the centroids to
        #[arg(short, long, default_value = "centroids.json")]
        output_file: String,
        /// The maximum distance between two points for one to be considered as in the neighborhood of the other.
        #[arg(short, long, default_value_t = 0.5)]
        epsilon: f32,
        /// The minimum number of points required to form a dense region (a cluster).
        #[arg(short, long, default_value_t = 3)]
        min_points: usize,
    },
    /// Ingest a file of new logs, updating centroids for matches and logging non-matches
    Ingest {
        /// Path to the file with new log messages
        #[arg(short, long, default_value = "new_logs.txt")]
        input_file: String,
        /// Path to the centroids file
        #[arg(short, long, default_value = "centroids.json")]
        centroids_file: String,
        /// Path for saving unmatched logs
        #[arg(short, long, default_value = "unmatched.log")]
        unmatched_file: String,
        /// Distance threshold for matching a cluster.
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f64,
        /// Learning rate for updating centroids on a match.
        #[arg(short, long, default_value_t = 0.1)]
        learning_rate: f64,
    },
    /// Retrain the model by creating new centroids from a log file
    Retrain {
        /// Path to the log file to create new centroids from
        #[arg(short, long, default_value = "unmatched.log")]
        input_file: String,
        /// Path to the centroids file to update
        #[arg(short, long, default_value = "centroids.json")]
        centroids_file: String,
    },
    /// Test the regex patterns on a log file
    TestPatterns {
        /// Path to the log file to test patterns on
        #[arg(short, long, default_value = "new_logs.txt")]
        input_file: String,
    },
}

/// The main entry point for the application.
///
/// This function parses the command-line arguments and calls the appropriate subcommand.
fn main() -> Result<()> {
    let cli = Cli::parse();

    env_logger::Builder::new()
        .filter_level(if cli.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    let preprocessor = LogPreprocessor::new(&cli.patterns_file)?;
    match &cli.command {
        Commands::Train {
            input_file,
            output_file,
            epsilon,
            min_points,
        } => {
            train(
                input_file,
                output_file,
                *epsilon,
                *min_points,
                &preprocessor,
                cli.verbose,
            )?;
        }
        Commands::Ingest {
            input_file,
            centroids_file,
            unmatched_file,
            threshold,
            learning_rate,
        } => {
            ingest(
                input_file,
                centroids_file,
                unmatched_file,
                *threshold,
                *learning_rate,
                &preprocessor,
                cli.verbose,
            )?;
        }
        Commands::Retrain {
            input_file,
            centroids_file,
        } => {
            retrain(input_file, centroids_file, &preprocessor, cli.verbose)?;
        }
        Commands::TestPatterns { input_file } => {
            test_patterns(input_file, &preprocessor)?;
        }
    }
    Ok(())
}
