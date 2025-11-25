use chrono::{DateTime, Local};
use crate::embedding::EmbeddingModel;
use crate::preprocessing::LogPreprocessor;
use anyhow::{Error as E, Result};
use dbscan::{Classification, Model};
use ndarray::{concatenate, Array1, Array2, Axis, s};
use ndarray_stats::DeviationExt;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Saves the centroids to a file in JSON format.
///
/// # Arguments
///
/// * `centroids` - A 2D array of centroids to save.
/// * `path` - The path to the file where the centroids will be saved.
fn save_centroids(centroids: &Array2<f32>, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, centroids)?;
    writer.flush()?;
    Ok(())
}

/// Processes a log file line by line, applying a preprocessor and a processor function.
///
/// # Arguments
///
/// * `path` - The path to the log file.
/// * `preprocessor` - The `LogPreprocessor` to apply to each line.
/// * `processor` - A closure that takes the original and preprocessed line and performs an action.
fn process_log_file<F>(path: &str, preprocessor: &LogPreprocessor, mut processor: F) -> Result<()>
where
    F: FnMut(String, String) -> Result<()>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let preprocessed = preprocessor.preprocess(&line);
        processor(line, preprocessed)?;
    }
    Ok(())
}

/// Trains the model on a log file to create initial centroids.
///
/// This function reads a log file in batches to avoid loading the entire file into memory.
/// It generates embeddings for each batch, concatenates them, and then uses DBSCAN
/// clustering to find patterns and create centroids.
///
/// # Arguments
///
/// * `input_file` - The path to the log file to train on.
/// * `output_file` - The path to save the centroids to.
/// * `epsilon` - The maximum distance between two points for one to be considered as in the neighborhood of the other.
/// * `min_points` - The minimum number of points required to form a dense region (a cluster).
/// * `preprocessor` - The `LogPreprocessor` to apply to each log message.
/// * `verbose` - A boolean flag to enable detailed logging.
pub fn train(input_file: &str, output_file: &str, epsilon: f32, min_points: usize, preprocessor: &LogPreprocessor, verbose: bool) -> Result<()> {
    let mut model = EmbeddingModel::load()?;
    
    const BATCH_SIZE: usize = 1024;
    let mut embedding_batches = Vec::new();

    println!("Reading and parsing log file in batches: {}", input_file);
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    let mut lines_iterator = reader.lines();
    
    loop {
        let mut batch_lines = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            if let Some(line_result) = lines_iterator.next() {
                batch_lines.push(line_result?);
            } else {
                break;
            }
        }

        if batch_lines.is_empty() {
            break;
        }

        let batch_preprocessed: Vec<String> = batch_lines.iter().map(|line| preprocessor.preprocess(line)).collect();
        let batch_str: Vec<&str> = batch_preprocessed.iter().map(|s| s.as_str()).collect();
        
        println!("Generating embeddings for batch of {} log messages...", batch_lines.len());
        let embeddings_tensor = model.embed(&batch_str)?;
        let (num_sentences, num_dims) = embeddings_tensor.dims2()?;
        let embeddings_vec: Vec<f32> = embeddings_tensor.flatten_all()?.to_vec1()?;
        let embeddings_array = Array2::from_shape_vec((num_sentences, num_dims), embeddings_vec)?;
        embedding_batches.push(embeddings_array);
    }

    if embedding_batches.is_empty() {
        println!("No log messages found in input file.");
        return Ok(());
    }

    let embeddings_array = concatenate(
        Axis(0),
        &embedding_batches.iter().map(|v| v.view()).collect::<Vec<_>>(),
    ).map_err(|e| E::msg(e.to_string()))?;

    let (_num_sentences, num_dims) = embeddings_array.dim();

    println!("Running DBSCAN clustering with epsilon={} and min_points={}...", epsilon, min_points);
    let dbscan = Model::new(epsilon as f64, min_points);
    let clusters = dbscan.run(&embeddings_array.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>());

    if verbose {
        println!("--- Cluster Assignments ---");
        let file = File::open(input_file)?;
        let reader = BufReader::new(file);
        for (i, line_result) in reader.lines().enumerate() {
            if i >= clusters.len() { break; }
            let line = line_result?;
            match clusters[i] {
                Classification::Noise => {
                    println!("'{}' -> Noise", line);
                }
                Classification::Core(id) | Classification::Edge(id) => {
                    println!("'{}' -> Cluster {}", line, id);
                }
            }
        }
        println!("-------------------------");
    }

    let mut cluster_map: HashMap<usize, Vec<Array1<f32>>> = HashMap::new();
    let mut noise_points = 0;

    for (i, &cluster_id) in clusters.iter().enumerate() {
        match cluster_id {
            Classification::Noise => noise_points += 1,
            Classification::Core(id) | Classification::Edge(id) => {
                cluster_map.entry(id).or_default().push(embeddings_array.row(i).to_owned());
            }
        }
    }

    if cluster_map.is_empty() {
        return Err(E::msg("DBSCAN did not find any clusters. Try adjusting epsilon or min_points."));
    }

    let mut centroids_list = Vec::new();
    for (_id, points) in cluster_map {
        let mut sum = Array1::zeros(num_dims);
        for p in &points {
            sum += p;
        }
        let mean = sum / points.len() as f32;
        centroids_list.push(mean.insert_axis(Axis(0)));
    }

    let centroids = concatenate(
        Axis(0),
        &centroids_list.iter().map(|v| v.view()).collect::<Vec<_>>(),
    ).map_err(|e| E::msg(e.to_string()))?;

    save_centroids(&centroids, output_file)?;

    println!("DBSCAN found {} clusters and {} noise points.", centroids.nrows(), noise_points);
    println!("Successfully saved {} centroids to {}", centroids.nrows(), output_file);

    Ok(())
}

/// Ingests a file of new logs, updating centroids for matches and logging non-matches.
/// It skips logs older than the centroids file and avoids reprocessing duplicate messages.
///
/// # Arguments
///
/// * `input_file` - The path to the file with new log messages.
/// * `centroids_file` - The path to the centroids file.
/// * `unmatched_file` - The path for saving unmatched logs.
/// * `threshold` - The distance threshold for matching a cluster.
/// * `learning_rate` - The learning rate for updating centroids on a match.
/// * `preprocessor` - The `LogPreprocessor` to apply to each log message.
/// * `_verbose` - A boolean flag to enable detailed logging (handled by the logger).
pub fn ingest(
    input_file: &str,
    centroids_file: &str,
    unmatched_file: &str,
    threshold: f64,
    learning_rate: f64,
    preprocessor: &LogPreprocessor,
    verbose: bool,
) -> Result<()> {
    let mut model = EmbeddingModel::load()?;

    println!("Loading centroids from {}...", centroids_file);
    let file = File::open(centroids_file)?;
    let mut centroids: Array2<f32> = serde_json::from_reader(file)?;

    let metadata = std::fs::metadata(centroids_file)?;
    let last_modified: DateTime<Local> = metadata.modified()?.into();

    println!("Reading and parsing new log file: {}", input_file);
    let mut unmatched_writer = BufWriter::new(
        OpenOptions::new().create(true).append(true).open(unmatched_file)?
    );
    let mut matched_count = 0;
    let mut total_count = 0;
    let mut seen_messages = HashSet::new();

    process_log_file(input_file, preprocessor, |original_line, preprocessed_message| {
        let log_timestamp_str = original_line.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
        let log_timestamp = if let Ok(parsed_time) = DateTime::parse_from_str(&format!("{} {}", log_timestamp_str, Local::now().format("%Y")), "%b %d %H:%M:%S %Y") {
            parsed_time.with_timezone(&Local)
        } else {
            // If parsing fails, default to now to process the line
            Local::now()
        };

        if log_timestamp < last_modified {
            return Ok(());
        }

        if !seen_messages.insert(preprocessed_message.clone()) {
            return Ok(());
        }

        total_count += 1;
        let message_embedding_tensor = model.embed(&[&preprocessed_message])?;
        let message_vec: Vec<f32> = message_embedding_tensor.flatten_all()?.to_vec1()?;
        let message_array = Array2::from_shape_vec((1, message_vec.len()), message_vec)?;
        let message_embedding = message_array.row(0);

        let mut min_dist = f64::INFINITY;
        let mut closest_cluster_index = 0;

        for (i, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
            let dist = centroid.l2_dist(&message_embedding)?;
            if dist < min_dist {
                min_dist = dist;
                closest_cluster_index = i;
            }
        }

        if min_dist < threshold {
            matched_count += 1;
            if verbose {
                println!("'{}' -> Match Cluster {} (distance: {:.4})", preprocessed_message, closest_cluster_index, min_dist);
            }
            let mut matched_centroid = centroids.slice_mut(s![closest_cluster_index, ..]);
            let update = &(&message_embedding - &matched_centroid) * learning_rate as f32;
            matched_centroid += &update;
        } else {
            if verbose {
                println!("'{}' -> No match (distance: {:.4})", preprocessed_message, min_dist);
            }
            writeln!(unmatched_writer, "{}", preprocessed_message)?;
        }
        Ok(())
    })?;

    println!("Ingestion complete.");
    println!("{} messages matched and updated centroids.", matched_count);
    println!("{} messages did not match and were written to {}.", total_count - matched_count, unmatched_file);

    save_centroids(&centroids, centroids_file)?;
    println!("Centroids file updated.");

    Ok(())
}

/// Retrains the model by creating new centroids from a log file.
///
/// This function is used to incorporate previously unmatched logs into the model.
///
/// # Arguments
///
/// * `input_file` - The path to the log file to create new centroids from.
/// * `centroids_file` - The path to the centroids file to update.
/// * `preprocessor` - The `LogPreprocessor` to apply to each log message.
pub fn retrain(input_file: &str, centroids_file: &str, preprocessor: &LogPreprocessor, verbose: bool) -> Result<()> {
    let mut model = EmbeddingModel::load()?;

    println!("Loading existing centroids from {}...", centroids_file);
    let file = File::open(centroids_file)?;
    let centroids: Array2<f32> = serde_json::from_reader(file)?;

    println!("Reading and parsing new training data from {}", input_file);
    let mut sentences = Vec::new();
    process_log_file(input_file, preprocessor, |_original_line, preprocessed_message| {
        if verbose {
            println!("Adding new centroid from: '{}'", preprocessed_message);
        }
        sentences.push(preprocessed_message);
        Ok(())
    })?;

    if sentences.is_empty() {
        println!("Input file is empty. No new centroids to add.");
        return Ok(());
    }
    
    let sentences_str: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();

    println!("Generating embeddings for {} new log messages...", sentences.len());
    let embeddings_tensor = model.embed(&sentences_str)?;
    let (num_sentences, num_dims) = embeddings_tensor.dims2()?;
    let embeddings_vec: Vec<f32> = embeddings_tensor.flatten_all()?.to_vec1()?;
    let new_centroids_array = Array2::from_shape_vec((num_sentences, num_dims), embeddings_vec)?;

    let updated_centroids = concatenate(Axis(0), &[centroids.view(), new_centroids_array.view()])
        .map_err(|e| E::msg(e.to_string()))?;
    
    save_centroids(&updated_centroids, centroids_file)?;

    println!("Successfully added {} new centroids. Total centroids: {}", new_centroids_array.nrows(), updated_centroids.nrows());

    Ok(())
}

/// Tests the regex patterns on a log file.
///
/// This function is a utility to help with debugging and refining the regex patterns.
/// It logs the original and preprocessed versions of each line in a log file.
///
/// # Arguments
///
/// * `input_file` - The path to the log file to test patterns on.
/// * `preprocessor` - The `LogPreprocessor` to apply to each log message.
pub fn test_patterns(input_file: &str, preprocessor: &LogPreprocessor) -> Result<()> {
    println!("Testing patterns on log file: {}", input_file);
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let preprocessed = preprocessor.preprocess(&line);
        println!("Original:  '{}'", line);
        println!("Processed: '{}'\n", preprocessed);
    }
    Ok(())
}
