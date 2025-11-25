# Railog: AI-Powered Log Pattern Analysis

Railog is a Rust-based command-line tool that uses machine learning to analyze log files, identify patterns, and classify new log messages. It transforms unstructured log messages into numerical vectors (embeddings) and groups them into clusters based on semantic similarity. This allows the system to learn the "normal" patterns in your logs and identify new, potentially interesting messages that don't fit known patterns.

The tool is designed with an online learning workflow in mind, allowing the model to adapt and improve as it processes more data over time.

## Key Features

- **Log Message Embedding**: Utilizes the `sentence-transformers/all-MiniLM-L6-v2` model to convert log messages into 384-dimensional vectors.
    - **Pattern Discovery via Clustering**: Employs DBSCAN clustering to group similar log vectors, effectively identifying distinct log patterns.- **Configurable Preprocessing**: Uses a customizable text file (`patterns.txt`) of regular expressions to normalize log messages before analysis (e.g., replacing PIDs, IP addresses with generic tokens like `<PID>` and `<IP>`).
- **Online Learning Workflow**:
    - **Train**: Create a baseline model of log patterns from a sample file.
    - **Ingest**: Process new logs, automatically updating the model for known patterns and separating unknown ones for review.
    - **Retrain**: Incorporate reviewed, previously unknown logs into the model by creating new clusters.
- **Command-Line Interface**: A simple and powerful CLI for managing the entire workflow.

## Core Workflow

The intended workflow allows the system to continuously learn and adapt to your log data.

1.  **Initial Training**:
    -   Start with a large, representative log file (e.g., `example.txt`).
    -   Run the `train` command to analyze this file and create an initial `centroids.json` file, which stores the mathematical centers of the identified log patterns.

2.  **Ongoing Ingestion**:
    -   As new logs are generated, collect them into a file (e.g., `new_logs.txt`).
    -   Run the `ingest` command. The tool will:
        -   Update the existing centroids for logs that match known patterns.
        -   Write any non-matching logs to an `unmatched.log` file.

3.  **Manual Review & Retraining**:
    -   Periodically, a human operator should review the `unmatched.log` file. This file contains logs that the system considers novel.
    -   After validating that these logs represent new, valid patterns, run the `retrain` command on `unmatched.log`. This will create new centroids for these patterns and add them to the model.

This cycle of ingesting, reviewing, and retraining allows the model to evolve without requiring a full, costly retraining from scratch.

## Usage

First, build the project using Cargo:
```bash
cargo build --release
```
The executable will be located at `target/release/railog`.

### 1. `train`
Creates the initial `centroids.json` file from a sample log file.

```bash
./target/release/railog train --input-file <path_to_your_logs.txt> --epsilon 0.5 --min-points 2
```
-   `--input-file` (`-i`): The log file to train on. Defaults to `example.txt`.
-   `--output-file` (`-o`): The file to save centroids to. Defaults to `centroids.json`.
-   `--epsilon` (`-e`): The maximum distance between two points for one to be considered as in the neighborhood of the other. Defaults to `0.5`.
-   `--min-points` (`-m`): The minimum number of points required to form a dense region (a cluster). Defaults to `2`.

### 2. `ingest`
Processes a file of new logs, updating centroids and separating non-matches.

```bash
./target/release/railog ingest --input-file new_logs.txt --threshold 0.5
```
-   `--input-file` (`-i`): The file containing new logs. Defaults to `new_logs.txt`.
-   `--centroids-file` (`-c`): The centroids model file. Defaults to `centroids.json`.
-   `--unmatched-file` (`-u`): The file to write non-matching logs to. Defaults to `unmatched.log`.
-   `--threshold` (`-t`): The distance threshold for considering a log a "match". Lower is stricter. Defaults to `1.0`.
-   `--learning-rate` (`-l`): The rate at which a matching log influences a cluster's centroid. Defaults to `0.1`.

### 3. `retrain`
Creates new centroids from a file of (typically unmatched) logs and adds them to the model.

```bash
./target/release/railog retrain --input-file unmatched.log
```
-   `--input-file` (`-i`): The log file to create new centroids from. Defaults to `unmatched.log`.
-   `--centroids-file` (`-c`): The centroids model file to update. Defaults to `centroids.json`.

### 4. `test-patterns`
A utility command to test your regex patterns on a file without performing any analysis. It prints the original and processed versions of each line.

```bash
./target/release/railog test-patterns --input-file new_logs.txt
```
-   `--input-file` (`-i`): The log file to test patterns on. Defaults to `new_logs.txt`.
-   `--patterns-file` (`-p`): A global flag to specify the location of your patterns file. Defaults to `patterns.txt`.

## Preprocessing with `patterns.txt`

To improve accuracy, Railog preprocesses each log message to normalize dynamic or high-variance tokens. The patterns for this are defined in `patterns.txt`.

The format is one pattern per line:
```
<REGEX> :: <REPLACEMENT>
```

**Example `patterns.txt`:**
```
# Each line should be in the format: regex :: replacement
# Note: Use raw string syntax for regex if needed in your language, but here just use standard regex.

\[\d+\]: :: [<PID>]:
\b(?:\d{1,3}\.){3}\d{1,3}\b :: <IP>
```
This file will replace process IDs like `[12345]:` with `[<PID>]:` and any IPv4 address with `<IP>`, allowing the model to learn the general pattern rather than the specific noisy data.
