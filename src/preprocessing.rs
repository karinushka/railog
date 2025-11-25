use anyhow::Result;
use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// A preprocessor for log messages that applies a series of regex patterns to normalize the text.
pub struct LogPreprocessor {
    patterns: Vec<(Regex, String)>,
}

impl LogPreprocessor {
    /// Creates a new `LogPreprocessor` from a file of regex patterns.
    ///
    /// Each line in the patterns file should be in the format: `regex :: replacement`.
    /// Lines starting with `#` or empty lines are ignored.
    ///
    /// # Arguments
    ///
    /// * `patterns_file` - The path to the file containing the regex patterns.
    pub fn new(patterns_file: &str) -> Result<Self> {
        let file = File::open(patterns_file)?;
        let reader = BufReader::new(file);
        let mut patterns = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, " :: ").collect();
            if parts.len() == 2 {
                let re = Regex::new(parts[0])?;
                patterns.push((re, parts[1].to_string()));
            }
        }
        Ok(Self { patterns })
    }

    /// Applies the loaded regex patterns to a single log message.
    ///
    /// # Arguments
    ///
    /// * `message` - The log message to preprocess.
    pub fn preprocess(&self, message: &str) -> String {
        let mut processed_message = message.to_string();
        for (re, replacement) in &self.patterns {
            processed_message = re.replace_all(&processed_message, replacement).to_string();
        }
        processed_message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_preprocess() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, r"\[\d+\]: :: [<PID>]:")?;
        writeln!(file, r"\b(?:\d{{1,3}}\.){{3}}\d{{1,3}}\b :: <IP>")?;
        let path = file.path().to_str().unwrap();

        let preprocessor = LogPreprocessor::new(path)?;

        let message1 = "sshd[12345]: Accepted publickey for user from 192.168.1.1 port 22";
        let expected1 = "sshd[<PID>]: Accepted publickey for user from <IP> port 22";
        assert_eq!(preprocessor.preprocess(message1), expected1);

        let message2 = "kernel: [67890]: a message";
        let expected2 = "kernel: [<PID>]: a message";
        assert_eq!(preprocessor.preprocess(message2), expected2);

        Ok(())
    }
}