//! Per-frame index written next to every recorded segment.
//!
//! One JSON object per line, in arrival order, so a trainer can align learner
//! steps with encoder input without decoding the video:
//!
//! ```text
//! {"frame":0,"timestamp_qpc":132141,"timestamp_utc_ms":1758163500000}
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Append-only writer for a `frames.jsonl` index.
#[derive(Debug)]
pub struct FrameIndexWriter {
    writer: BufWriter<File>,
}

impl FrameIndexWriter {
    /// Creates (or truncates) the index file.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`std::io::Error`] when the file cannot be created.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Appends one frame record.
    ///
    /// `timestamp_qpc` is the capture timestamp in 100 ns ticks on the same
    /// clock as `QueryPerformanceCounter`; `timestamp_utc_ms` is wall clock time.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`std::io::Error`] when the write fails.
    pub fn write(
        &mut self,
        frame: u64,
        timestamp_qpc: i64,
        timestamp_utc_ms: i64,
    ) -> std::io::Result<()> {
        writeln!(
            self.writer,
            "{{\"frame\":{frame},\"timestamp_qpc\":{timestamp_qpc},\"timestamp_utc_ms\":{timestamp_utc_ms}}}"
        )
    }

    /// Flushes buffered records to disk.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`std::io::Error`] when flushing fails.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::FrameIndexWriter;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn writes_one_line_per_frame() {
        let path: PathBuf = std::env::temp_dir().join(format!(
            "glr-recording-index-{:?}.jsonl",
            std::process::id()
        ));
        let mut writer = FrameIndexWriter::create(&path).expect("index file");
        writer
            .write(0, 100, 1_700_000_000_000)
            .expect("first frame");
        writer
            .write(1, 433, 1_700_000_000_033)
            .expect("second frame");
        writer.flush().expect("flush");
        drop(writer);
        let text = fs::read_to_string(&path).expect("read back");
        assert_eq!(
            text,
            "{\"frame\":0,\"timestamp_qpc\":100,\"timestamp_utc_ms\":1700000000000}\n\
             {\"frame\":1,\"timestamp_qpc\":433,\"timestamp_utc_ms\":1700000000033}\n"
        );
        let _ = fs::remove_file(&path);
    }
}
