use std::io::{self, BufReader, BufWriter};

use glr_host::{DEFAULT_MAX_FRAME_BYTES, Host, SyntheticCounterProvider, serve_json_lines};

struct Cli {
    provider: String,
    max_frame_bytes: usize,
}

fn main() {
    if let Err(message) = run() {
        eprintln!("glr-hostd: {message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let cli = parse_args(std::env::args().skip(1))?;
    if cli.provider != "synthetic-counter" {
        return Err(format!(
            "unsupported provider {:?}; first release supports only synthetic-counter",
            cli.provider
        ));
    }
    let mut host = Host::with_max_frame_bytes(
        Box::new(SyntheticCounterProvider::new(4)),
        cli.max_frame_bytes,
    )
    .map_err(|error| error.to_string())?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    serve_json_lines(
        &mut host,
        &mut BufReader::new(stdin.lock()),
        &mut BufWriter::new(stdout.lock()),
    )
    .map_err(|error| format!("stdio transport failed: {error}"))
}

fn parse_args(arguments: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut provider = None;
    let mut max_frame_bytes = DEFAULT_MAX_FRAME_BYTES;
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--provider" => {
                provider = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--provider requires a value".to_string())?,
                );
            }
            "--transport" => {
                let transport = arguments
                    .next()
                    .ok_or_else(|| "--transport requires a value".to_string())?;
                if transport != "stdio" {
                    return Err("first release supports only --transport stdio".into());
                }
            }
            "--max-frame-bytes" => {
                max_frame_bytes = arguments
                    .next()
                    .ok_or_else(|| "--max-frame-bytes requires a value".to_string())?
                    .parse()
                    .map_err(|_| "--max-frame-bytes must be an integer".to_string())?;
            }
            "--version" => {
                println!("glr-hostd {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                println!(
                    "glr-hostd {}\n\nUsage:\n  glr-hostd --provider synthetic-counter [--transport stdio] [--max-frame-bytes N]",
                    env!("CARGO_PKG_VERSION")
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Cli {
        provider: provider.ok_or_else(|| "--provider is required".to_string())?,
        max_frame_bytes,
    })
}
