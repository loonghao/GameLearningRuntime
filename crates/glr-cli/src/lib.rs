mod args;
mod commands;
mod contracts;
mod error;
mod process;
mod project;
mod store;
pub mod update;

use std::ffi::OsString;

use clap::Parser;
use serde_json::json;

use crate::args::Cli;
use crate::commands::{CLI_OUTPUT_SCHEMA_VERSION, execute};

pub fn entrypoint(arguments: impl IntoIterator<Item = OsString>) -> i32 {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let json_requested = arguments.iter().any(|argument| argument == "--json");
    let cli = match Cli::try_parse_from(arguments) {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = error.exit_code();
            let _ = error.print();
            return exit_code;
        }
    };
    match execute(cli) {
        Ok(exit_code) => exit_code,
        Err(error) => {
            if json_requested {
                eprintln!(
                    "{}",
                    serde_json::to_string(&json!({
                        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
                        "command": "error",
                        "error": {"type": error.kind(), "message": error.to_string()},
                    }))
                    .unwrap_or_else(|_| "{\"command\":\"error\"}".into())
                );
            } else {
                eprintln!("{error}");
            }
            2
        }
    }
}
