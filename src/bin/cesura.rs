//! `cesura` binary -- agent-facing CLI. See `src/cli/` for subcommands.

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = cesura::cli::Cli::parse();
    cesura::cli::run(cli)
}
