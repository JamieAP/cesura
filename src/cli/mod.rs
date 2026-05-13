//! `cesura` CLI surface. Gate: `feature = "cli"`.
//!
//! Two subcommands:
//! - `watch` -- read JSONL observations from stdin, emit JSONL fires
//!   to stdout. One stdout line per `Fire`, suitable for piping into
//!   Claude Code `Monitor` so each line becomes one notification.
//! - `info` -- emit detector schema JSON. Discovery surface for agents.

use anyhow::Result;
use clap::{Parser, Subcommand};

pub mod info;
pub mod watch;

#[derive(Parser, Debug)]
#[command(
    name = "cesura",
    version,
    about = "Cesura BOCPD as an agent-consumable surface",
    long_about = None,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Stream JSONL observations from stdin, emit JSONL fires to stdout.
    Watch(watch::WatchArgs),
    /// Print detector schemas. Discovery surface for agents.
    Info(info::InfoArgs),
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Watch(args) => watch::run(args),
        Command::Info(args) => info::run(args),
    }
}
