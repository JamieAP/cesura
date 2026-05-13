//! `cesura-mcp` binary -- stdio MCP server. See `src/mcp/`.

use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Stdout discipline: only JSON-RPC frames on stdout. Logs → stderr.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    tracing::info!("cesura-mcp v{} starting on stdio", env!("CARGO_PKG_VERSION"));

    let service = cesura::mcp::CesuraServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
