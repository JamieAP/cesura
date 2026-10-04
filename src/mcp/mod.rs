//! cesura-mcp stdio MCP server. Gate: `feature = "mcp"`.

pub mod registry;
pub mod server;
pub mod tools;
pub mod transport;
pub mod validation;

pub use server::CesuraServer;
