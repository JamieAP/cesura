//! CesuraServer -- rmcp ServerHandler impl + `#[tool_router]`.
//!
//! Registers five tools with rmcp using `#[tool_router]`, `#[tool]`,
//! and `#[tool_handler]`. Tool bodies delegate to the stream registry
//! and detector runtime.

use std::fmt::Write as _;
use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};

use crate::mcp::registry::Registry;
use crate::mcp::tools::{
    cesura_close_stream_impl, cesura_feed_impl, cesura_list_streams_impl, cesura_restore_impl,
    cesura_snapshot_impl, CloseRequest, FeedRequest, RestoreRequest, SnapshotRequest,
};
use crate::runtime::DetectorKind;

/// MCP server façade over the cesura Runner registry.
///
/// NOT safe to instantiate twice in one process: the stream-id <-> UUID
/// resolver in `tools.rs` is a process-global static. Two `CesuraServer`s
/// in one process will see each other's stream ids and surface
/// `internal_error("registry / id-map drift")` on cross-talk. The
/// `cesura-mcp` bin instantiates exactly one.
#[derive(Clone)]
#[allow(dead_code)] // tool_router field is read by the #[tool_router] macro via reflection
pub struct CesuraServer {
    pub(crate) streams: Arc<Registry>,
    tool_router: ToolRouter<CesuraServer>,
}

#[tool_router]
impl CesuraServer {
    pub fn new() -> Self {
        Self {
            streams: Arc::new(Registry::default()),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Feed observations to a streaming detector. Lazy-creates the stream on first call.\n\nObservation shape: univariate kinds (streaming, chen-wu) take an array of f64, e.g. `[1.0, 2.0, 3.0]`. Multistream kinds (sum-cusum, hc, filter-tick, dm-bocd) take an array of f64 arrays, e.g. `[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]`. Length of each inner array must equal the stream's `d` param."
    )]
    async fn cesura_feed(
        &self,
        Parameters(req): Parameters<FeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        cesura_feed_impl(&self.streams, req).await
    }

    #[tool(
        description = "Capture a snapshot of a stream's state. Errors with structured_error for kind=dm-bocd (upstream StreamingDmBocd save_state not yet implemented)."
    )]
    async fn cesura_snapshot(
        &self,
        Parameters(req): Parameters<SnapshotRequest>,
    ) -> Result<CallToolResult, McpError> {
        cesura_snapshot_impl(&self.streams, req).await
    }

    #[tool(
        description = "Restore a stream from a previously captured snapshot. Returns the new stream_id."
    )]
    async fn cesura_restore(
        &self,
        Parameters(req): Parameters<RestoreRequest>,
    ) -> Result<CallToolResult, McpError> {
        cesura_restore_impl(&self.streams, req).await
    }

    #[tool(description = "List every live stream with its id and detector kind.")]
    async fn cesura_list_streams(&self) -> Result<CallToolResult, McpError> {
        cesura_list_streams_impl(&self.streams).await
    }

    #[tool(
        description = "Close a stream. Returns ok=true plus was_present=bool so client bugs (closing unknown ids) surface."
    )]
    async fn cesura_close_stream(
        &self,
        Parameters(req): Parameters<CloseRequest>,
    ) -> Result<CallToolResult, McpError> {
        cesura_close_stream_impl(&self.streams, req).await
    }
}

impl Default for CesuraServer {
    fn default() -> Self {
        Self::new()
    }
}

fn build_instructions() -> String {
    let mut s = String::from(
        "cesura BOCPD detector runtime exposed over MCP.\n\
         \n\
         Tools:\n\
         - cesura_feed: feed observations. Lazy-creates a stream on first call (provide kind+params).\n\
         - cesura_snapshot: capture stream state. dm-bocd kind returns structured_error.\n\
         - cesura_restore: restore from a previous snapshot.\n\
         - cesura_list_streams: list live streams.\n\
         - cesura_close_stream: remove a stream.\n\
         \n\
         Detector kinds:",
    );
    for k in DetectorKind::all() {
        let _ = write!(s, " {}", k.as_wire());
    }
    s.push_str(
        ".\n\
         Univariate kinds (streaming, chen-wu) take scalar observations.\n\
         Multistream / multivariate kinds take vector observations.\n\
         Param shapes per kind are documented in the cesura README.",
    );
    s
}

#[tool_handler]
impl ServerHandler for CesuraServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "cesura-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_protocol_version(ProtocolVersion::V_2025_11_25)
            .with_instructions(build_instructions())
    }
}
