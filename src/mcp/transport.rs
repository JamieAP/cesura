//! Resilient stdio transport for `cesura-mcp`.
//!

use std::marker::PhantomData;
use std::sync::Arc;

use rmcp::service::{RxJsonRpcMessage, ServiceRole, TxJsonRpcMessage};
use rmcp::transport::Transport;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdin, Stdout};
use tokio::sync::Mutex;

const PARSE_ERROR_FRAME: &[u8] =
    b"{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32700,\"message\":\"Parse error\"}}\n";

pub struct ResilientStdio<R: ServiceRole> {
    read: BufReader<Stdin>,
    write: Arc<Mutex<Stdout>>,
    line: Vec<u8>,
    _role: PhantomData<fn() -> R>,
}

impl<R: ServiceRole> ResilientStdio<R> {
    fn new() -> Self {
        Self {
            read: BufReader::new(tokio::io::stdin()),
            write: Arc::new(Mutex::new(tokio::io::stdout())),
            line: Vec::with_capacity(4096),
            _role: PhantomData,
        }
    }
}

/// Stdio transport with malformed-frame recovery. Each malformed line
/// produces a `-32700 Parse error` JSON-RPC reply and the channel
/// continues; valid frames are forwarded to rmcp's service loop.
pub fn resilient_stdio<R: ServiceRole>() -> impl Transport<R, Error = std::io::Error> + 'static {
    ResilientStdio::<R>::new()
}

impl<R: ServiceRole> Transport<R> for ResilientStdio<R> {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<R>,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send + 'static {
        let write = self.write.clone();
        async move {
            let mut buf = serde_json::to_vec(&item)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            buf.push(b'\n');
            let mut out = write.lock().await;
            out.write_all(&buf).await?;
            out.flush().await?;
            Ok(())
        }
    }

    fn receive(&mut self) -> impl std::future::Future<Output = Option<RxJsonRpcMessage<R>>> + Send {
        async move {
            loop {
                self.line.clear();
                match self.read.read_until(b'\n', &mut self.line).await {
                    Ok(0) => return None,
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("stdin read error: {e}");
                        return None;
                    }
                }
                let slice = match self.line.last() {
                    Some(b'\n') => &self.line[..self.line.len() - 1],
                    _ => &self.line[..],
                };
                let slice = match slice.last() {
                    Some(b'\r') => &slice[..slice.len() - 1],
                    _ => slice,
                };
                if slice.is_empty() {
                    continue;
                }
                match serde_json::from_slice::<RxJsonRpcMessage<R>>(slice) {
                    Ok(msg) => return Some(msg),
                    Err(e) => {
                        tracing::warn!(
                            "malformed frame: {} | replying -32700 and continuing",
                            e
                        );
                        let mut out = self.write.lock().await;
                        if let Err(write_err) = out.write_all(PARSE_ERROR_FRAME).await {
                            tracing::error!("stdout write error on -32700 reply: {write_err}");
                            return None;
                        }
                        if let Err(flush_err) = out.flush().await {
                            tracing::error!("stdout flush error on -32700 reply: {flush_err}");
                            return None;
                        }
                    }
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        let mut out = self.write.lock().await;
        out.flush().await
    }
}
