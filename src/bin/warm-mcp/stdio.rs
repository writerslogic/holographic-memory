// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::RoleServer;
use tokio::io::{AsyncRead, AsyncWrite};

pub fn bounded<R, W>(read: R, write: W, limit: usize) -> AsyncRwTransport<RoleServer, R, W>
where
    R: AsyncRead + Send + Unpin + 'static,
    W: AsyncWrite + Send + Unpin + 'static,
{
    AsyncRwTransport::new_server(read, write).with_max_line_length(limit)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::time::Duration;

    use rmcp::model::{ClientJsonRpcMessage, RequestId};
    use rmcp::transport::Transport;
    use tokio::io::AsyncWriteExt;

    use super::*;

    fn ping(id: u64, bytes: usize, ending: &[u8]) -> Vec<u8> {
        let mut message = format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"ping"}}"#).into_bytes();
        assert!(message.len() + ending.len() <= bytes);
        message.resize(bytes - ending.len(), b' ');
        message.extend_from_slice(ending);
        message
    }

    fn id(message: Option<ClientJsonRpcMessage>) -> RequestId {
        match message.unwrap() {
            ClientJsonRpcMessage::Request(request) => request.id,
            _ => panic!("expected request"),
        }
    }

    #[tokio::test]
    async fn complete_line_limit_is_inclusive_and_oversized_frames_recover() {
        let limit = 1024;
        for ending in [b"\n".as_slice(), b"\r\n".as_slice()] {
            for bytes in [limit - 1, limit, limit + 1] {
                let mut input = ping(1, bytes, ending);
                input.extend(ping(2, 128, b"\n"));
                let mut transport = bounded(Cursor::new(input), tokio::io::sink(), limit);
                assert_eq!(
                    id(transport.receive().await),
                    RequestId::Number(if bytes <= limit { 1 } else { 2 })
                );
                if bytes <= limit {
                    assert_eq!(id(transport.receive().await), RequestId::Number(2));
                }
                assert!(transport.receive().await.is_none());
            }
        }
    }

    #[tokio::test]
    async fn cancelling_receive_mid_discard_preserves_framing_across_large_binary_input() {
        let limit = 1024;
        let (read, mut input) = tokio::io::duplex(2048);
        let mut transport = bounded(read, tokio::io::sink(), limit);
        input.write_all(&vec![0xff; limit + 1]).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), transport.receive())
                .await
                .is_err()
        );
        let writer = tokio::spawn(async move {
            let block = [0xff; 8192];
            for _ in 0..256 {
                input.write_all(&block).await.unwrap();
            }
            input.write_all(b"\n").await.unwrap();
            input.write_all(&ping(7, 128, b"\n")).await.unwrap();
            input.shutdown().await.unwrap();
        });
        let message = tokio::time::timeout(Duration::from_secs(2), transport.receive())
            .await
            .unwrap();
        assert_eq!(id(message), RequestId::Number(7));
        assert!(transport.receive().await.is_none());
        writer.await.unwrap();
    }
}
