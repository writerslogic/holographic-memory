// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures_util::Stream;
use parking_lot::Mutex;
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_server::session::local::{
    LocalSessionManager, LocalSessionManagerError, LocalSessionWorker,
};
use rmcp::transport::streamable_http_server::session::{
    ServerSseMessage, SessionId, SessionManager,
};
use rmcp::transport::{Transport, WorkerTransport};
use rmcp::RoleServer;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy)]
pub struct Limits {
    pub sessions: usize,
    pub idle: Duration,
    pub requests: usize,
    pub bytes: usize,
    pub message_bytes: usize,
    pub initialize: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MCP session resource limit reached")]
    Limit,
    #[error("MCP session is unavailable")]
    Missing,
    #[error("MCP session cleanup failed")]
    Cleanup,
    #[error("MCP session operation failed: {0}")]
    Local(#[from] LocalSessionManagerError),
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error(
        "MCP session response budget exhausted; reconcile unacknowledged mutations before retrying"
    )]
    Limit,
    #[error("MCP session transport failed: {0}")]
    Native(Box<<WorkerTransport<LocalSessionWorker> as Transport<RoleServer>>::Error>),
}

struct Activity {
    last: Instant,
    active: usize,
    requests: usize,
    bytes: usize,
    initialized: bool,
}

struct Entry {
    born: Instant,
    activity: Mutex<Activity>,
    cancel: CancellationToken,
    _permit: OwnedSemaphorePermit,
}

struct Shared {
    inner: Arc<LocalSessionManager>,
    entries: Mutex<HashMap<SessionId, Arc<Entry>>>,
    slots: Arc<Semaphore>,
    limits: Limits,
}

impl Shared {
    fn retire(self: &Arc<Self>, id: &SessionId) -> Option<tokio::task::JoinHandle<()>> {
        let entry = self.entries.lock().remove(id)?;
        entry.cancel.cancel();
        let shared = self.clone();
        let id = id.clone();
        Some(tokio::spawn(async move {
            shared.inner.sessions.write().await.remove(&id);
            drop(entry);
        }))
    }

    fn acquire(&self, id: &SessionId) -> Result<Use, Error> {
        let entries = self.entries.lock();
        let entry = entries.get(id).ok_or(Error::Missing)?;
        let mut activity = entry.activity.lock();
        if entry.cancel.is_cancelled() || activity.requests >= self.limits.requests {
            return Err(Error::Limit);
        }
        activity.requests += 1;
        activity.active += 1;
        activity.last = Instant::now();
        Ok(Use(entry.clone()))
    }
}

struct Use(Arc<Entry>);

impl Drop for Use {
    fn drop(&mut self) {
        let mut activity = self.0.activity.lock();
        assert!(activity.active > 0, "MCP session lease underflow");
        activity.active -= 1;
        activity.last = Instant::now();
    }
}

struct Initialization {
    shared: Arc<Shared>,
    id: SessionId,
    complete: bool,
}

impl Drop for Initialization {
    fn drop(&mut self) {
        if !self.complete {
            self.shared.retire(&self.id);
        }
    }
}

pub struct BoundedSessions {
    shared: Arc<Shared>,
}

impl BoundedSessions {
    pub fn new(limits: Limits) -> Arc<Self> {
        let mut inner = LocalSessionManager::default();
        inner.session_config.channel_capacity = 4;
        inner.session_config.keep_alive = Some(limits.idle);
        inner.session_config.init_timeout = Some(limits.initialize);
        Arc::new(Self {
            shared: Arc::new(Shared {
                inner: Arc::new(inner),
                entries: Mutex::new(HashMap::new()),
                slots: Arc::new(Semaphore::new(limits.sessions)),
                limits,
            }),
        })
    }

    pub async fn reap(&self, cancel: CancellationToken) {
        let mut interval =
            tokio::time::interval(self.shared.limits.idle.min(Duration::from_secs(1)));
        loop {
            tokio::select! {
                () = cancel.cancelled() => return,
                _ = interval.tick() => self.reap_expired().await,
            }
        }
    }

    async fn reap_expired(&self) {
        let ids: Vec<_> = self
            .shared
            .entries
            .lock()
            .iter()
            .filter_map(|(id, entry)| {
                let activity = entry.activity.lock();
                let expired = (!activity.initialized
                    && entry.born.elapsed() >= self.shared.limits.initialize)
                    || (activity.active == 0 && activity.last.elapsed() >= self.shared.limits.idle)
                    || (activity.active == 0 && activity.requests >= self.shared.limits.requests);
                (expired || entry.cancel.is_cancelled()).then(|| id.clone())
            })
            .collect();
        for id in ids {
            if let Some(task) = self.shared.retire(&id) {
                if task.await.is_err() {
                    crate::log::error("MCP session cleanup task failed");
                }
            }
        }
    }

    pub async fn shutdown(&self) {
        self.shared.slots.close();
        let ids: Vec<_> = self.shared.entries.lock().keys().cloned().collect();
        for id in ids {
            if let Some(task) = self.shared.retire(&id) {
                if task.await.is_err() {
                    crate::log::error("MCP session cleanup task failed");
                }
            }
        }
    }
}

pub struct SessionTransport {
    inner: WorkerTransport<LocalSessionWorker>,
    entry: Arc<Entry>,
    shared: Weak<Shared>,
    id: SessionId,
    message_bytes: usize,
    session_bytes: usize,
}

struct Count {
    count: usize,
    limit: usize,
}

impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.count = self
            .count
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| std::io::Error::other("MCP serialized message exceeds budget"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Transport<RoleServer> for SessionTransport {
    type Error = TransportError;

    fn send(
        &mut self,
        item: ServerJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let mut count = Count {
            count: 0,
            limit: self.message_bytes,
        };
        let serialized = serde_json::to_writer(&mut count, &item).is_ok();
        let allowed = {
            let mut activity = self.entry.activity.lock();
            match activity.bytes.checked_add(count.count) {
                Some(total)
                    if serialized
                        && total <= self.session_bytes
                        && !self.entry.cancel.is_cancelled() =>
                {
                    activity.bytes = total;
                    true
                }
                _ => false,
            }
        };
        if !allowed {
            if let Some(shared) = self.shared.upgrade() {
                shared.retire(&self.id);
            }
        }
        let future = self.inner.send(item);
        async move {
            if !allowed {
                return Err(TransportError::Limit);
            }
            future
                .await
                .map_err(|error| TransportError::Native(Box::new(error)))
        }
    }

    async fn receive(&mut self) -> Option<ClientJsonRpcMessage> {
        tokio::select! {
            () = self.entry.cancel.cancelled() => None,
            message = self.inner.receive() => message,
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        if let Some(shared) = self.shared.upgrade() {
            shared.retire(&self.id);
        }
        self.inner
            .close()
            .await
            .map_err(|error| TransportError::Native(Box::new(error)))
    }
}

impl Drop for SessionTransport {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            shared.retire(&self.id);
        }
    }
}

struct GuardedStream {
    inner: Pin<Box<dyn Stream<Item = ServerSseMessage> + Send + Sync>>,
    _guard: Option<Use>,
    cancelled: Pin<Box<dyn Future<Output = ()> + Send + Sync>>,
}

impl GuardedStream {
    fn new(
        stream: impl Stream<Item = ServerSseMessage> + Send + Sync + 'static,
        guard: Use,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            cancelled: Box::pin(guard.0.cancel.clone().cancelled_owned()),
            _guard: Some(guard),
        }
    }
}

impl Stream for GuardedStream {
    type Item = ServerSseMessage;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.cancelled.as_mut().poll(cx).is_ready() {
            self._guard.take();
            return Poll::Ready(None);
        }
        let next = self.inner.as_mut().poll_next(cx);
        if next.is_ready() {
            if let Some(guard) = &self._guard {
                guard.0.activity.lock().last = Instant::now();
            }
            if matches!(next, Poll::Ready(None)) {
                self._guard.take();
            }
        }
        next
    }
}

impl SessionManager for BoundedSessions {
    type Error = Error;
    type Transport = SessionTransport;

    async fn create_session(&self) -> Result<(SessionId, Self::Transport), Self::Error> {
        let permit = self
            .shared
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Limit)?;
        let (id, inner) = self.shared.inner.create_session().await?;
        let entry = Arc::new(Entry {
            born: Instant::now(),
            activity: Mutex::new(Activity {
                last: Instant::now(),
                active: 0,
                requests: 0,
                bytes: 0,
                initialized: false,
            }),
            cancel: inner.cancel_token(),
            _permit: permit,
        });
        self.shared.entries.lock().insert(id.clone(), entry.clone());
        Ok((
            id.clone(),
            SessionTransport {
                inner,
                entry,
                shared: Arc::downgrade(&self.shared),
                id,
                message_bytes: self.shared.limits.message_bytes,
                session_bytes: self.shared.limits.bytes,
            },
        ))
    }

    async fn initialize_session(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<ServerJsonRpcMessage, Self::Error> {
        let guard = self.shared.acquire(id)?;
        let mut initialization = Initialization {
            shared: self.shared.clone(),
            id: id.clone(),
            complete: false,
        };
        let response = self.shared.inner.initialize_session(id, message).await?;
        guard.0.activity.lock().initialized = true;
        initialization.complete = true;
        Ok(response)
    }

    async fn has_session(&self, id: &SessionId) -> Result<bool, Self::Error> {
        Ok(self
            .shared
            .entries
            .lock()
            .get(id)
            .is_some_and(|entry| !entry.cancel.is_cancelled()))
    }

    async fn close_session(&self, id: &SessionId) -> Result<(), Self::Error> {
        if let Some(task) = self.shared.retire(id) {
            task.await.map_err(|_| Error::Cleanup)?;
        }
        Ok(())
    }

    async fn create_stream(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let guard = self.shared.acquire(id)?;
        let stream = self.shared.inner.create_stream(id, message).await?;
        Ok(GuardedStream::new(stream, guard))
    }

    async fn accept_message(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<(), Self::Error> {
        let _guard = self.shared.acquire(id)?;
        self.shared
            .inner
            .accept_message(id, message)
            .await
            .map_err(Error::Local)
    }

    async fn create_standalone_stream(
        &self,
        id: &SessionId,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let guard = self.shared.acquire(id)?;
        let stream = self.shared.inner.create_standalone_stream(id).await?;
        Ok(GuardedStream::new(stream, guard))
    }

    async fn resume(
        &self,
        id: &SessionId,
        last_event_id: String,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let guard = self.shared.acquire(id)?;
        let stream = self.shared.inner.resume(id, last_event_id).await?;
        Ok(GuardedStream::new(stream, guard))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use rmcp::model::{CallToolResult, ContentBlock, RequestId, ServerResult};
    use serde_json::json;

    fn limits() -> Limits {
        Limits {
            sessions: 2,
            idle: Duration::from_millis(20),
            initialize: Duration::from_secs(1),
            requests: 100,
            bytes: 1 << 20,
            message_bytes: 1 << 16,
        }
    }

    fn initialize() -> ClientJsonRpcMessage {
        serde_json::from_value(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params": {
            "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}
        }})).unwrap()
    }

    #[derive(Clone)]
    struct Ping;
    impl rmcp::ServerHandler for Ping {}

    async fn started(manager: &Arc<BoundedSessions>) -> (SessionId, tokio::task::JoinHandle<()>) {
        let (id, transport) = manager.create_session().await.unwrap();
        let task = tokio::spawn(async move {
            if let Ok(service) = rmcp::serve_server(Ping, transport).await {
                let _ = service.waiting().await;
            }
        });
        manager.initialize_session(&id, initialize()).await.unwrap();
        manager
            .accept_message(
                &id,
                serde_json::from_value(
                    json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
                )
                .unwrap(),
            )
            .await
            .unwrap();
        (id, task)
    }

    #[tokio::test]
    async fn session_capacity_idle_expiry_and_transport_drop_remove_both_maps() {
        let manager = BoundedSessions::new(limits());
        let (a, first) = manager.create_session().await.unwrap();
        let (b, second) = manager.create_session().await.unwrap();
        assert!(matches!(manager.create_session().await, Err(Error::Limit)));
        drop(first);
        let permit = tokio::time::timeout(
            Duration::from_secs(1),
            manager.shared.slots.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!manager.shared.inner.sessions.read().await.contains_key(&a));
        drop(permit);
        let (_, third) = manager.create_session().await.unwrap();
        drop(third);
        tokio::time::sleep(Duration::from_millis(30)).await;
        manager.reap_expired().await;
        assert!(!manager.has_session(&b).await.unwrap());
        assert!(manager.shared.inner.sessions.read().await.is_empty());
        drop(second);
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn cancelled_initialization_reclaims_session_capacity() {
        let manager = BoundedSessions::new(Limits {
            sessions: 1,
            ..limits()
        });
        let (id, mut transport) = manager.create_session().await.unwrap();
        let task = tokio::spawn({
            let manager = manager.clone();
            let id = id.clone();
            async move { manager.initialize_session(&id, initialize()).await }
        });
        assert!(
            tokio::time::timeout(Duration::from_secs(1), transport.receive())
                .await
                .unwrap()
                .is_some()
        );
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(!manager.has_session(&id).await.unwrap());
        drop(transport);
        let permit = tokio::time::timeout(
            Duration::from_secs(1),
            manager.shared.slots.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(manager.shared.inner.sessions.read().await.is_empty());
        drop(permit);
        let (_, transport) = manager.create_session().await.unwrap();
        drop(transport);
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn concurrent_request_streams_keep_response_association_and_quota_is_exact() {
        let manager = BoundedSessions::new(Limits {
            requests: 4,
            idle: Duration::from_secs(30),
            ..limits()
        });
        let (id, task) = started(&manager).await;
        let a = manager
            .create_stream(
                &id,
                serde_json::from_value(json!({"jsonrpc":"2.0","id":1,"method":"ping"})).unwrap(),
            )
            .await
            .unwrap();
        let b = manager
            .create_stream(
                &id,
                serde_json::from_value(json!({"jsonrpc":"2.0","id":2,"method":"ping"})).unwrap(),
            )
            .await
            .unwrap();
        let replies = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(a.collect::<Vec<_>>(), b.collect::<Vec<_>>())
        })
        .await
        .unwrap();
        for (expected, stream) in [(1, replies.0), (2, replies.1)] {
            let ids: Vec<_> = stream
                .iter()
                .filter_map(|event| match event.message.as_deref() {
                    Some(ServerJsonRpcMessage::Response(response)) => Some(response.id.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(ids, vec![RequestId::Number(expected)]);
        }
        assert!(matches!(
            manager.create_stream(&id, initialize()).await,
            Err(Error::Limit)
        ));
        manager.reap_expired().await;
        assert!(!manager.has_session(&id).await.unwrap());
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn serialized_byte_limits_are_enforced_before_retaining_messages() {
        let message = || {
            ServerJsonRpcMessage::response(
                ServerResult::CallToolResult(CallToolResult::success(vec![ContentBlock::text(
                    "x".repeat(1024),
                )])),
                RequestId::Number(7),
            )
        };
        let size = serde_json::to_vec(&message()).unwrap().len();
        for budget in [size - 1, size, size + 1] {
            let manager = BoundedSessions::new(Limits {
                message_bytes: budget,
                bytes: budget * 2,
                ..limits()
            });
            let (id, mut transport) = manager.create_session().await.unwrap();
            let send = transport.send(message());
            if budget < size {
                assert!(matches!(send.await, Err(TransportError::Limit)));
                assert!(!manager.has_session(&id).await.unwrap());
            } else {
                drop(send);
                assert!(manager.has_session(&id).await.unwrap());
                assert_eq!(transport.entry.activity.lock().bytes, size);
            }
            drop(transport);
            manager.shutdown().await;
        }
    }
}
