// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

const MAX_TOKEN: usize = 8192;
const MAX_RESPONSE: u64 = 16 * 1024;
const READ: &str = "hms:read";
const WRITE: &str = "hms:write";

#[derive(clap::Args, Clone, Debug, Default)]
pub struct Options {
    /// External authorization-server issuer; enables the scoped OAuth resource profile.
    #[arg(long, env = writerslogic_warm::compatibility::env_name("WARM_MCP_OAUTH_ISSUER", "HMS_MCP_OAUTH_ISSUER"), requires_all = ["oauth_resource", "oauth_introspection"])]
    pub oauth_issuer: Option<String>,
    /// Public HTTPS MCP endpoint, also the exact access-token audience.
    #[arg(long, env = writerslogic_warm::compatibility::env_name("WARM_MCP_OAUTH_RESOURCE", "HMS_MCP_OAUTH_RESOURCE"), requires = "oauth_issuer")]
    pub oauth_resource: Option<String>,
    /// Pinned HTTPS RFC 7662 endpoint on the issuer origin. No discovery fetch or redirects.
    #[arg(long, env = writerslogic_warm::compatibility::env_name("WARM_MCP_OAUTH_INTROSPECTION", "HMS_MCP_OAUTH_INTROSPECTION"), requires = "oauth_issuer")]
    pub oauth_introspection: Option<String>,
}

fn https_uri(value: &str) -> Result<Uri> {
    ensure!(
        value.len() <= 2048 && !value.contains(['#', '"', '\\']),
        "invalid OAuth URL"
    );
    let uri: Uri = value.parse().context("invalid OAuth URL")?;
    ensure!(
        uri.scheme_str() == Some("https")
            && uri.authority().is_some_and(|a| !a.as_str().contains('@'))
            && uri.query().is_none(),
        "OAuth URLs require HTTPS, an authority and no credentials, query or fragment"
    );
    Ok(uri)
}

impl Options {
    pub fn enabled(&self) -> bool {
        self.oauth_issuer.is_some()
    }

    pub fn validate(&self, cli: &crate::config::Cli, static_token: bool) -> Result<()> {
        if !self.enabled() {
            ensure!(
                self.oauth_resource.is_none() && self.oauth_introspection.is_none(),
                "incomplete OAuth configuration"
            );
            return Ok(());
        }
        ensure!(
            !static_token,
            "static bearer and OAuth modes are mutually exclusive"
        );
        ensure!(
            !cli.namespace_arg
                && cli.namespace_strategy == crate::config::NamespaceStrategy::Global,
            "OAuth requires a fixed global namespace and forbids --namespace-arg"
        );
        let issuer = https_uri(self.oauth_issuer.as_deref().unwrap())?;
        let resource = https_uri(
            self.oauth_resource
                .as_deref()
                .context("missing OAuth resource")?,
        )?;
        let endpoint = https_uri(
            self.oauth_introspection
                .as_deref()
                .context("missing introspection endpoint")?,
        )?;
        ensure!(
            issuer.authority() == endpoint.authority(),
            "introspection must use the pinned issuer origin"
        );
        ensure!(
            resource.path() == cli.path,
            "OAuth resource path must equal the MCP endpoint path"
        );
        Ok(())
    }
}

pub struct Authority {
    issuer: String,
    resource: String,
    endpoint: String,
    credential: String,
    namespace: String,
    agent: ureq::Agent,
    slots: Arc<Semaphore>,
    max_body: usize,
    pub metadata_path: String,
    metadata_url: String,
}

impl Authority {
    fn agent() -> ureq::Agent {
        ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .https_only(true)
                .proxy(None)
                .max_redirects(0)
                .timeout_global(Some(Duration::from_secs(5)))
                .build(),
        )
    }

    #[cfg(test)]
    pub(super) fn for_test(endpoint: String) -> Arc<Self> {
        Arc::new(Self {
            issuer: "https://issuer.example".into(),
            resource: "https://memory.example/mcp".into(),
            endpoint,
            credential: "resource-credential".into(),
            namespace: "team".into(),
            agent: ureq::Agent::new_with_config(
                ureq::Agent::config_builder()
                    .proxy(None)
                    .max_redirects(0)
                    .timeout_global(Some(Duration::from_secs(1)))
                    .build(),
            ),
            slots: Arc::new(Semaphore::new(8)),
            max_body: 1024,
            metadata_path: "/.well-known/oauth-protected-resource/mcp".into(),
            metadata_url: "https://memory.example/.well-known/oauth-protected-resource/mcp".into(),
        })
    }

    pub fn from_cli(cli: &crate::config::Cli) -> Result<Option<Arc<Self>>> {
        if !cli.oauth.enabled() {
            return Ok(None);
        }
        cli.oauth
            .validate(cli, crate::config::http_token()?.is_some())?;
        let credential = writerslogic_warm::compatibility::var(
            "WARM_MCP_OAUTH_INTROSPECTION_TOKEN",
        )
        .context(
            "OAuth requires WARM_MCP_OAUTH_INTROSPECTION_TOKEN for resource-server authentication",
        )?;
        ensure!(valid_token(&credential), "invalid introspection credential");
        let issuer = cli.oauth.oauth_issuer.clone().unwrap();
        let resource = cli.oauth.oauth_resource.clone().unwrap();
        let uri = https_uri(&resource)?;
        let metadata_path = format!(
            "/.well-known/oauth-protected-resource{}",
            if uri.path() == "/" { "" } else { uri.path() }
        );
        let metadata_url = format!("https://{}{}", uri.authority().unwrap(), metadata_path);
        Ok(Some(Arc::new(Self {
            issuer,
            resource,
            endpoint: cli.oauth.oauth_introspection.clone().unwrap(),
            credential,
            namespace: cli.namespace.clone(),
            agent: Self::agent(),
            slots: Arc::new(Semaphore::new(8)),
            max_body: cli.max_request_bytes,
            metadata_path,
            metadata_url,
        })))
    }

    pub fn metadata(&self) -> Value {
        json!({"resource":self.resource, "authorization_servers":[self.issuer],
            "scopes_supported":[READ,WRITE], "bearer_methods_supported":["header"]})
    }

    fn challenge(&self, status: StatusCode, error: Option<&str>, scope: &str) -> Response {
        let mut challenge = format!(
            "Bearer resource_metadata=\"{}\", scope=\"{scope}\"",
            self.metadata_url
        );
        if let Some(error) = error {
            challenge.push_str(&format!(", error=\"{error}\""));
        }
        (
            status,
            [
                (header::WWW_AUTHENTICATE, challenge),
                (header::CACHE_CONTROL, "no-store".into()),
            ],
            "authorization required",
        )
            .into_response()
    }

    fn introspect(&self, token: &str) -> Result<Introspection> {
        let mut response = self
            .agent
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.credential))
            .header("Accept", "application/json")
            .send_form([("token", token), ("token_type_hint", "access_token")])?;
        ensure!(response.status().as_u16() == 200, "introspection status");
        let mut bytes = Vec::new();
        response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE + 1)
            .reader()
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= MAX_RESPONSE,
            "introspection response too large"
        );
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn check(self: &Arc<Self>, token: String) -> Result<Introspection, &'static str> {
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| "authorization service busy")?;
        let worker = self.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            worker.introspect(&token)
        })
        .await
        .map_err(|_| "authorization service unavailable")?
        .map_err(|_| "authorization service unavailable")
    }
}

fn valid_token(token: &str) -> bool {
    !token.is_empty()
        && token.len() <= MAX_TOKEN
        && !token.trim_end_matches('=').is_empty()
        && token
            .trim_end_matches('=')
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~+/".contains(&b))
}

pub(super) fn bearer_value(headers: &HeaderMap) -> Option<&str> {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim_start_matches(' '))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    bearer_value(headers).filter(|token| valid_token(token))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Audience {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Introspection {
    active: bool,
    iss: Option<String>,
    aud: Option<Audience>,
    sub: Option<String>,
    exp: Option<u64>,
    nbf: Option<u64>,
    iat: Option<u64>,
    token_type: Option<String>,
    scope: Option<String>,
    hms_namespace: Option<String>,
    cnf: Option<Value>,
}

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Invalid,
    MissingRead,
    Allowed { write: bool },
}

impl Introspection {
    fn decision(&self, issuer: &str, resource: &str, namespace: &str, now: u64) -> Decision {
        let audience = match &self.aud {
            Some(Audience::One(a)) => a == resource,
            Some(Audience::Many(a)) => {
                !a.is_empty() && a.len() <= 16 && a.iter().any(|a| a == resource)
            }
            None => false,
        };
        if !self.active
            || self.iss.as_deref() != Some(issuer)
            || !audience
            || !self
                .sub
                .as_ref()
                .is_some_and(|s| !s.is_empty() && s.len() <= 1024)
            || self.exp.is_none_or(|t| t <= now)
            || self.nbf.is_some_and(|t| t > now)
            || self.iat.is_some_and(|t| t > now)
            || !self
                .token_type
                .as_deref()
                .is_some_and(|t| t.eq_ignore_ascii_case("bearer"))
            || self.hms_namespace.as_deref() != Some(namespace)
            || self.cnf.is_some()
        {
            return Decision::Invalid;
        }
        let Some(scope) = &self.scope else {
            return Decision::MissingRead;
        };
        if scope.len() > 2048
            || !scope
                .bytes()
                .all(|b| b == b' ' || ((0x21..=0x7e).contains(&b) && b != b'"' && b != b'\\'))
        {
            return Decision::Invalid;
        }
        if !scope.split(' ').any(|s| s == READ) {
            return Decision::MissingRead;
        }
        Decision::Allowed {
            write: scope.split(' ').any(|s| s == WRITE),
        }
    }
}

fn needs_write(message: &Value) -> bool {
    message.get("method").and_then(Value::as_str) == Some("tools/call")
        && message
            .pointer("/params/name")
            .and_then(Value::as_str)
            .is_some_and(crate::tools::requires_write)
}

pub async fn authorize(
    State(authority): State<Arc<Authority>>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(token) = bearer(request.headers()).map(str::to_owned) else {
        return authority.challenge(StatusCode::UNAUTHORIZED, None, READ);
    };
    let claims = match authority.check(token).await {
        Ok(claims) => claims,
        Err(message) => return (StatusCode::SERVICE_UNAVAILABLE, message).into_response(),
    };
    let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let write = match claims.decision(
        &authority.issuer,
        &authority.resource,
        &authority.namespace,
        now.as_secs(),
    ) {
        Decision::Invalid => {
            return authority.challenge(StatusCode::UNAUTHORIZED, Some("invalid_token"), READ)
        }
        Decision::MissingRead => {
            return authority.challenge(StatusCode::FORBIDDEN, Some("insufficient_scope"), READ)
        }
        Decision::Allowed { write } => write,
    };
    if request.headers().contains_key("mcp-session-id") {
        return (
            StatusCode::BAD_REQUEST,
            "OAuth profile uses stateless requests",
        )
            .into_response();
    }
    if request.method() == Method::POST {
        let (parts, body) = request.into_parts();
        let bytes = match to_bytes(body, authority.max_body).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        };
        let Ok(message) = serde_json::from_slice::<Value>(&bytes) else {
            return StatusCode::BAD_REQUEST.into_response();
        };
        if !message.is_object() {
            return StatusCode::BAD_REQUEST.into_response();
        }
        if !write && needs_write(&message) {
            return authority.challenge(
                StatusCode::FORBIDDEN,
                Some("insufficient_scope"),
                "hms:read hms:write",
            );
        }
        request = Request::from_parts(parts, Body::from(bytes));
    }
    request.headers_mut().remove(header::AUTHORIZATION);
    if SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .is_none_or(|now| claims.exp.is_none_or(|expiry| expiry <= now.as_secs()))
    {
        return authority.challenge(StatusCode::UNAUTHORIZED, Some("invalid_token"), READ);
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn production_introspection_client_rejects_plaintext_endpoints() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut authority = Authority::for_test(format!(
            "http://{}/introspect",
            listener.local_addr().unwrap()
        ));
        Arc::get_mut(&mut authority).unwrap().agent = Authority::agent();
        assert!(authority.check("access-token".into()).await.is_err());
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err()
        );
    }

    #[cfg(feature = "server")]
    #[tokio::test]
    async fn production_introspection_client_rejects_untrusted_tls_peers() {
        let signed = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![signed.cert.der().clone()],
            rustls::pki_types::PrivatePkcs8KeyDer::from(signed.signing_key.serialize_der()).into(),
        )
        .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut authority = Authority::for_test(format!(
            "https://{}/introspect",
            listener.local_addr().unwrap()
        ));
        Arc::get_mut(&mut authority).unwrap().agent = Authority::agent();
        let peer = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            assert!(acceptor.accept(stream).await.is_err());
        });
        assert!(authority.check("access-token".into()).await.is_err());
        tokio::time::timeout(Duration::from_secs(2), peer)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn introspection_does_not_follow_redirects_or_retry_failures() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let redirected = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let authority = Authority::for_test(format!(
            "http://{}/introspect",
            listener.local_addr().unwrap()
        ));
        let destination = redirected.local_addr().unwrap();
        let peer = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 8192];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream.write_all(format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{destination}/stolen\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
            drop(stream);
            assert!(
                tokio::time::timeout(Duration::from_millis(100), listener.accept())
                    .await
                    .is_err()
            );
        });
        assert!(authority.check("access-token".into()).await.is_err());
        peer.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), redirected.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn timed_out_introspection_releases_its_worker_permit() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let authority = Authority::for_test(format!(
            "http://{}/introspect",
            listener.local_addr().unwrap()
        ));
        let checking = tokio::spawn({
            let authority = authority.clone();
            async move { authority.check("access-token".into()).await }
        });
        let (stream, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
            .await
            .unwrap()
            .unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(2), checking)
            .await
            .unwrap()
            .unwrap()
            .is_err());
        assert_eq!(authority.slots.available_permits(), 8);
        drop(stream);
    }

    #[tokio::test]
    async fn cancelled_introspection_retains_admission_until_blocking_workers_exit() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let authority = Authority::for_test(format!(
            "http://{}/introspect",
            listener.local_addr().unwrap()
        ));
        let (arrived, mut arrivals) = tokio::sync::mpsc::channel(8);
        let release = Arc::new(tokio::sync::Notify::new());
        let responder = tokio::spawn({
            let release = release.clone();
            async move {
                let mut streams = Vec::new();
                for _ in 0..8 {
                    streams.push(listener.accept().await.unwrap().0);
                    arrived.send(()).await.unwrap();
                }
                release.notified().await;
                drop(streams);
            }
        });
        let mut checks = Vec::new();
        for _ in 0..8 {
            let authority = authority.clone();
            checks.push(tokio::spawn(async move {
                authority.check("access-token".into()).await
            }));
        }
        for _ in 0..8 {
            tokio::time::timeout(Duration::from_secs(2), arrivals.recv())
                .await
                .unwrap()
                .unwrap();
        }
        for check in checks {
            check.abort();
            let _ = check.await;
        }
        assert!(matches!(
            authority.check("access-token".into()).await,
            Err("authorization service busy")
        ));
        release.notify_one();
        responder.await.unwrap();
        let permits = tokio::time::timeout(
            Duration::from_secs(2),
            authority.slots.clone().acquire_many_owned(8),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(permits.num_permits(), 8);
    }

    fn active() -> Value {
        json!({"active":true,"iss":"https://issuer.example","aud":"https://memory.example/mcp",
            "sub":"alice","exp":101,"nbf":99,"iat":98,"token_type":"Bearer",
            "scope":"hms:read hms:write","hms_namespace":"team"})
    }
    fn decision(value: Value) -> Decision {
        serde_json::from_value::<Introspection>(value)
            .unwrap()
            .decision(
                "https://issuer.example",
                "https://memory.example/mcp",
                "team",
                100,
            )
    }
    #[test]
    fn scope_issuer_audience_namespace_expiry_and_revocation_are_independent() {
        assert_eq!(decision(active()), Decision::Allowed { write: true });
        for (field, value) in [
            ("active", json!(false)),
            ("iss", json!("https://attacker.example")),
            ("aud", json!(["https://other.example/mcp"])),
            ("hms_namespace", json!("other")),
            ("exp", json!(100)),
            ("exp", json!(99)),
            ("nbf", json!(101)),
            ("iat", json!(101)),
            ("cnf", json!({"jkt":"unsupported"})),
            ("token_type", json!("DPoP")),
            ("sub", json!("")),
        ] {
            let mut value_under_test = active();
            value_under_test[field] = value;
            assert_eq!(decision(value_under_test), Decision::Invalid, "{field}");
        }
        for field in ["iss", "aud", "exp", "sub", "hms_namespace", "token_type"] {
            let mut value = active();
            value.as_object_mut().unwrap().remove(field);
            assert_eq!(decision(value), Decision::Invalid, "missing {field}");
        }
        let mut value = active();
        value["scope"] = json!("hms:read");
        assert_eq!(decision(value.clone()), Decision::Allowed { write: false });
        value["scope"] = json!("hms:write");
        assert_eq!(decision(value), Decision::MissingRead);
        let mut value = active();
        value["aud"] = json!(["https://other.example", "https://memory.example/mcp"]);
        assert_eq!(decision(value), Decision::Allowed { write: true });
    }
    #[test]
    fn bearer_and_url_boundaries_fail_closed() {
        for size in [MAX_TOKEN - 1, MAX_TOKEN, MAX_TOKEN + 1] {
            assert_eq!(valid_token(&"a".repeat(size)), size <= MAX_TOKEN);
        }
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "bEaReR secret".parse().unwrap());
        assert_eq!(bearer(&headers), Some("secret"));
        headers.insert(header::AUTHORIZATION, "Bearer   secret".parse().unwrap());
        assert_eq!(bearer(&headers), Some("secret"));
        headers.append(header::AUTHORIZATION, "Bearer other".parse().unwrap());
        assert!(bearer(&headers).is_none());
        for url in [
            "http://example.com",
            "https://u:p@example.com",
            "https://example.com/#x",
            "https://example.com/?x",
        ] {
            assert!(https_uri(url).is_err());
        }
        assert!(https_uri("https://issuer.example/tenant").is_ok());
    }
    #[test]
    fn every_mutating_tool_requires_write_scope() {
        for name in ["remember", "forget"] {
            assert!(needs_write(
                &json!({"method":"tools/call", "params":{"name":name}})
            ));
        }
        for name in ["recall", "search", "get", "list", "stats", "export"] {
            assert!(!needs_write(
                &json!({"method":"tools/call", "params":{"name":name}})
            ));
        }
    }
}
