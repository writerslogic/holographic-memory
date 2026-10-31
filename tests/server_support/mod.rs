// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Black-box harness for the `warm-server` integration tests: starts the real binary, talks
//! HTTP/1.1 over TCP, a unix socket or TLS, and reads its structured log.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

pub const DIM: usize = 16;
pub const BIN: &str = env!("CARGO_BIN_EXE_warm-server");

pub enum Endpoint {
    Tcp(SocketAddr),
    #[cfg(unix)]
    Unix(PathBuf),
}

pub struct Server {
    pub child: Child,
    pub endpoint: Endpoint,
    pub data_dir: PathBuf,
    pub stderr_path: PathBuf,
    pub tls_root: Option<Vec<u8>>,
    _scratch: Option<tempfile::TempDir>,
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Resp {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

pub struct Request<'a> {
    server: &'a Server,
    method: &'a str,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl<'a> Request<'a> {
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn bearer(self, token: &str) -> Self {
        let value = format!("Bearer {token}");
        self.header("Authorization", &value)
    }

    pub fn tenant(self, tenant: &str) -> Self {
        self.header("X-Tenant-ID", tenant)
    }

    pub fn body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }

    pub fn json(self, body: &Value) -> Self {
        self.header("Content-Type", "application/json")
            .body(body.to_string().into_bytes())
    }

    pub fn send(self) -> Resp {
        let method = self.method;
        let path = self.path.clone();
        let (server, headers, body) = self.encode();
        let raw = server.exchange_parts([headers, body]);
        assert!(
            !raw.is_empty(),
            "no response to {method} {path}; recent server events: {:?}",
            server
                .log_lines()
                .into_iter()
                .rev()
                .take(10)
                .collect::<Vec<_>>()
        );
        parse_response(&raw)
    }

    /// `None` when the server closed the connection without answering.
    pub fn try_send(self) -> Option<Resp> {
        let (server, headers, body) = self.encode();
        let raw = server.exchange_parts([headers, body]);
        (!raw.is_empty()).then(|| parse_response(&raw))
    }

    fn encode(self) -> (&'a Server, Vec<u8>, Vec<u8>) {
        let mut raw = format!(
            "{} {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n",
            self.method,
            self.path,
            self.body.len()
        );
        for (k, v) in &self.headers {
            raw.push_str(&format!("{k}: {v}\r\n"));
        }
        raw.push_str("\r\n");
        (self.server, raw.into_bytes(), self.body)
    }
}

fn parse_response(raw: &[u8]) -> Resp {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no header terminator in {:?}", String::from_utf8_lossy(raw)));
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("bad status line in {head:?}"));
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let mut body = raw[split + 4..].to_vec();
    if headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.contains("chunked"))
    {
        body = dechunk(&body);
    }
    Resp {
        status,
        headers,
        body,
    }
}

fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(eol) = data.windows(2).position(|w| w == b"\r\n") {
        let size = usize::from_str_radix(
            std::str::from_utf8(&data[..eol])
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .trim(),
            16,
        )
        .unwrap_or(0);
        data = &data[eol + 2..];
        if size == 0 || data.len() < size {
            break;
        }
        out.extend_from_slice(&data[..size]);
        data = data.get(size + 2..).unwrap_or(&[]);
    }
    out
}

pub trait Stream: Read + Write {}
impl<T: Read + Write> Stream for T {}

impl Server {
    pub fn start(extra: &[&str], envs: &[(&str, &str)]) -> Self {
        let scratch = tempfile::tempdir().unwrap();
        let dir = scratch.path().join("data");
        let mut server = Self::start_in(&dir, extra, envs);
        server._scratch = Some(scratch);
        server
    }

    /// Credential-less loopback server (`--no-auth`).
    pub fn open(extra: &[&str]) -> Self {
        let mut args = vec!["--no-auth"];
        args.extend_from_slice(extra);
        Self::start(&args, &[])
    }

    pub fn with_key(key: &str, extra: &[&str]) -> Self {
        Self::start(extra, &[("WARM_API_KEY", key)])
    }

    pub fn start_in(dir: &Path, extra: &[&str], envs: &[(&str, &str)]) -> Self {
        let stderr_path = dir.with_extension("stderr.log");
        std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
        let stderr = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&stderr_path)
            .unwrap();
        let mut cmd = Command::new(BIN);
        cmd.args(["--bind", "127.0.0.1:0", "--input-dim", &DIM.to_string()])
            .args(["--dim", "2048", "--data-dir"])
            .arg(dir)
            .args(extra)
            .env_remove("WARM_API_KEY")
            .stdout(Stdio::piped())
            .stderr(stderr);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP);
        }
        let mut child = cmd.spawn().unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        let target = line.trim().rsplit(' ').next().unwrap_or("").to_string();
        let endpoint = match target.strip_prefix("unix:") {
            #[cfg(unix)]
            Some(path) => Endpoint::Unix(PathBuf::from(path)),
            _ => Endpoint::Tcp(target.parse().unwrap_or_else(|_| {
                panic!(
                    "server did not start: {line:?}\n{}",
                    std::fs::read_to_string(&stderr_path).unwrap_or_default()
                )
            })),
        };
        let tls_root = extra
            .iter()
            .position(|a| *a == "--tls-cert")
            .and_then(|i| std::fs::read(extra[i + 1]).ok());
        Self {
            child,
            endpoint,
            data_dir: dir.to_path_buf(),
            stderr_path,
            tls_root,
            _scratch: None,
        }
    }

    /// Run the binary expecting it to exit with an error; returns its stderr.
    pub fn refuses(extra: &[&str], envs: &[(&str, &str)]) -> String {
        let scratch = tempfile::tempdir().unwrap();
        let mut cmd = Command::new(BIN);
        cmd.args(["--data-dir"])
            .arg(scratch.path())
            .args(extra)
            .env_remove("WARM_API_KEY");
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        assert!(
            !out.status.success(),
            "server unexpectedly started with {extra:?}"
        );
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    pub fn connect(&self) -> Box<dyn Stream> {
        self.try_connect().unwrap()
    }

    fn try_connect(&self) -> std::io::Result<Box<dyn Stream>> {
        let timeout = Some(Duration::from_secs(20));
        Ok(match &self.endpoint {
            Endpoint::Tcp(addr) => {
                let tcp = TcpStream::connect(addr)?;
                tcp.set_read_timeout(timeout)?;
                match &self.tls_root {
                    None => Box::new(tcp),
                    Some(pem) => Box::new(tls_stream(tcp, pem)),
                }
            }
            #[cfg(unix)]
            Endpoint::Unix(path) => {
                let stream = std::os::unix::net::UnixStream::connect(path)?;
                stream.set_read_timeout(timeout)?;
                Box::new(stream)
            }
        })
    }

    /// The bytes the server sent back, empty when the connection or handshake failed.
    pub fn exchange_raw(&self, bytes: &[u8]) -> Vec<u8> {
        self.exchange_parts([bytes])
    }

    fn exchange_parts(&self, parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Vec<u8> {
        let Ok(mut stream) = self.try_connect() else {
            return Vec::new();
        };
        // The server may answer and close before reading an oversized body.
        let mut request_bytes = 0usize;
        let mut write_error = None;
        for part in parts {
            request_bytes += part.as_ref().len();
            if write_error.is_none() {
                write_error = stream.write_all(part.as_ref()).err();
            }
        }
        let mut raw = Vec::new();
        let start = std::time::Instant::now();
        if let Err(error) = stream.read_to_end(&mut raw) {
            if raw.is_empty() {
                eprintln!(
                    "response read failed after {:?}: {error}; request bytes: {request_bytes}; write error: {write_error:?}",
                    start.elapsed(),
                );
            }
        }
        raw
    }

    pub fn exchange(&self, bytes: &[u8]) -> Resp {
        parse_response(&self.exchange_raw(bytes))
    }

    pub fn req<'a>(&'a self, method: &'a str, path: &str) -> Request<'a> {
        Request {
            server: self,
            method,
            path: path.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn addr(&self) -> SocketAddr {
        match &self.endpoint {
            Endpoint::Tcp(a) => *a,
            #[cfg(unix)]
            Endpoint::Unix(_) => panic!("unix endpoint"),
        }
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    #[cfg(unix)]
    pub fn signal(&self, name: &str) {
        let status = Command::new("kill")
            .args([&format!("-{name}"), &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[cfg(windows)]
    pub fn signal(&self, name: &str) {
        use windows_sys::Win32::System::Console::{GenerateConsoleCtrlEvent, CTRL_BREAK_EVENT};
        assert_eq!(name, "TERM", "Windows reloads must use the admin endpoints");
        // Each child owns a process group; never signal group zero (the test runner).
        let sent = unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, self.child.id()) };
        assert_ne!(sent, 0, "{}", std::io::Error::last_os_error());
    }

    /// Request a graceful shutdown and wait for a successful exit.
    pub fn stop(&mut self) -> bool {
        self.signal("TERM");
        self.child.wait().unwrap().success()
    }

    pub fn log_lines(&self) -> Vec<String> {
        std::fs::read_to_string(&self.stderr_path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    pub fn log_json(&self) -> Vec<Value> {
        self.log_lines()
            .iter()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|_| panic!("non-JSON log line {l:?}")))
            .collect()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        #[cfg(unix)]
        if let Endpoint::Unix(path) = &self.endpoint {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn tls_stream(tcp: TcpStream, root_pem: &[u8]) -> impl Read + Write {
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, ServerName};
    let mut roots = rustls::RootCertStore::empty();
    for cert in CertificateDer::pem_slice_iter(root_pem) {
        roots.add(cert.unwrap()).unwrap();
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    let conn =
        rustls::ClientConnection::new(Arc::new(config), ServerName::try_from("localhost").unwrap())
            .unwrap();
    rustls::StreamOwned::new(conn, tcp)
}

/// Deterministic pseudo-random embedding for `seed`.
pub fn embedding(seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f64 / 1000.0 - 1.0
        })
        .collect()
}

pub fn doc(id: &str, seed: u64) -> Value {
    json!({"id": id, "text": format!("document {id}"), "vector": embedding(seed), "metadata": {"seed": seed}})
}

/// Create a key with the binary's own `keygen` and return its token.
pub fn keygen(file: &Path, id: &str, tenants: &str, scopes: &str) -> String {
    let out = Command::new(BIN)
        .args([
            "keygen",
            "--id",
            id,
            "--tenants",
            tenants,
            "--scopes",
            scopes,
            "--keys-file",
        ])
        .arg(file)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}
