// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Authentication and authorization.
//!
//! A token is `warm_<id>.<64 hex>`. The server stores only `sha256(token)` per key; verification
//! hashes the presented token and compares digests in constant time, so neither the stored value
//! nor the token length leaks through timing. Each key is bound to tenants and scopes.
//!
//! `WARM_API_KEY` stays supported as one key for every tenant with every scope. `--no-auth` grants
//! the same to every caller and is refused off loopback (see `Config::validate`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, ensure, Context, Result};
use axum::http::{header, HeaderMap, StatusCode};
use clap::Args;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::error::ApiError;
use crate::tenants::valid_tenant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scope {
    Read,
    Write,
    Admin,
}

impl Scope {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "admin" => Some(Self::Admin),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
        }
    }
}

/// `admin` includes `read` and `write`; `write` does not include `read`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeSet(u8);

impl ScopeSet {
    pub const ALL: Self = Self(0b111);

    fn bit(scope: Scope) -> u8 {
        match scope {
            Scope::Read => 1,
            Scope::Write => 2,
            Scope::Admin => 4,
        }
    }

    pub fn from_names(names: &[String]) -> Result<Self> {
        ensure!(!names.is_empty(), "scopes must not be empty");
        let mut bits = 0;
        for name in names {
            let scope = Scope::parse(name)
                .with_context(|| format!("unknown scope {name:?}; use read, write or admin"))?;
            bits |= Self::bit(scope);
        }
        Ok(Self(bits))
    }

    pub fn has(self, scope: Scope) -> bool {
        self.0 & Self::bit(Scope::Admin) != 0 || self.0 & Self::bit(scope) != 0
    }
}

#[derive(Clone, Debug)]
pub enum TenantAccess {
    Any,
    Only(HashSet<String>),
}

#[derive(Clone, Debug)]
pub struct Principal {
    pub key_id: String,
    pub tenants: TenantAccess,
    pub scopes: ScopeSet,
}

impl Principal {
    pub fn require(&self, scope: Scope) -> Result<(), ApiError> {
        if self.scopes.has(scope) {
            Ok(())
        } else {
            Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "forbidden",
                format!("this key lacks the {} scope", scope.name()),
            ))
        }
    }

    pub fn is_global(&self) -> bool {
        matches!(self.tenants, TenantAccess::Any)
    }

    pub fn require_global(&self) -> Result<(), ApiError> {
        if self.is_global() {
            Ok(())
        } else {
            Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "tenant_forbidden",
                "this operation needs a key that is valid for every tenant",
            ))
        }
    }

    /// The tenant a request addresses: `X-Tenant-ID`, else the key's only tenant, else `default`.
    pub fn tenant(&self, headers: &HeaderMap) -> Result<String, ApiError> {
        let mut values = headers.get_all("x-tenant-id").iter();
        let value = values.next();
        if values.next().is_some() {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_tenant",
                "X-Tenant-ID must occur at most once",
            ));
        }
        let name = match value {
            Some(value) => match value.to_str() {
                Ok(v) if valid_tenant(v) => v.to_string(),
                _ => {
                    return Err(ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_tenant",
                        "X-Tenant-ID must be 1..=64 characters of [A-Za-z0-9_-]",
                    ))
                }
            },
            None => match &self.tenants {
                TenantAccess::Only(set) if set.len() == 1 => {
                    set.iter().next().cloned().unwrap_or_default()
                }
                _ => "default".to_string(),
            },
        };
        self.allow_tenant(&name)?;
        Ok(name)
    }

    pub fn allow_tenant(&self, name: &str) -> Result<(), ApiError> {
        match &self.tenants {
            TenantAccess::Any => Ok(()),
            TenantAccess::Only(set) if set.contains(name) => Ok(()),
            TenantAccess::Only(_) => Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "tenant_forbidden",
                "this key is not valid for that tenant",
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyEntry {
    pub id: String,
    /// `sha256:<64 hex>` of the full token.
    pub hash: String,
    /// Tenant ids, or `["*"]` for every tenant.
    pub tenants: Vec<String>,
    pub scopes: Vec<String>,
    /// Unix seconds before which the key is not yet valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_before: Option<u64>,
    /// Unix seconds from which the key is no longer valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub disabled: bool,
}

#[derive(Default, Serialize, Deserialize)]
struct KeyFile {
    keys: Vec<KeyEntry>,
}

struct LoadedKey {
    digest: [u8; 32],
    principal: Principal,
    not_before: u64,
    expires_at: u64,
    disabled: bool,
}

#[derive(Default)]
struct KeyStore {
    keys: HashMap<String, LoadedKey>,
}

fn valid_key_id(id: &str) -> bool {
    (1..=32).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn parse_digest(hash: &str) -> Result<[u8; 32]> {
    let hex = hash
        .strip_prefix("sha256:")
        .context("hash must start with \"sha256:\"")?;
    ensure!(hex.len() == 64, "hash must be 64 hex digits");
    let bytes = unhex(hex).context("hash is not hex")?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("hash must be 64 hex digits"))
}

/// Decode pairs of hex digits; `None` for any pair that is not two hex digits. A trailing odd
/// digit is ignored.
pub fn unhex(text: &str) -> Option<Vec<u8>> {
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect()
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest_token(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn load_key_file(path: &Path) -> Result<KeyStore> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let file: KeyFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not a valid key file", path.display()))?;
    let mut store = KeyStore::default();
    for entry in file.keys {
        ensure!(
            valid_key_id(&entry.id),
            "key id {:?} must be 1..=32 of [A-Za-z0-9_-]",
            entry.id
        );
        let digest = parse_digest(&entry.hash).with_context(|| format!("key {}", entry.id))?;
        ensure!(!entry.tenants.is_empty(), "key {} has no tenants", entry.id);
        let tenants = if entry.tenants.iter().any(|t| t == "*") {
            ensure!(
                entry.tenants.len() == 1,
                "key {}: \"*\" cannot be combined with tenant ids",
                entry.id
            );
            TenantAccess::Any
        } else {
            for t in &entry.tenants {
                ensure!(valid_tenant(t), "key {}: invalid tenant id {t:?}", entry.id);
            }
            TenantAccess::Only(entry.tenants.iter().cloned().collect())
        };
        let scopes =
            ScopeSet::from_names(&entry.scopes).with_context(|| format!("key {}", entry.id))?;
        let loaded = LoadedKey {
            digest,
            principal: Principal {
                key_id: entry.id.clone(),
                tenants,
                scopes,
            },
            not_before: entry.not_before.unwrap_or(0),
            expires_at: entry.expires_at.unwrap_or(u64::MAX),
            disabled: entry.disabled,
        };
        ensure!(
            store.keys.insert(entry.id.clone(), loaded).is_none(),
            "duplicate key id {}",
            entry.id
        );
    }
    Ok(store)
}

pub struct Authenticator {
    mode: Mode,
    legacy: Option<[u8; 32]>,
    store: RwLock<Arc<KeyStore>>,
    keys_file: Option<PathBuf>,
}

#[derive(PartialEq, Eq)]
enum Mode {
    Open,
    Keys,
}

impl Authenticator {
    pub fn new(
        no_auth: bool,
        legacy_key: Option<&str>,
        keys_file: Option<PathBuf>,
    ) -> Result<Self> {
        let store = match &keys_file {
            Some(path) => load_key_file(path)?,
            None => KeyStore::default(),
        };
        Ok(Self {
            mode: if no_auth { Mode::Open } else { Mode::Keys },
            legacy: legacy_key.map(digest_token),
            store: RwLock::new(Arc::new(store)),
            keys_file,
        })
    }

    pub fn is_open(&self) -> bool {
        self.mode == Mode::Open
    }

    pub fn has_credentials(&self) -> bool {
        self.legacy.is_some() || self.keys_file.is_some()
    }

    /// Re-read the key file. A broken file leaves the previous keys in force.
    pub fn reload(&self) -> Result<usize> {
        let Some(path) = &self.keys_file else {
            bail!("no --keys-file is configured");
        };
        let store = load_key_file(path)?;
        let count = store.keys.len();
        *self.store.write() = Arc::new(store);
        Ok(count)
    }

    pub fn authenticate(&self, headers: &HeaderMap) -> Result<Principal, ApiError> {
        if self.mode == Mode::Open {
            return Ok(Principal {
                key_id: "anonymous".into(),
                tenants: TenantAccess::Any,
                scopes: ScopeSet::ALL,
            });
        }
        let mut values = headers.get_all(header::AUTHORIZATION).iter();
        let token = values
            .next()
            .filter(|_| values.next().is_none())
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split_once(' '))
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, token)| token.trim_start_matches(' '))
            .unwrap_or("");
        let digest = digest_token(token);
        // Evaluate every comparison before branching so timing does not reveal which matched.
        let legacy_ok = self
            .legacy
            .as_ref()
            .is_some_and(|l| bool::from(l.ct_eq(&digest)));
        let store = self.store.read().clone();
        let id = token
            .strip_prefix("warm_")
            .or_else(|| token.strip_prefix("hms_"))
            .and_then(|rest| rest.split_once('.'))
            .map(|(id, _)| id);
        let keyed = id.and_then(|id| store.keys.get(id));
        let keyed_ok = keyed.is_some_and(|k| {
            let now = now_secs();
            bool::from(k.digest.ct_eq(&digest))
                && !k.disabled
                && now >= k.not_before
                && now < k.expires_at
        });
        if legacy_ok {
            return Ok(Principal {
                key_id: "legacy".into(),
                tenants: TenantAccess::Any,
                scopes: ScopeSet::ALL,
            });
        }
        match (keyed, keyed_ok) {
            (Some(k), true) => Ok(k.principal.clone()),
            _ => {
                let mut error = ApiError::new(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "missing or invalid API key",
                );
                error.www_authenticate = true;
                Err(error)
            }
        }
    }
}

#[derive(Args)]
pub struct KeygenArgs {
    /// Key id (1..=32 of [A-Za-z0-9_-]); it appears in logs, never the secret.
    #[arg(long)]
    pub id: String,
    /// Comma-separated tenant ids the key may use, or `*` for every tenant.
    #[arg(long, value_delimiter = ',', required = true)]
    pub tenants: Vec<String>,
    /// Comma-separated scopes: read, write, admin.
    #[arg(long, value_delimiter = ',', default_value = "read,write")]
    pub scopes: Vec<String>,
    /// Expire the key after this many days.
    #[arg(long)]
    pub expires_in_days: Option<u64>,
    /// Append the entry to this key file (created with mode 0600 if missing). Without it the
    /// entry is printed on stderr for you to place.
    #[arg(long, env = writerslogic_warm::compatibility::env_name("WARM_KEYS_FILE", "HMS_KEYS_FILE"))]
    pub keys_file: Option<PathBuf>,
}

/// Build a fresh token and its stored entry.
pub fn new_key(args: &KeygenArgs) -> Result<(String, KeyEntry)> {
    ensure!(
        valid_key_id(&args.id),
        "--id must be 1..=32 of [A-Za-z0-9_-]"
    );
    ScopeSet::from_names(&args.scopes)?;
    if args.tenants.iter().any(|t| t == "*") {
        ensure!(
            args.tenants.len() == 1,
            "\"*\" cannot be combined with tenant ids"
        );
    } else {
        for t in &args.tenants {
            ensure!(valid_tenant(t), "invalid tenant id {t:?}");
        }
    }
    let secret: [u8; 32] = rand::random();
    let token = format!("warm_{}.{}", args.id, hex(&secret));
    let entry = KeyEntry {
        id: args.id.clone(),
        hash: format!("sha256:{}", hex(&digest_token(&token))),
        tenants: args.tenants.clone(),
        scopes: args.scopes.clone(),
        not_before: None,
        expires_at: args
            .expires_in_days
            .map(|d| now_secs() + d.saturating_mul(86_400)),
        disabled: false,
    };
    Ok((token, entry))
}

pub fn keygen(args: &KeygenArgs) -> Result<()> {
    let (token, entry) = new_key(args)?;
    match &args.keys_file {
        Some(path) => append_entry(path, entry)?,
        None => eprintln!("{}", serde_json::to_string_pretty(&entry)?),
    }
    println!("{token}");
    Ok(())
}

fn append_entry(path: &Path, entry: KeyEntry) -> Result<()> {
    let mut file: KeyFile = match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("{} is not a valid key file", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => KeyFile::default(),
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    ensure!(
        file.keys.iter().all(|k| k.id != entry.id),
        "key id {} already exists in {}",
        entry.id,
        path.display()
    );
    file.keys.push(entry);
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(dir)?;
    std::io::Write::write_all(
        &mut temporary,
        serde_json::to_string_pretty(&file)?.as_bytes(),
    )?;
    temporary.as_file().sync_all()?;
    // NamedTempFile is created 0600; persist keeps that mode.
    temporary
        .persist(path)
        .with_context(|| format!("cannot write {}", path.display()))?;
    // The rename is durable only once the directory holding the new entry is synced.
    crate::vindex::sync_dir(dir).with_context(|| format!("cannot sync {}", dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_header_syntax_preserves_key_file_scope_and_secret_case() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let (token, entry) = new_key(&args("Scoped", &["acme"], &["read"])).unwrap();
        append_entry(&path, entry).unwrap();
        let auth = Authenticator::new(false, None, Some(path)).unwrap();
        for prefix in ["Bearer ", "bearer ", "bEaReR   "] {
            let mut headers = HeaderMap::new();
            headers.insert(
                header::AUTHORIZATION,
                format!("{prefix}{token}").parse().unwrap(),
            );
            let principal = auth.authenticate(&headers).unwrap();
            assert_eq!(principal.tenant(&HeaderMap::new()).unwrap(), "acme");
            assert!(principal.scopes.has(Scope::Read));
            assert!(!principal.scopes.has(Scope::Write));
            headers.append(
                header::AUTHORIZATION,
                format!("Bearer {token}").parse().unwrap(),
            );
            assert_eq!(
                auth.authenticate(&headers).unwrap_err().code,
                "unauthorized"
            );
        }
        assert!(auth.authenticate(&headers(&token.to_uppercase())).is_err());
    }

    #[test]
    fn legacy_key_prefix_keeps_its_original_digest_and_tenant_scope() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let (current, mut entry) = new_key(&args("legacy", &["acme"], &["read"])).unwrap();
        let legacy = current.replacen("warm_", "hms_", 1);
        entry.hash = format!("sha256:{}", hex(&digest_token(&legacy)));
        append_entry(&path, entry).unwrap();
        let auth = Authenticator::new(false, None, Some(path)).unwrap();
        let principal = auth.authenticate(&headers(&legacy)).unwrap();
        assert_eq!(principal.tenant(&HeaderMap::new()).unwrap(), "acme");
        assert!(principal.scopes.has(Scope::Read));
        assert!(!principal.scopes.has(Scope::Write));
        assert!(auth.authenticate(&headers(&current)).is_err());
    }

    fn headers(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        h
    }

    fn args(id: &str, tenants: &[&str], scopes: &[&str]) -> KeygenArgs {
        KeygenArgs {
            id: id.into(),
            tenants: tenants.iter().map(|s| s.to_string()).collect(),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            expires_in_days: None,
            keys_file: None,
        }
    }

    #[test]
    fn keys_are_bound_to_tenants_and_scopes_and_stored_hashed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let a = new_key(&args("a", &["acme"], &["read"])).unwrap();
        let b = new_key(&args("b", &["*"], &["admin"])).unwrap();
        append_entry(&path, a.1).unwrap();
        append_entry(&path, b.1).unwrap();
        assert!(!std::fs::read_to_string(&path).unwrap().contains(&a.0));
        let auth = Authenticator::new(false, None, Some(path.clone())).unwrap();

        let p = auth.authenticate(&headers(&a.0)).unwrap();
        assert!(p.scopes.has(Scope::Read) && !p.scopes.has(Scope::Write));
        assert_eq!(p.tenant(&HeaderMap::new()).unwrap(), "acme");
        let mut other = HeaderMap::new();
        other.insert("x-tenant-id", "beta".parse().unwrap());
        assert_eq!(p.tenant(&other).unwrap_err().code, "tenant_forbidden");

        let p = auth.authenticate(&headers(&b.0)).unwrap();
        assert!(p.scopes.has(Scope::Write), "admin implies write");
        assert!(p.is_global());

        // Same id, wrong secret; truncated token; no header.
        let forged = format!("warm_a.{}", "0".repeat(64));
        for bad in [
            forged.as_str(),
            &a.0[..a.0.len() - 1],
            "",
            "warm_a",
            "warm_.x",
        ] {
            assert_eq!(
                auth.authenticate(&headers(bad)).unwrap_err().code,
                "unauthorized"
            );
        }
        assert!(auth.authenticate(&HeaderMap::new()).is_err());
    }

    #[test]
    fn rotation_overlaps_then_old_key_stops_working_after_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let old = new_key(&args("old", &["acme"], &["write"])).unwrap();
        append_entry(&path, old.1.clone()).unwrap();
        let auth = Authenticator::new(false, None, Some(path.clone())).unwrap();
        let new = new_key(&args("new", &["acme"], &["write"])).unwrap();
        append_entry(&path, new.1).unwrap();
        assert!(
            auth.authenticate(&headers(&new.0)).is_err(),
            "unloaded until reload"
        );
        assert_eq!(auth.reload().unwrap(), 2);
        assert!(auth.authenticate(&headers(&old.0)).is_ok());
        assert!(auth.authenticate(&headers(&new.0)).is_ok());
        let mut disabled = old.1;
        disabled.disabled = true;
        let file = KeyFile {
            keys: vec![disabled],
        };
        std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
        auth.reload().unwrap();
        assert!(auth.authenticate(&headers(&old.0)).is_err());
        // A broken file leaves the previous keys in force.
        std::fs::write(&path, b"{").unwrap();
        assert!(auth.reload().is_err());
        assert_eq!(auth.store.read().keys.len(), 1);
    }

    #[test]
    fn expiry_and_not_before_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let (token, mut entry) = new_key(&args("k", &["*"], &["read"])).unwrap();
        entry.expires_at = Some(now_secs() - 1);
        append_entry(&path, entry.clone()).unwrap();
        let auth = Authenticator::new(false, None, Some(path.clone())).unwrap();
        assert!(auth.authenticate(&headers(&token)).is_err());
        entry.expires_at = None;
        entry.not_before = Some(now_secs() + 3600);
        std::fs::write(
            &path,
            serde_json::to_vec(&KeyFile { keys: vec![entry] }).unwrap(),
        )
        .unwrap();
        auth.reload().unwrap();
        assert!(auth.authenticate(&headers(&token)).is_err());
    }

    #[test]
    fn legacy_key_grants_everything_and_open_mode_needs_no_header() {
        let auth = Authenticator::new(false, Some("s3cret"), None).unwrap();
        let p = auth.authenticate(&headers("s3cret")).unwrap();
        assert!(p.is_global() && p.scopes.has(Scope::Admin));
        assert!(auth.authenticate(&headers("s3cre")).is_err());
        let open = Authenticator::new(true, None, None).unwrap();
        assert!(open
            .authenticate(&HeaderMap::new())
            .unwrap()
            .scopes
            .has(Scope::Admin));
    }

    #[test]
    fn invalid_key_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.json");
        let good = format!("sha256:{}", "ab".repeat(32));
        for body in [
            format!(
                r#"{{"keys":[{{"id":"a b","hash":"{good}","tenants":["x"],"scopes":["read"]}}]}}"#
            ),
            r#"{"keys":[{"id":"a","hash":"sha256:zz","tenants":["x"],"scopes":["read"]}]}"#
                .to_string(),
            format!(r#"{{"keys":[{{"id":"a","hash":"{good}","tenants":[],"scopes":["read"]}}]}}"#),
            format!(
                r#"{{"keys":[{{"id":"a","hash":"{good}","tenants":["x"],"scopes":["root"]}}]}}"#
            ),
            format!(
                r#"{{"keys":[{{"id":"a","hash":"{good}","tenants":["*","x"],"scopes":["read"]}}]}}"#
            ),
            format!(
                r#"{{"keys":[{{"id":"a","hash":"{good}","tenants":["x"],"scopes":["read"]}},{{"id":"a","hash":"{good}","tenants":["x"],"scopes":["read"]}}]}}"#
            ),
        ] {
            std::fs::write(&path, body).unwrap();
            assert!(load_key_file(&path).is_err());
        }
    }
}
