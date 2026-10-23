// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The migration framework behind `hms-format upgrade`.
//!
//! A `Migration` moves one artefact from one version to the next. `MIGRATIONS` is the
//! registry; a format change adds an entry (and a golden store, see `tests/golden/`). The
//! planner chains entries until the artefact reaches the version this build writes, so a
//! store several versions behind upgrades in one run.
//!
//! The source is never modified. The upgraded store is staged next to the destination,
//! verified, and published with one rename.

use std::fs::File;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use serde::Serialize;

use super::scan::{self, SegmentEnd};
use super::verify::{verify_path, VerifyOptions};
use super::{Report, Verdict, STORAGE_FORMAT_CURRENT};
use crate::core::storage::StagedArena;
use crate::core::store_lock::{StoreLock, LOCK_FILE};

const ARENA_DIRECTORY: &str = "vectors_data.bin";

/// One step of a plan, as shown to the operator.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradeStep {
    pub id: &'static str,
    pub artefact: &'static str,
    pub from: u32,
    pub to: u32,
    pub description: &'static str,
}

/// What a store needs to reach the current format.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradePlan {
    pub source: String,
    /// Arena manifest version found (1 for a store with no manifest).
    pub storage_format: u32,
    pub target_storage_format: u32,
    pub up_to_date: bool,
    pub steps: Vec<UpgradeStep>,
}

/// What an upgrade did.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradeReport {
    pub plan: UpgradePlan,
    pub destination: Option<String>,
    pub frames: usize,
    pub verification: Option<Report>,
}

/// Everything a step may use. Steps run in registry order against the same staging directory.
struct StepContext<'a> {
    source: &'a Path,
    staging: &'a Path,
    frames: usize,
    /// `(raw_len, crc)` of every frame written, in order, to compare with what the staged
    /// arena scans back as.
    written: Vec<(u32, u32)>,
}

struct Migration {
    step: UpgradeStep,
    /// True if the store, at `detected` arena version, needs this step.
    applies: fn(&Detected) -> bool,
    run: fn(&mut StepContext<'_>) -> Result<()>,
}

struct Detected {
    storage_format: u32,
    text_segment: Option<u8>,
    has_caches: bool,
}

/// Every upgrade step that exists. Order matters only between steps that apply together.
static MIGRATIONS: &[Migration] = &[
    Migration {
        step: UpgradeStep {
            id: "arena-1-to-4",
            artefact: "arena",
            from: 1,
            to: 4,
            description: "rewrite the journal frames of a manifest-less store into a format-4 \
                          arena (1 GiB segments allocated in 64 MiB steps, end markers, \
                          checksummed manifest)",
        },
        applies: |d| d.storage_format == 1,
        run: rewrite_arena,
    },
    Migration {
        step: UpgradeStep {
            id: "arena-2-to-4",
            artefact: "arena",
            from: 2,
            to: 4,
            description: "rewrite the journal frames of a format-2 arena (fully preallocated 1 \
                          GiB segments, zero terminator) into a format-4 arena",
        },
        applies: |d| d.storage_format == 2,
        run: rewrite_arena,
    },
    Migration {
        step: UpgradeStep {
            id: "arena-3-to-4",
            artefact: "arena",
            from: 3,
            to: 4,
            description: "rewrite the journal frames of a format-3 arena (64 MiB fixed \
                          segments) into a format-4 arena with 1 GiB segments",
        },
        applies: |d| d.storage_format == 3,
        run: rewrite_arena,
    },
    Migration {
        step: UpgradeStep {
            id: "text-segment-1-to-2",
            artefact: "text-segment",
            from: 1,
            to: 2,
            description: "drop documents.text (a cache; the engine rebuilds it with the current \
                          header from the journal on first open)",
        },
        applies: |d| {
            d.text_segment
                .is_none_or(|v| v < super::TEXT_SEGMENT_VERSION)
        },
        run: |_| Ok(()),
    },
    Migration {
        step: UpgradeStep {
            id: "index-caches",
            artefact: "index-caches",
            from: 0,
            to: 0,
            description: "drop indices.json and the trained NSG/IVF files: they are bound to \
                          the old journal length and the engine retrains them",
        },
        applies: |d| d.has_caches,
        run: |_| Ok(()),
    },
];

fn detect(source: &Path, report: &Report) -> Detected {
    Detected {
        storage_format: report
            .versions
            .storage_format
            .unwrap_or(STORAGE_FORMAT_CURRENT),
        text_segment: source
            .join("documents.text")
            .exists()
            .then_some(report.versions.text_segment.unwrap_or(1)),
        has_caches: source.join("indices.json").exists(),
    }
}

/// Plans the upgrade of the store at `source` without touching it.
pub fn plan_upgrade(source: &Path) -> Result<UpgradePlan> {
    let report = verify_path(source, &VerifyOptions::default())?;
    plan_from(source, &report)
}

fn plan_from(source: &Path, report: &Report) -> Result<UpgradePlan> {
    ensure!(
        report.tenants.is_empty(),
        "{} is a server data directory; plan each tenant store, or upgrade the directory",
        source.display()
    );
    refuse_unupgradable(report)?;
    let detected = detect(source, report);
    let arena_current = detected.storage_format == STORAGE_FORMAT_CURRENT;
    let steps = if arena_current {
        Vec::new()
    } else {
        MIGRATIONS
            .iter()
            .filter(|m| (m.applies)(&detected))
            .map(|m| m.step.clone())
            .collect()
    };
    Ok(UpgradePlan {
        source: source.display().to_string(),
        storage_format: detected.storage_format,
        target_storage_format: STORAGE_FORMAT_CURRENT,
        up_to_date: arena_current,
        steps,
    })
}

fn refuse_unupgradable(report: &Report) -> Result<()> {
    match report.verdict {
        Verdict::Ok | Verdict::Recoverable => Ok(()),
        Verdict::Unsupported => {
            let reasons: Vec<_> = report
                .findings
                .iter()
                .filter(|f| f.severity == super::Severity::Unsupported)
                .map(|f| format!("{}: {}", f.code, f.message))
                .collect();
            bail!(
                "this build cannot upgrade the store ({}); a store with no store.json or a \
                 pre-HMS-TXN-2 journal is rebuilt with `hms-admin reencode`, a store from a \
                 newer release needs that release's hms-format",
                reasons.join("; ")
            )
        }
        Verdict::Corrupt => {
            let reasons: Vec<_> = report
                .findings
                .iter()
                .filter(|f| f.severity == super::Severity::Error)
                .map(|f| format!("{}: {}", f.code, f.message))
                .collect();
            bail!(
                "the store is damaged and is not upgraded: {}. Run `hms-format verify` for \
                 the full report; restore a snapshot first",
                reasons.join("; ")
            )
        }
    }
}

/// Creates the parent directory of `destination`, which must not exist yet, and returns it.
fn prepare_destination(destination: &Path) -> Result<&Path> {
    ensure!(
        !destination.exists(),
        "upgrade destination already exists: {}",
        destination.display()
    );
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    Ok(parent)
}

/// Upgrades the store at `source` into a new directory `destination` (which must not exist).
/// `dry_run` plans and verifies the source only. If the store is already current, nothing
/// is written and the report says so.
pub fn upgrade_store(source: &Path, destination: &Path, dry_run: bool) -> Result<UpgradeReport> {
    let sniff = verify_path(source, &VerifyOptions::default())?;
    if !sniff.tenants.is_empty() {
        return upgrade_data_dir(source, destination, dry_run, &sniff);
    }
    let plan = plan_from(source, &sniff)?;
    if plan.up_to_date || dry_run {
        return Ok(UpgradeReport {
            plan,
            destination: None,
            frames: sniff.frames,
            verification: None,
        });
    }
    let source_canonical = source.canonicalize()?;
    let parent = prepare_destination(destination)?;
    ensure!(
        !parent.canonicalize()?.starts_with(&source_canonical),
        "upgrade destination cannot be inside the source store"
    );
    let _lock = StoreLock::acquire(source)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let staging = parent.join(format!(".hms-upgrade-{}-{nonce}", std::process::id()));
    let outcome = stage_and_publish(source, &staging, destination, plan.clone());
    if outcome.is_err() && staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    outcome
}

fn upgrade_data_dir(
    source: &Path,
    destination: &Path,
    dry_run: bool,
    sniff: &Report,
) -> Result<UpgradeReport> {
    let mut plans = Vec::new();
    for tenant in &sniff.tenants {
        let dir = source
            .join("tenants")
            .join(tenant_dir(source, &tenant.name)?);
        plans.push(plan_from(&dir, &tenant.report)?);
    }
    let behind = plans.iter().any(|p| !p.up_to_date);
    let plan = UpgradePlan {
        source: source.display().to_string(),
        storage_format: plans
            .iter()
            .map(|p| p.storage_format)
            .min()
            .unwrap_or(STORAGE_FORMAT_CURRENT),
        target_storage_format: STORAGE_FORMAT_CURRENT,
        up_to_date: !behind,
        steps: plans
            .iter()
            .flat_map(|p| p.steps.clone())
            .fold(Vec::new(), |mut acc, s| {
                if !acc.iter().any(|a: &UpgradeStep| a.id == s.id) {
                    acc.push(s);
                }
                acc
            }),
    };
    if dry_run || !behind {
        return Ok(UpgradeReport {
            plan,
            destination: None,
            frames: sniff.tenants.iter().map(|t| t.report.frames).sum(),
            verification: None,
        });
    }
    let parent = prepare_destination(destination)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let staging = parent.join(format!(".hms-upgrade-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<usize> {
        std::fs::create_dir_all(staging.join("tenants"))?;
        let mut frames = 0;
        for tenant in &sniff.tenants {
            let name = tenant_dir(source, &tenant.name)?;
            let from = source.join("tenants").join(&name);
            let to = staging.join("tenants").join(&name);
            let _lock = StoreLock::acquire(&from)?;
            let tenant_plan = plan_from(&from, &tenant.report)?;
            if tenant_plan.up_to_date {
                copy_store_files(&from, &to, &["documents.text"])?;
                frames += tenant.report.frames;
            } else {
                let staged = stage_store(&from, &to)?;
                frames += staged.frames;
            }
        }
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            if entry.file_name() != "tenants" && entry.file_type()?.is_file() {
                std::fs::copy(entry.path(), staging.join(entry.file_name()))?;
            }
        }
        std::fs::rename(&staging, destination)?;
        Ok(frames)
    })();
    match result {
        Ok(frames) => Ok(UpgradeReport {
            plan,
            destination: Some(destination.display().to_string()),
            frames,
            verification: Some(verify_path(destination, &VerifyOptions::default())?),
        }),
        Err(error) => {
            let _ = std::fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

/// Directory name of tenant `name` (hex of its UTF-8 bytes), or the name itself if it was
/// reported undecoded.
fn tenant_dir(source: &Path, name: &str) -> Result<String> {
    let hex: String = name.bytes().map(|b| format!("{b:02x}")).collect();
    if source.join("tenants").join(&hex).is_dir() {
        Ok(hex)
    } else if source.join("tenants").join(name).is_dir() {
        Ok(name.to_owned())
    } else {
        bail!("tenant directory for {name:?} not found")
    }
}

fn stage_and_publish(
    source: &Path,
    staging: &Path,
    destination: &Path,
    plan: UpgradePlan,
) -> Result<UpgradeReport> {
    let staged = stage_store(source, staging)?;
    let verification = verify_path(staging, &VerifyOptions::default())?;
    ensure!(
        matches!(verification.verdict, Verdict::Ok),
        "the upgraded store does not verify: {:?}",
        verification.findings
    );
    ensure!(
        verification.frames == staged.frames,
        "upgraded store holds {} frames, {} were written",
        verification.frames,
        staged.frames
    );
    std::fs::rename(staging, destination).with_context(|| {
        format!(
            "failed to publish {} as {}",
            staging.display(),
            destination.display()
        )
    })?;
    if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
        crate::core::durable_file::sync_directory(parent)?;
    }
    let mut verification = verification;
    verification.path = destination.display().to_string();
    Ok(UpgradeReport {
        plan,
        destination: Some(destination.display().to_string()),
        frames: staged.frames,
        verification: Some(verification),
    })
}

/// Copies the files that survive an upgrade unchanged; the caches and the lock are skipped.
fn copy_store_files(source: &Path, staging: &Path, skip: &[&str]) -> Result<()> {
    std::fs::create_dir_all(staging)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let cache = name.starts_with("nsg_index") || name.starts_with("ivf_index");
        let skipped = name == LOCK_FILE
            || name == ARENA_DIRECTORY
            || name == "indices.json"
            || name == "indices.json.crc32"
            || cache
            || skip.contains(&name.as_str());
        if skipped {
            continue;
        }
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "upgrade refuses symbolic link {name}");
        if kind.is_file() {
            let target = staging.join(&name);
            std::fs::copy(entry.path(), &target)?;
            std::fs::set_permissions(&target, entry.metadata()?.permissions())?;
        } else {
            bail!("upgrade does not know how to carry the directory {name}");
        }
    }
    Ok(())
}

struct Staged {
    frames: usize,
}

/// Runs the plan: carries the unchanged files, then every step the registry selects.
fn stage_store(source: &Path, staging: &Path) -> Result<Staged> {
    let report = verify_path(source, &VerifyOptions::default())?;
    refuse_unupgradable(&report)?;
    let detected = detect(source, &report);
    copy_store_files(source, staging, &["documents.text"])?;
    // `store.json` was copied byte for byte; rewrite its sidecar so the copy is checksummed
    // even when the source predates sidecars.
    let schema = staging.join("store.json");
    let bytes = std::fs::read(&schema)?;
    crate::core::durable_file::atomic_write(&schema, &bytes)?;
    let mut context = StepContext {
        source,
        staging,
        frames: 0,
        written: Vec::new(),
    };
    for migration in MIGRATIONS.iter().filter(|m| (m.applies)(&detected)) {
        (migration.run)(&mut context)
            .with_context(|| format!("upgrade step {} failed", migration.step.id))?;
    }
    ensure!(context.frames > 0, "the store has no journal frames");
    // The staged arena must hold exactly the frames that were read, in order.
    let mut scanned = Vec::new();
    let arena = staging.join(ARENA_DIRECTORY);
    for id in 0.. {
        let path = arena.join(format!("seg_{id}.bin"));
        if !path.exists() {
            break;
        }
        let map = unsafe { Mmap::map(&File::open(&path)?)? };
        let scan = scan::scan_segment(&map, true, |f, _| scanned.push((f.raw_len, f.crc)));
        ensure!(
            matches!(scan.end, SegmentEnd::Marker | SegmentEnd::Exhausted),
            "staged segment {id} does not end cleanly: {:?}",
            scan.end
        );
    }
    ensure!(
        scanned == context.written,
        "staged arena differs from the frames read from the source"
    );
    Ok(Staged {
        frames: context.frames,
    })
}

/// Reads every committed frame of the source arena in order and appends it to a new
/// format-4 arena. Payloads are opaque here: encrypted frames are carried as they are.
fn rewrite_arena(context: &mut StepContext<'_>) -> Result<()> {
    let base = context.source.join(ARENA_DIRECTORY);
    let generation = crate::core::durable_file::arena_generation(&base)?;
    let ids = super::verify::list_segments(&generation)?;
    ensure!(!ids.is_empty(), "the source arena has no segments");
    // Frames are staged without a sync each; `finish` makes the arena durable once.
    let directory = context.staging.join(ARENA_DIRECTORY);
    std::fs::create_dir_all(&directory)?;
    let mut target = StagedArena::new(&directory, None)?;
    let mut failure: Option<anyhow::Error> = None;
    for (n, id) in ids.iter().enumerate() {
        let path = generation.join(format!("seg_{id}.bin"));
        let map = unsafe { Mmap::map(&File::open(&path)?)? };
        let scan = scan::scan_segment(&map, n + 1 == ids.len(), |frame, payload| {
            if failure.is_none() {
                match target.write_slice(payload) {
                    Ok(_) => {
                        context.frames += 1;
                        context.written.push((frame.raw_len, frame.crc));
                    }
                    Err(error) => failure = Some(error),
                }
            }
        });
        if let Some(error) = failure.take() {
            return Err(error);
        }
        if let SegmentEnd::Corrupt { reason, .. } = scan.end {
            bail!(
                "{} is corrupt at byte {} ({reason})",
                path.display(),
                scan.end_offset
            )
        }
    }
    target.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every registered step moves forward and names an artefact; the arena steps reach the
    /// version this build writes.
    #[test]
    fn the_registry_is_consistent() {
        let mut ids = std::collections::BTreeSet::new();
        for migration in MIGRATIONS {
            assert!(ids.insert(migration.step.id), "duplicate step id");
            if migration.step.artefact == "arena" {
                assert!(migration.step.from < migration.step.to);
                assert_eq!(migration.step.to, STORAGE_FORMAT_CURRENT);
            }
        }
        for from in super::super::STORAGE_FORMAT_OLDEST..STORAGE_FORMAT_CURRENT {
            assert!(
                MIGRATIONS
                    .iter()
                    .any(|m| m.step.artefact == "arena" && m.step.from == from),
                "no arena step leaves format {from}"
            );
        }
    }

    #[test]
    fn data_directory_paths_are_the_hex_of_the_name() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("tenants/616c696365")).unwrap();
        assert_eq!(tenant_dir(root.path(), "alice").unwrap(), "616c696365");
        assert!(tenant_dir(root.path(), "bob").is_err());
    }
}
