// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded JSON-lines transport for the production conversation memory API.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use holographic_memory::core::memory_api::{
    ConversationContextLimits, ConversationInput, ConversationMemory, ConversationRecallOptions,
    MemoryError,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::HmsCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[cfg(feature = "local-models")]
use std::sync::Arc;

const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
const MAX_REPLY_BYTES: usize = 16 * 1024 * 1024;
const DIMENSIONS: u32 = 4096;

#[derive(Parser)]
#[command(
    name = "conversation-eval",
    about = "JSON-lines bridge for isolated conversation memory histories"
)]
struct Args {
    /// Existing local directory containing the three pinned Qwen model directories.
    #[arg(long, conflicts_with = "lexical")]
    models: Option<PathBuf>,
    /// Use lexical retrieval without loading any model.
    #[arg(long)]
    lexical: bool,
    /// Total fused weight of generated subqueries; zero retains only the main query and rewrite.
    #[arg(long, default_value_t = 0.0)]
    subquery_weight: f64,
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Initialize {
        #[serde(default, rename = "modelRoot")]
        model_root: Option<PathBuf>,
    },
    Ingest {
        conversation: ConversationInput,
    },
    Recall {
        query: String,
        #[serde(default = "profile_recall_options")]
        options: ConversationRecallOptions,
        #[serde(default, rename = "compactContext")]
        compact_context: bool,
        #[serde(default, rename = "contextExpansion")]
        context_expansion: Option<ConversationContextLimits>,
    },
    Reset {},
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Self::Initialize { .. } => "initialize",
            Self::Ingest { .. } => "ingest",
            Self::Recall { .. } => "recall",
            Self::Reset {} => "reset",
        }
    }
}

fn profile_recall_options() -> ConversationRecallOptions {
    ConversationRecallOptions::default()
}

#[derive(Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Timings {
    model_load_seconds: f64,
    ingest_seconds: f64,
    query_seconds: f64,
    reset_seconds: f64,
}

impl Timings {
    fn add(&mut self, other: &Self) {
        self.model_load_seconds += other.model_load_seconds;
        self.ingest_seconds += other.ingest_seconds;
        self.query_seconds += other.query_seconds;
        self.reset_seconds += other.reset_seconds;
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Counts {
    session_count: usize,
    source_count: usize,
    chunk_count: usize,
    ingest_attempts: usize,
    recall_attempts: usize,
}

struct History {
    memory: ConversationMemory,
    _directory: tempfile::TempDir,
    counts: Counts,
}

struct Backend {
    config: HmsConfig,
    profile: Value,
    #[cfg(feature = "local-models")]
    stages: Option<Arc<holographic_memory::core::models::ModelStages>>,
}

impl Backend {
    fn lexical() -> Self {
        Self {
            config: HmsConfig::default(),
            profile: json!({"mode": "lexical", "models": [], "extractFacts": false,
                            "rewriteQueries": false, "embeddingQueryTask": null, "stageParams": null}),
            #[cfg(feature = "local-models")]
            stages: None,
        }
    }

    #[cfg(not(feature = "local-models"))]
    fn models(_root: PathBuf, _subquery_weight: f64) -> anyhow::Result<Self> {
        anyhow::bail!("model stages are unavailable in this build")
    }

    #[cfg(feature = "local-models")]
    fn models(root: PathBuf, subquery_weight: f64) -> anyhow::Result<Self> {
        use holographic_memory::core::models::{
            default_device, Embedder, EmbeddingQueryTask, Generator, ModelSource, ModelStages,
            Reranker, StageParams, PINNED,
        };
        use holographic_memory::EmbeddingSpace;

        anyhow::ensure!(root.is_dir(), "local model root is unavailable");
        let definitions = [
            ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
            ("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B"),
            ("Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
        ];
        let mut receipts = Vec::new();
        let mut sources = Vec::new();
        for (directory, name) in definitions {
            let revision = PINNED
                .iter()
                .find(|(model, _)| *model == name)
                .map(|(_, revision)| *revision)
                .ok_or_else(|| anyhow::anyhow!("pinned model is unavailable"))?;
            let source = ModelSource::new(root.join(directory), revision);
            holographic_memory::core::models::check_revision(&source)?;
            receipts.push(json!({"model": name, "revision": revision,
                                 "files": model_files(&source.dir, directory)?}));
            sources.push(source);
        }
        let device = default_device()?;
        let mut embedder = Embedder::load(&sources[0], &device)?;
        embedder.set_query_task(EmbeddingQueryTask::Conversation);
        embedder.set_batch(4);
        let dimensions = embedder.dimensions();
        let mut reranker = Reranker::load(&sources[1], &device)?;
        reranker.set_batch(4);
        let mut generator = Generator::load(&sources[2], &device)?;
        generator.set_batch(1);
        let params = StageParams {
            orig_weight: 1.0,
            rewrite_weight: 1.0,
            subquery_weight,
            ..StageParams::default()
        };
        params.validate()?;
        let profile = json!({"mode": "models", "models": receipts, "device": format!("{device:?}"),
                             "extractFacts": false, "rewriteQueries": true,
                             "embeddingQueryTask": "conversation", "stageParams": &params,
                             "embeddingBatch": 4, "rerankBatch": 4, "generatorBatch": 1,
                             "modelLoadIncludesFileFingerprinting": true});
        let stages = ModelStages {
            embedder: Some(embedder),
            reranker: Some(reranker),
            generator: Some(generator),
            extract_facts: false,
            rewrite_queries: true,
            params,
        };
        Ok(Self {
            config: HmsConfig {
                embedding_space: Some(EmbeddingSpace {
                    model: definitions[0].1.into(),
                    revision: sources[0].revision.clone(),
                    dimensions,
                    normalization: "l2".into(),
                    metric: "cosine".into(),
                }),
                ..HmsConfig::default()
            },
            profile,
            stages: Some(Arc::new(stages)),
        })
    }

    fn history(&self) -> anyhow::Result<History> {
        let directory = tempfile::tempdir()?;
        let core = HmsCore::new(
            DIMENSIONS,
            Some(directory.path().display().to_string()),
            Some(self.config.clone()),
        )?;
        #[cfg(feature = "local-models")]
        core.set_model_stages(self.stages.clone())?;
        Ok(History {
            memory: ConversationMemory::new(core)?,
            _directory: directory,
            counts: Counts::default(),
        })
    }
}

#[cfg(feature = "local-models")]
fn model_files(root: &std::path::Path, directory: &str) -> anyhow::Result<Vec<Value>> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if matches!(
            name,
            "config.json" | "tokenizer.json" | "generation_config.json" | "REVISION"
        ) || path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            paths.push(path);
        }
    }
    let revision_metadata = root.join(".cache/huggingface/download/config.json.metadata");
    if revision_metadata.is_file() {
        paths.push(revision_metadata);
    }
    paths.sort();
    anyhow::ensure!(paths.len() <= 64, "too many model files");
    let mut receipts = Vec::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in paths {
        let mut file = std::fs::File::open(&path)?;
        let metadata = file.metadata()?;
        anyhow::ensure!(
            metadata.is_file() && metadata.len() <= 64 * 1024 * 1024 * 1024,
            "invalid model file"
        );
        let mut digest = Sha256::new();
        let mut size = 0u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
            size = size
                .checked_add(read as u64)
                .ok_or_else(|| anyhow::anyhow!("model file size overflow"))?;
        }
        let sha256: String = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        receipts.push(
            json!({"path": format!("{directory}/{}", path.strip_prefix(root)?.display()),
                             "sizeBytes": size, "sha256": sha256}),
        );
    }
    Ok(receipts)
}

struct Runtime {
    args: Args,
    backend: Option<Backend>,
    history: Option<History>,
    epoch: u64,
    cumulative: Timings,
}

impl Runtime {
    fn new(args: Args) -> Self {
        Self {
            args,
            backend: None,
            history: None,
            epoch: 0,
            cumulative: Timings::default(),
        }
    }

    fn execute(&mut self, command: Command, timings: &mut Timings) -> Result<Value, &'static str> {
        match command {
            Command::Initialize { model_root } => {
                if self.backend.is_some() {
                    return Err("already_initialized");
                }
                if !self.args.subquery_weight.is_finite() || self.args.subquery_weight < 0.0 {
                    return Err("invalid_subquery_weight");
                }
                if self.args.lexical && model_root.is_some() {
                    return Err("conflicting_backend_options");
                }
                let root = match (&self.args.models, model_root) {
                    (Some(configured), Some(requested)) if configured != &requested => {
                        return Err("conflicting_model_root")
                    }
                    (Some(configured), _) => Some(configured.clone()),
                    (None, requested) => requested,
                };
                if root.is_none() && !self.args.lexical {
                    return Err("backend_not_configured");
                }
                let started = Instant::now();
                let backend = root
                    .map(|root| Backend::models(root, self.args.subquery_weight))
                    .unwrap_or_else(|| Ok(Backend::lexical()));
                timings.model_load_seconds = started.elapsed().as_secs_f64();
                let backend = backend.map_err(|_| "model_load_failed")?;
                let started = Instant::now();
                let history = backend.history();
                timings.reset_seconds = started.elapsed().as_secs_f64();
                let history = history.map_err(|_| "history_initialization_failed")?;
                self.backend = Some(backend);
                self.history = Some(history);
                self.epoch = 1;
                Ok(json!({}))
            }
            Command::Reset {} => {
                let backend = self.backend.as_ref().ok_or("not_initialized")?;
                let next = self.epoch.checked_add(1).ok_or("epoch_overflow")?;
                let started = Instant::now();
                let history = backend.history();
                self.history = Some(history.map_err(|_| "history_initialization_failed")?);
                timings.reset_seconds = started.elapsed().as_secs_f64();
                self.epoch = next;
                Ok(json!({}))
            }
            Command::Ingest { conversation } => {
                let models = self
                    .backend
                    .as_ref()
                    .is_some_and(|backend| backend.profile["mode"] == "models");
                let history = self.history.as_mut().ok_or("not_initialized")?;
                history.counts.ingest_attempts = history
                    .counts
                    .ingest_attempts
                    .checked_add(1)
                    .ok_or("count_overflow")?;
                let started = Instant::now();
                let result = history.memory.remember(conversation);
                timings.ingest_seconds = started.elapsed().as_secs_f64();
                let remembered = result.map_err(memory_error_code)?;
                if remembered.created {
                    history.counts.session_count = history
                        .counts
                        .session_count
                        .checked_add(1)
                        .ok_or("count_overflow")?;
                    history.counts.source_count = history
                        .counts
                        .source_count
                        .checked_add(remembered.source_count)
                        .ok_or("count_overflow")?;
                    history.counts.chunk_count = history
                        .counts
                        .chunk_count
                        .checked_add(remembered.chunk_count)
                        .ok_or("count_overflow")?;
                }
                let embedding_inputs = if models && remembered.created {
                    remembered.source_count + remembered.chunk_count + 1
                } else {
                    0
                };
                Ok(json!({"remembered": remembered,
                          "modelUsageBounds": {"maxIngestEmbeddingInputs": embedding_inputs}}))
            }
            Command::Recall {
                query,
                options,
                compact_context,
                context_expansion,
            } => {
                let models = self
                    .backend
                    .as_ref()
                    .is_some_and(|backend| backend.profile["mode"] == "models");
                let history = self.history.as_mut().ok_or("not_initialized")?;
                history.counts.recall_attempts = history
                    .counts
                    .recall_attempts
                    .checked_add(1)
                    .ok_or("count_overflow")?;
                let searches = if models {
                    options.roles.len() * 4 * if context_expansion.is_some() { 2 } else { 1 }
                } else {
                    0
                };
                let started = Instant::now();
                let result = if let Some(limits) = &context_expansion {
                    history
                        .memory
                        .recall_with_aggregation_context(&query, &options, limits)
                        .map(|(evidence, context)| (evidence, Some(context)))
                } else {
                    history
                        .memory
                        .recall(&query, &options)
                        .and_then(|evidence| {
                            let context = compact_context
                                .then(|| evidence.pack_context(options.max_bytes))
                                .transpose()?;
                            Ok((evidence, context))
                        })
                };
                timings.query_seconds = started.elapsed().as_secs_f64();
                let (evidence, context) = result.map_err(memory_error_code)?;
                let mut response = json!({"evidence": evidence, "recallOptions": options,
                          "modelUsageBounds": {"maxNativeSearches": searches, "maxQueryPlanCalls": searches,
                                               "maxQueryEmbeddingInputs": searches * 6,
                                               "maxGeneratedPlanTokens": searches * 1536,
                                               "maxRerankPairs": searches * 20}});
                if let Some(context) = context {
                    response["context"] =
                        serde_json::to_value(context).map_err(|_| "internal_error")?;
                }
                Ok(response)
            }
        }
    }

    fn reply(&mut self, command: Command) -> Value {
        let name = command.name();
        let mut timings = Timings::default();
        let result = self.execute(command, &mut timings);
        self.cumulative.add(&timings);
        let mut response = match result {
            Ok(value) => value,
            Err(code) => json!({"error": {"code": code}}),
        };
        let status = if response.get("error").is_some() {
            "error"
        } else {
            "ok"
        };
        let fields = response
            .as_object_mut()
            .expect("command result is an object");
        fields.insert("status".into(), json!(status));
        fields.insert("command".into(), json!(name));
        fields.insert("epoch".into(), json!(self.epoch));
        fields.insert("processId".into(), json!(std::process::id()));
        fields.insert("cumulativeTimings".into(), json!(self.cumulative));
        for (key, value) in serde_json::to_value(timings)
            .expect("finite timings")
            .as_object()
            .unwrap()
        {
            fields.insert(key.clone(), value.clone());
        }
        let empty = Counts::default();
        let counts = self
            .history
            .as_ref()
            .map(|history| &history.counts)
            .unwrap_or(&empty);
        for (key, value) in serde_json::to_value(counts)
            .expect("counts serialize")
            .as_object()
            .unwrap()
        {
            fields.insert(key.clone(), value.clone());
        }
        if let Some(backend) = &self.backend {
            fields.insert("profile".into(), backend.profile.clone());
        }
        fields.insert("transportPolicy".into(), json!({
            "schemaVersion": 1, "maxLineBytes": MAX_LINE_BYTES, "maxReplyBytes": MAX_REPLY_BYTES,
            "requestTimeoutSeconds": null, "timeoutPolicy": "caller-enforced process termination; no automatic retry",
            "epochScope": "process; each successful reset creates an empty isolated history",
            "defaultRecallOptions": profile_recall_options(),
        }));
        fields.insert("modelUsage".into(), json!({
            "exactCountersAvailable": false, "reason": "production model stages do not expose token and pair counters",
            "remoteApiCalls": 0, "localComputeUsd": null, "wallTimeIncludesAllModelStages": true,
            "maxEmbeddingTokensPerInput": 2048, "maxQueryVariantsPerSearch": 6,
            "maxGeneratedPlanTokensPerSearch": 1536, "maxRerankPairsPerSearch": 20,
            "maxIngestEmbeddingInputsPerAttempt": 4096,
        }));
        response
    }
}

fn memory_error_code(error: anyhow::Error) -> &'static str {
    match error.downcast_ref::<MemoryError>() {
        Some(MemoryError::Invalid(_)) => "invalid_memory_request",
        Some(MemoryError::Conflict(_)) => "immutable_conflict",
        Some(MemoryError::NotFound(_)) => "memory_source_missing",
        Some(MemoryError::Concurrent(_)) => "memory_concurrent_mutation",
        None => "memory_operation_failed",
    }
}

enum Frame {
    Line(Vec<u8>),
    TooLong,
}

fn read_frame(input: &mut impl BufRead) -> io::Result<Option<Frame>> {
    let mut line = Vec::new();
    let mut oversized = false;
    let mut received = false;
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Ok(if !received {
                None
            } else if oversized {
                Some(Frame::TooLong)
            } else {
                Some(Frame::Line(line))
            });
        }
        received = true;
        let newline = available.iter().position(|byte| *byte == b'\n');
        let content = newline.unwrap_or(available.len());
        if !oversized {
            if content > MAX_LINE_BYTES - line.len() {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(&available[..content]);
            }
        }
        input.consume(content + usize::from(newline.is_some()));
        if newline.is_some() {
            return Ok(Some(if oversized {
                Frame::TooLong
            } else {
                Frame::Line(line)
            }));
        }
    }
}

fn protocol_error(epoch: u64, code: &str) -> Value {
    json!({"status": "error", "command": null, "epoch": epoch, "error": {"code": code}})
}

fn write_reply(output: &mut impl Write, response: &Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(response).map_err(io::Error::other)?;
    if bytes.len() > MAX_REPLY_BYTES {
        bytes = serde_json::to_vec(&protocol_error(
            response["epoch"].as_u64().unwrap_or(0),
            "reply_too_large",
        ))
        .map_err(io::Error::other)?;
    }
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    output.flush()
}

fn run(args: Args, input: &mut impl BufRead, output: &mut impl Write) -> io::Result<()> {
    let mut runtime = Runtime::new(args);
    while let Some(frame) = read_frame(input)? {
        let response = match frame {
            Frame::TooLong => protocol_error(runtime.epoch, "line_too_large"),
            Frame::Line(bytes) => match serde_json::from_slice::<Command>(&bytes) {
                Ok(command) => runtime.reply(command),
                Err(_) => protocol_error(runtime.epoch, "invalid_command"),
            },
        };
        write_reply(output, &response)?;
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            eprintln!("{error}");
            return std::process::ExitCode::SUCCESS;
        }
        Err(_) => {
            let _ = write_reply(
                &mut io::stdout().lock(),
                &protocol_error(0, "invalid_arguments"),
            );
            return std::process::ExitCode::from(2);
        }
    };
    if run(args, &mut io::stdin().lock(), &mut io::stdout().lock()).is_err() {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn oversized_lines_are_drained_without_losing_the_next_command() {
        for size in [MAX_LINE_BYTES - 1, MAX_LINE_BYTES, MAX_LINE_BYTES + 1] {
            let mut bytes = vec![b'x'; size];
            bytes.extend_from_slice(b"\n{\"command\":\"reset\"}\n");
            let mut input = Cursor::new(bytes);
            match read_frame(&mut input).unwrap().unwrap() {
                Frame::Line(line) => {
                    assert!(size <= MAX_LINE_BYTES);
                    assert_eq!(line.len(), size);
                }
                Frame::TooLong => assert!(size > MAX_LINE_BYTES),
            }
            let Frame::Line(next) = read_frame(&mut input).unwrap().unwrap() else {
                panic!("next frame was lost")
            };
            assert!(matches!(
                serde_json::from_slice::<Command>(&next).unwrap(),
                Command::Reset {}
            ));
        }
    }

    #[test]
    fn malformed_and_unknown_fields_return_clean_errors_before_valid_commands() {
        let input = b"not-json\n{\"command\":\"reset\",\"privatePath\":\"DO_NOT_ECHO\"}\n{\"command\":\"initialize\"}\n";
        let mut output = Vec::new();
        run(
            Args {
                lexical: true,
                models: None,
                subquery_weight: 0.0,
            },
            &mut Cursor::new(input),
            &mut output,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(!text.contains("DO_NOT_ECHO"));
        let replies: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(replies.len(), 3);
        assert_eq!(replies[0]["error"]["code"], "invalid_command");
        assert_eq!(replies[1]["error"]["code"], "invalid_command");
        assert_eq!(replies[2]["status"], "ok");
        assert_eq!(replies[2]["epoch"], 1);
    }

    #[test]
    fn reset_separates_histories_and_zeroes_counts_without_changing_profile() {
        let mut runtime = Runtime::new(Args {
            lexical: true,
            models: None,
            subquery_weight: 0.0,
        });
        let initialized = runtime.reply(Command::Initialize { model_root: None });
        let input: ConversationInput = serde_json::from_value(json!({"sessionId": "synthetic", "version": "1", "turns": [
            {"sourceId": "original", "role": "assistant", "text": "Asterneedle has a recorded setting.", "observationTime": 1000}
        ]})).unwrap();
        let ingested = runtime.reply(Command::Ingest {
            conversation: input,
        });
        assert_eq!(ingested["sourceCount"], 1);
        let before = runtime.reply(Command::Recall {
            query: "Asterneedle".into(),
            options: profile_recall_options(),
            compact_context: true,
            context_expansion: None,
        });
        assert_eq!(before["evidence"]["sources"].as_array().unwrap().len(), 1);
        let context: holographic_memory::core::memory_api::ConversationContext =
            serde_json::from_value(before["context"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(context.sources().unwrap()).unwrap(),
            before["evidence"]["sources"]
        );
        let reset = runtime.reply(Command::Reset {});
        assert_eq!(reset["epoch"], 2);
        assert_eq!(reset["sourceCount"], 0);
        assert_eq!(reset["chunkCount"], 0);
        assert_eq!(reset["profile"], initialized["profile"]);
        let after = runtime.reply(Command::Recall {
            query: "Asterneedle".into(),
            options: profile_recall_options(),
            compact_context: false,
            context_expansion: None,
        });
        assert_eq!(after["epoch"], 2);
        assert!(after["evidence"]["sources"].as_array().unwrap().is_empty());
        assert!(after.get("context").is_none());
    }

    #[test]
    fn invalid_subquery_weights_fail_before_backend_initialization() {
        for weight in [-f64::EPSILON, f64::NAN, f64::INFINITY] {
            let mut runtime = Runtime::new(Args {
                lexical: true,
                models: None,
                subquery_weight: weight,
            });
            let reply = runtime.reply(Command::Initialize { model_root: None });
            assert_eq!(reply["error"]["code"], "invalid_subquery_weight");
            assert_eq!(reply["epoch"], 0);
            assert!(runtime.backend.is_none());
        }
    }
}
