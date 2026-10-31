# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""`warm-import`: migrate memories from another system into a WARM server."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Mapping, Optional

from .common import register_secret, scrub
from .model import ImporterError, MissingDependency
from .options import ID_MAPPINGS, ID_PREFIX_MODES, NAMESPACE_MODES, OVERSIZE_MODES, SPLIT_POLICIES, VECTOR_MODES, ImportOptions
from .pipeline import exit_code, run_import
from .sink import WarmSink
from .sources import source_class, source_names

EPILOG = """\
Credentials come from the environment, never from logs or reports: WARM_API_KEY and
WARM_ZERO_TRUST_KEY for the destination, and each source's own variable (see
`warm-import <source> --help`). Re-running is idempotent: documents replace by id.

Exit codes: 0 success, 1 failure (the checkpoint allows a resume), 2 usage,
3 finished but some records were rejected (see <checkpoint>.rejects.jsonl).
"""


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("destination")
    g.add_argument("--warm-url", default=None, help="WARM server URL (default: $WARM_URL or http://127.0.0.1:8080)")
    g.add_argument("--api-key", default=None, help="discouraged: visible in the process list; prefer $WARM_API_KEY")
    g.add_argument("--tenant", default=None, help="WARM tenant id (X-Tenant-ID); default tenant when omitted")
    g.add_argument("--zero-trust-key", default=None, help="mask vectors client-side (SDK VectorMask); prefer $WARM_ZERO_TRUST_KEY")
    g.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    v = p.add_argument_group("vectors")
    v.add_argument("--vectors", choices=VECTOR_MODES, default="auto",
                   help="auto (default): carry source vectors of the server's dimension, encode text otherwise, "
                        "fail on a dimension mismatch; carry; reencode (ignore source vectors); "
                        "project (seeded random projection to the server dimension); none (text only)")
    v.add_argument("--embedder", default=None, metavar="MODULE:FUNC",
                   help="client-side encoder for records that need one: callable(list[str]) -> list[list[float]]")
    v.add_argument("--projection-seed", type=int, default=None, help="seed for --vectors project")
    v.add_argument("--target-dim", type=int, default=None, help="server input dimension when no server is probed (dry run)")
    m = p.add_argument_group("mapping")
    m.add_argument("--namespace-mode", choices=NAMESPACE_MODES, default="metadata",
                   help="metadata (default): hms_namespace key and id prefix; tenant: one WARM tenant per namespace")
    m.add_argument("--default-namespace", default="", help="namespace for records that have none")
    m.add_argument("--id-prefix-namespace", choices=ID_PREFIX_MODES, default="auto")
    m.add_argument("--id-mapping", choices=ID_MAPPINGS, default="v1",
                   help="v1 preserves existing ids; v2 hashes typed source/namespace/id/part identities")
    m.add_argument("--adopt-legacy", action="store_true",
                   help="explicitly adopt unstamped imports with matching original source identity and part metadata")
    m.add_argument("--split-policy", choices=SPLIT_POLICIES, default="reject",
                   help="reject changed layouts (default), or replace and conditionally remove obsolete owned parts")
    m.add_argument("--tenant-prefix", default="", help="prefix for namespace-derived tenant ids")
    m.add_argument("--oversize-metadata", choices=OVERSIZE_MODES, default="reject",
                   help="reject (default) or truncate (drop the largest keys, listed in warm_dropped_keys)")
    m.add_argument("--max-text-words", type=int, default=4000)
    m.add_argument("--keep-empty", action="store_true", help="reject (instead of skipping) records with no text and no vector")
    r = p.add_argument_group("run")
    r.add_argument("--batch-size", type=int, default=256)
    r.add_argument("--checkpoint", default=None,
                   help="checkpoint file (default: derived from source, target and mapping under "
                        "$WARM_IMPORT_STATE_DIR or ~/.cache/warm-import); an interrupted run resumes from it")
    r.add_argument("--no-checkpoint", action="store_true", help="do not write a checkpoint or a rejects file")
    r.add_argument("--max-rejects", type=int, default=1000, help="abort after the server refuses this many documents")
    r.add_argument("--restart", action="store_true", help="discard an incomplete checkpoint")
    r.add_argument("--retries", type=int, default=5)
    r.add_argument("--limit", type=int, default=None, help="stop after this many source records")
    r.add_argument("--dry-run", action="store_true", help="read and map everything, write nothing, report fidelity")
    r.add_argument("--verify-sample", type=int, default=50, help="queries for the old-vs-new recall check; 0 disables")
    r.add_argument("--verify-k", type=int, default=10)
    r.add_argument("--no-maintain-index", action="store_true", help="skip index maintenance after the import")
    r.add_argument("--report", default=None, help="write the fidelity report (JSON) here")
    r.add_argument("--allow-rejects", action="store_true", help="exit 0 even when records were rejected")
    r.add_argument("--quiet", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warm-import",
        description="Migrate memories from another vector store or memory system into WARM.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="source", metavar="SOURCE")
    common = _common_parser()
    sub.add_parser("list", help="list the supported sources")
    for name in source_names():
        try:
            cls = source_class(name)
        except ImportError as exc:  # a broken optional install must not take the other sources down
            sub.add_parser(name, help=f"unavailable: {exc}")
            continue
        sp = sub.add_parser(name, parents=[common], help=cls.summary, description=cls.summary,
                            epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
        cls.add_arguments(sp)
    return parser


def options_from_args(args: argparse.Namespace) -> ImportOptions:
    opts = ImportOptions(
        vectors=args.vectors,
        embedder=args.embedder,
        target_dim=args.target_dim,
        namespace_mode=args.namespace_mode,
        default_namespace=args.default_namespace,
        id_prefix_namespace=args.id_prefix_namespace,
        id_mapping=args.id_mapping,
        adopt_legacy=args.adopt_legacy,
        split_policy=args.split_policy,
        tenant_prefix=args.tenant_prefix,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        no_checkpoint=args.no_checkpoint,
        max_rejects=args.max_rejects,
        restart=args.restart,
        retries=args.retries,
        limit=args.limit,
        max_text_words=args.max_text_words,
        oversize_metadata=args.oversize_metadata,
        skip_empty=not args.keep_empty,
        dry_run=args.dry_run,
        verify_sample=args.verify_sample,
        verify_k=args.verify_k,
        report_path=args.report,
        allow_rejects=args.allow_rejects,
    )
    if args.projection_seed is not None:
        opts.projection_seed = args.projection_seed
    opts.extra["maintain_index"] = not args.no_maintain_index
    return opts


def main(argv: Optional[List[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source in (None, "list"):
        if args.source is None:
            parser.print_help()
            return 2
        for name in source_names():
            print(f"{name:20s} {source_class(name).summary}")
        return 0
    api_key = args.api_key or env.get("WARM_API_KEY")
    zero_trust = args.zero_trust_key or env.get("WARM_ZERO_TRUST_KEY") or ""
    register_secret(api_key)
    register_secret(zero_trust)
    say = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
    sink: Optional[WarmSink] = None
    try:
        opts = options_from_args(args)
        opts.validate()
        source = source_class(args.source).from_args(args, env)
        url = args.warm_url or env.get("WARM_URL") or "http://127.0.0.1:8080"
        if not (args.dry_run and args.target_dim and not (args.warm_url or env.get("WARM_URL"))):
            sink = WarmSink(url, api_key=api_key, tenant=args.tenant, zero_trust_key=zero_trust,
                           timeout=args.timeout, retries=opts.retries)
        with source:
            report = run_import(source, sink, opts, tenant=args.tenant, progress=say)
    except MissingDependency as exc:
        print(f"warm-import: {exc}", file=sys.stderr)
        return 1
    except ImporterError as exc:
        print(scrub(f"warm-import: {exc}"), file=sys.stderr)
        partial = getattr(exc, "report", None)
        if partial is not None and not args.quiet:
            print(partial.summary(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("warm-import: interrupted; rerun the same command to resume from the checkpoint", file=sys.stderr)
        return 1
    except Exception as exc:  # a client library error may embed a DSN or key: print it scrubbed, never a raw traceback
        print(scrub(f"warm-import: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1
    finally:
        if sink is not None:
            sink.close()
    if not args.quiet:
        print(report.summary())
    return exit_code(report, opts)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
