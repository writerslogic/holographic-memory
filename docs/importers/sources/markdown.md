# markdown (agent memory folders)

Reads CLAUDE.md-style files, a Claude memory-tool directory (`/memories`), and Claude Code auto-memory folders (`MEMORY.md` plus linked files with YAML frontmatter). Verified on real temporary directories.

- One record per file; with `--split auto` (default) a file over `--max-section-words` (800) is split per ATX heading section (`--split headings` always, `none` never). Fenced code never starts a section.
- id = path relative to `--root`, with `#slug` for sections (`#preamble` for text before the first heading, `-2` for repeated headings). Section records carry `heading_path` and `title`.
- YAML frontmatter keys become metadata (flat); a frontmatter-only file keeps its fields as text. `path` and `filename` are always set.
- `created_at` is the file's birth time (modification time where the filesystem has none), `updated_at` the modification time.
- `--include` / `--exclude` globs, `--namespace`. Symlinks are never followed out of `--root`; such files, and files that are not UTF-8, are rejected visibly, not skipped.
- Text only: needs a server with `local-models` or `--embedder`.
