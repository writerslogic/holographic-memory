# HMS Development Plan

## How to Use This Plan

1. Read `ARCHITECTURE.md` first — understand how all pieces connect
2. Read `MODULES.md` — the public API of every new module
3. Read `TASKS.md` — what to build, in what order
4. When implementing a task, read the relevant section of `ARCHITECTURE.md` for system context
5. Run cargo ONCE after completing each phase group (not per-task)

## Files
- `ARCHITECTURE.md` — system design, data flow, module connections, invariants
- `MODULES.md` — every new file, its public types/methods, who calls it
- `TASKS.md` — ordered task list with dependencies, verification milestones
- `CONTEXT.md` — existing codebase map (what already exists, what to adapt)
- `DEBATES.md` — key decisions from multi-model debates (reference, not required reading)
- `0xx-*.md` — orchestrator-compatible task files (numbered, YAML frontmatter)
