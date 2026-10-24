#!/bin/bash
set -euo pipefail
cd ~/hms-sbg-positions
uv run --no-project python paired.py > paired.log 2>&1
uv run --no-project python paired_ingest.py > ingest.log 2>&1
date -u +%FT%TZ > timings-done
uv run --no-project python correctness.py > exhaustive-summary.json 2> exhaustive.log
date -u +%FT%TZ > exhaustive-done
