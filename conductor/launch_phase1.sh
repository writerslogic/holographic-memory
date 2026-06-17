#!/bin/bash
# Launch all 5 debates in parallel
set -euo pipefail

P="/Volumes/A/HMS/conductor/prompts"
R="/Volumes/A/HMS/conductor/run_debate.sh"
mkdir -p /Volumes/A/HMS/conductor/debate_phase1

echo "Launching 5 debates in parallel..."

$R A "C_max + Embedding Collapse"          "$P/A/system.txt" "$P/A/r1.txt" "$P/A/r2.txt" "$P/A/r3.txt" &
PID_A=$!
$R B "True Holographic Properties"         "$P/B/system.txt" "$P/B/r1.txt" "$P/B/r2.txt" "$P/B/r3.txt" &
PID_B=$!
$R C "Relation Composition + Arity"        "$P/C/system.txt" "$P/C/r1.txt" "$P/C/r2.txt" "$P/C/r3.txt" &
PID_C=$!
$R D "Self-Supervised Semantic Embeddings" "$P/D/system.txt" "$P/D/r1.txt" "$P/D/r2.txt" "$P/D/r3.txt" &
PID_D=$!
$R E "Adversarial + Production Hardening"  "$P/E/system.txt" "$P/E/r1.txt" "$P/E/r2.txt" "$P/E/r3.txt" &
PID_E=$!

echo "PIDs: A=$PID_A B=$PID_B C=$PID_C D=$PID_D E=$PID_E"
echo "Waiting for all to complete..."

FAILED=0
for pid in $PID_A $PID_B $PID_C $PID_D $PID_E; do
  wait $pid || FAILED=$((FAILED + 1))
done

echo ""
echo "=== ALL PHASE 1 DEBATES COMPLETE ($FAILED failures) ==="
for d in A B C D E; do
  f="/Volumes/A/HMS/conductor/debate_phase1/$d/transcript.txt"
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    turns=$(grep -c '>>>' "$f" || true)
    echo "  Debate $d: $lines lines, $turns model turns"
  else
    echo "  Debate $d: NO OUTPUT"
  fi
done
