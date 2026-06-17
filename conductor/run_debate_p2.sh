#!/bin/bash
# Phase 2 debate runner. Same as run_debate.sh but with updated model list and rounds 4-6.
set -euo pipefail

OPENROUTER_KEY="${OPENROUTER_API_KEY}"
BASE_DIR="/Volumes/A/HMS/conductor/debate_phase2"

DEBATE_ID="$1"
DEBATE_NAME="$2"
SYSTEM_FILE="$3"
R4_FILE="$4"
R5_FILE="$5"
R6_FILE="$6"

OUTDIR="$BASE_DIR/$DEBATE_ID"
mkdir -p "$OUTDIR"

# Replaced Nemotron-Ultra and Nex-N2-Pro (both failed in Phase 1)
MODELS=(
  "anthropic/claude-opus-4-8"
  "deepseek/deepseek-v4-pro"
  "qwen/qwen3.7-max"
  "google/gemini-2.5-pro"
  "meta-llama/llama-4-maverick"
)
MODEL_NAMES=(
  "Claude-Opus"
  "DeepSeek-V4"
  "Qwen-Max"
  "Gemini-Pro"
  "Llama-Maverick"
)

SYSTEM_PROMPT=$(cat "$SYSTEM_FILE")
R4_PROMPT=$(cat "$R4_FILE")
R5_PROMPT=$(cat "$R5_FILE")
R6_PROMPT=$(cat "$R6_FILE")

call_model() {
  local model="$1"
  local messages_file="$2"
  curl -s --max-time 180 -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer $OPENROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/writerslogic/hms" \
    -H "X-Title: HMS Debate P2 $DEBATE_ID" \
    -d "$(jq -n --arg model "$model" --slurpfile msgs "$messages_file" \
      '{model: $model, messages: $msgs[0], temperature: 0.8, max_tokens: 4096}')" 2>/dev/null \
    | jq -r '.choices[0].message.content // .error.message // "ERROR: No response"' 2>/dev/null \
    || echo "ERROR: parse failed"
}

echo "=== DEBATE $DEBATE_ID P2: $DEBATE_NAME ===" | tee "$OUTDIR/transcript.txt"

# Load Phase 1 conversation if it exists, otherwise start fresh
P1_CONV="/Volumes/A/HMS/conductor/debate_phase1/$DEBATE_ID/conversation.json"
CONV_FILE=$(mktemp)
if [ -f "$P1_CONV" ]; then
  # Take the system message + last 6 assistant messages from Phase 1 (to stay under context limits)
  jq '[.[0]] + [.[] | select(.role == "assistant")] | .[-6:]' "$P1_CONV" > "$CONV_FILE" 2>/dev/null || \
    jq -n --arg sys "$SYSTEM_PROMPT" '[{"role": "system", "content": $sys}]' > "$CONV_FILE"
else
  jq -n --arg sys "$SYSTEM_PROMPT" '[{"role": "system", "content": $sys}]' > "$CONV_FILE"
fi

# Prepend new system prompt
CONV_WITH_SYS=$(mktemp)
jq --arg sys "$SYSTEM_PROMPT" '.[0] = {"role": "system", "content": $sys}' "$CONV_FILE" > "$CONV_WITH_SYS"
mv "$CONV_WITH_SYS" "$CONV_FILE"

ROUND_PROMPTS=("$R4_PROMPT" "$R5_PROMPT" "$R6_PROMPT")
ROUND_LABELS=("Refined Questions" "Ruthless Prioritization" "Final Contribution")

for round in 0 1 2; do
  rn=$((round + 4))
  echo "" | tee -a "$OUTDIR/transcript.txt"
  echo "--- Round $rn: ${ROUND_LABELS[$round]} ---" | tee -a "$OUTDIR/transcript.txt"

  for i in "${!MODELS[@]}"; do
    echo "" | tee -a "$OUTDIR/transcript.txt"
    echo ">>> ${MODEL_NAMES[$i]} (R$rn) <<<" | tee -a "$OUTDIR/transcript.txt"

    MSG_FILE=$(mktemp)
    jq --arg content "You are ${MODEL_NAMES[$i]}. Round $rn of 6 (Phase 2). ${ROUND_PROMPTS[$round]}" \
      '. + [{"role": "user", "content": $content}]' "$CONV_FILE" > "$MSG_FILE"

    RESPONSE=$(call_model "${MODELS[$i]}" "$MSG_FILE")
    echo "$RESPONSE" | tee -a "$OUTDIR/transcript.txt"
    echo "---" | tee -a "$OUTDIR/transcript.txt"

    jq --arg name "${MODEL_NAMES[$i]}" --arg rn "R$rn" --arg resp "$RESPONSE" \
      '. + [{"role": "assistant", "content": ("[" + $name + " " + $rn + "]: " + $resp)}]' \
      "$CONV_FILE" > "${CONV_FILE}.tmp" && mv "${CONV_FILE}.tmp" "$CONV_FILE"

    rm -f "$MSG_FILE"
  done
done

cp "$CONV_FILE" "$OUTDIR/conversation.json"
rm -f "$CONV_FILE"
echo "" | tee -a "$OUTDIR/transcript.txt"
echo "=== DEBATE $DEBATE_ID PHASE 2 COMPLETE ===" | tee -a "$OUTDIR/transcript.txt"
