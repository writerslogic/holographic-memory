#!/bin/bash
# Single debate runner. Args: debate_id debate_name system_file r1_file r2_file r3_file
set -euo pipefail

OPENROUTER_KEY="${OPENROUTER_API_KEY}"
BASE_DIR="/Volumes/A/HMS/conductor/debate_phase1"

DEBATE_ID="$1"
DEBATE_NAME="$2"
SYSTEM_FILE="$3"
R1_FILE="$4"
R2_FILE="$5"
R3_FILE="$6"

OUTDIR="$BASE_DIR/$DEBATE_ID"
mkdir -p "$OUTDIR"

MODELS=(
  "anthropic/claude-opus-4-8"
  "deepseek/deepseek-v4-pro"
  "qwen/qwen3.7-max"
  "nvidia/nemotron-3-ultra-550b-a55b:free"
  "nex-agi/nex-n2-pro:free"
)
MODEL_NAMES=(
  "Claude-Opus"
  "DeepSeek-V4"
  "Qwen-Max"
  "Nemotron-Ultra"
  "Nex-N2-Pro"
)

SYSTEM_PROMPT=$(cat "$SYSTEM_FILE")
R1_PROMPT=$(cat "$R1_FILE")
R2_PROMPT=$(cat "$R2_FILE")
R3_PROMPT=$(cat "$R3_FILE")

call_model() {
  local model="$1"
  local messages_file="$2"
  local response
  response=$(curl -s --max-time 180 -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer $OPENROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/writerslogic/hms" \
    -H "X-Title: HMS Debate $DEBATE_ID" \
    -d "$(jq -n --arg model "$model" --slurpfile msgs "$messages_file" \
      '{model: $model, messages: $msgs[0], temperature: 0.8, max_tokens: 4096}')" 2>/dev/null)
  echo "$response" | jq -r '.choices[0].message.content // .error.message // "ERROR: No response"' 2>/dev/null || echo "ERROR: parse failed"
}

echo "=== DEBATE $DEBATE_ID: $DEBATE_NAME ===" | tee "$OUTDIR/transcript.txt"

# Init conversation JSON
CONV_FILE=$(mktemp)
jq -n --arg sys "$SYSTEM_PROMPT" '[{"role": "system", "content": $sys}]' > "$CONV_FILE"

ROUND_PROMPTS=("$R1_PROMPT" "$R2_PROMPT" "$R3_PROMPT")
ROUND_LABELS=("Opening Positions" "Challenge & Demolish" "Concrete Implementation")

for round in 0 1 2; do
  rn=$((round + 1))
  echo "" | tee -a "$OUTDIR/transcript.txt"
  echo "--- Round $rn: ${ROUND_LABELS[$round]} ---" | tee -a "$OUTDIR/transcript.txt"

  for i in "${!MODELS[@]}"; do
    echo "" | tee -a "$OUTDIR/transcript.txt"
    echo ">>> ${MODEL_NAMES[$i]} (R$rn) <<<" | tee -a "$OUTDIR/transcript.txt"

    # Build messages: conversation + user prompt
    MSG_FILE=$(mktemp)
    jq --arg content "You are ${MODEL_NAMES[$i]}. Round $rn of 3. ${ROUND_PROMPTS[$round]}" \
      '. + [{"role": "user", "content": $content}]' "$CONV_FILE" > "$MSG_FILE"

    RESPONSE=$(call_model "${MODELS[$i]}" "$MSG_FILE")
    echo "$RESPONSE" | tee -a "$OUTDIR/transcript.txt"
    echo "---" | tee -a "$OUTDIR/transcript.txt"

    # Append to conversation
    jq --arg name "${MODEL_NAMES[$i]}" --arg rn "R$rn" --arg resp "$RESPONSE" \
      '. + [{"role": "assistant", "content": ("[" + $name + " " + $rn + "]: " + $resp)}]' \
      "$CONV_FILE" > "${CONV_FILE}.tmp" && mv "${CONV_FILE}.tmp" "$CONV_FILE"

    rm -f "$MSG_FILE"
  done
done

cp "$CONV_FILE" "$OUTDIR/conversation.json"
rm -f "$CONV_FILE"
echo "" | tee -a "$OUTDIR/transcript.txt"
echo "=== DEBATE $DEBATE_ID PHASE 1 COMPLETE ===" | tee -a "$OUTDIR/transcript.txt"
