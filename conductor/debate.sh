#!/bin/bash
# HMS Multi-Model Debate Orchestrator
# Queries 6 frontier models via OpenRouter to compose a robust prompt for HMS elevation

set -euo pipefail

OPENROUTER_KEY="${OPENROUTER_API_KEY}"
OUTPUT_DIR="/Volumes/A/HMS/conductor/debate_output"
mkdir -p "$OUTPUT_DIR"

# The 6 models
MODELS=(
  "anthropic/claude-opus-4-8"
  "deepseek/deepseek-v4-pro"
  "xiaomi/mimo-v2.5-pro"
  "stepfun/step-3.7-flash"
  "minimax/minimax-m3"
  "qwen/qwen3.7-max"
)
MODEL_NAMES=(
  "Claude-Opus-4.8"
  "DeepSeek-V4-Pro"
  "MiMo-V2.5-Pro"
  "Step-3.7-Flash"
  "MiniMax-M3"
  "Qwen-3.7-Max"
)

# Read HMS context
HMS_CONTEXT=$(cat <<'CONTEXT_EOF'
# HMS (Holographic Memory System) - Current State & Design Challenge

## What HMS Is
A Rust + N-API library implementing Vector Symbolic Architectures (VSA) with sparse Binary Spatter Code (BSC). D=16384 dimensions, rho=1/256 sparsity (~64 active indices per vector).

## Core Operations
- **Bind (XOR)**: Reversible composition. A⊕B⊕B=A (involutory). Creates relation-vectors from entity-vectors.
- **Bundle (majority vote)**: Superposition. Threshold at ceil(n/2). Combines N vectors preserving similarity to majority.
- **Permute (cyclic shift)**: Position encoding. Breaks commutativity for sequences.
- **Similarity (Jaccard)**: |A∩B|/|A∪B| on sparse index sets. O(k) via sorted merge.

## What Exists
- NSG (Navigable Small World) graph index: O(log N) approximate nearest neighbor
- IVF with product quantization and Nystrom projection
- Persistent mmap arena with LZ4 compression and CRC32 framing
- Ed25519 signing, AES-256-GCM encryption at rest, append-only audit trail
- Epsilon-differential privacy in bundle operations
- Graph engine: explicit typed relations with multi-hop BFS traversal, transitive/symmetric inference, temporal valid_from/valid_to
- Federated queries across multiple instances
- Diffusion-based vector factorization (decomposes bound products)
- Concept synthesis via similarity-based clustering + bundling
- Knowledge graph: triplet encoding (h⊕r⊕t), sequence memory, analogical reasoning
- 133 tests pass, clippy clean, published to npm as @writerslogic/hms-native

## The Problem: "Holographic" Is Branding, Not Behavior
An honest audit revealed HMS FAILS to deliver on the defining properties of holographic memory:
1. **Pattern completion**: Present partial vector → reconstruct full vector. MISSING.
2. **Graceful degradation**: Remove random dimensions → proportional quality loss. UNPROVEN.
3. **Associative recall**: Store A⊕B, present A, get B back exactly. UNTESTED at scale.
4. **Superposition storage**: N items in 1 vector, retrieved by association key. NOT IMPLEMENTED.
5. **Content-addressable retrieval**: Query is just K-NN similarity search, not holographic reconstruction.
6. **Interference-based storage**: Arena is just append-only log, not physics-inspired.

## What's Also Missing (Competitive Gaps)
- No attractor dynamics (Hopfield-like convergence to stored patterns)
- No holographic plates (Kanerva/Gayler trace model)
- Relation composition not discoverable (father⊕father≈grandfather)
- No graph-aware weighted scoring in traversal
- Federated queries reopen peer instances per call (no caching)
- No temporal vector versioning (only relation timestamps)
- Concept synthesis is greedy O(n²), not optimal
- No GNN-equivalent message passing via bundle(neighbors)
- No continuous graph space (gradient-based traversal)
- No self-organizing schema discovery

## The Goal
Design a comprehensive implementation prompt that will make HMS:
1. TRULY holographic (pattern completion, attractor dynamics, superposition plates, graceful degradation)
2. Superior to Neo4j in graph capabilities (holographic traversal, relation composition, soft edges)
3. Superior to Pinecone/Weaviate/Milvus in vector capabilities (holographic recall, not just similarity)
4. The first system where "holographic" is mathematically proven, not marketing
5. Privacy-preserving by physics (not just encryption)

## Constraints
- Rust, no new external crates (build on existing primitives)
- Must not break existing 133 tests or API
- Must be additive, not destructive to current architecture
- Performance: pattern completion < 10ms, attractor recall < 50ms
- Everything must have tests that PROVE the property, not just check it compiles
CONTEXT_EOF
)

call_model() {
  local model="$1"
  local messages="$2"
  local temp="${3:-0.8}"

  local response
  response=$(curl -s -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer $OPENROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/writerslogic/hms" \
    -H "X-Title: HMS Debate" \
    -d "$(jq -n \
      --arg model "$model" \
      --argjson messages "$messages" \
      --argjson temp "$temp" \
      '{model: $model, messages: $messages, temperature: $temp, max_tokens: 4096}')" 2>/dev/null)

  echo "$response" | jq -r '.choices[0].message.content // .error.message // "ERROR: No response"' 2>/dev/null || echo "ERROR: Failed to parse response"
}

# Build initial system prompt
SYSTEM_PROMPT="You are participating in a multi-model debate about how to make HMS (Holographic Memory System) the most advanced database/memory system ever built. You must be BRUTALLY HONEST. Do not agree with other models just to be agreeable. If you think an idea is bad, say so and explain why. If you think something is missing, say what. If another model is being lazy or vague, call them out. Your goal: produce concrete, implementable, mathematically grounded ideas that would make every other database system obsolete. Be specific — name algorithms, cite complexity bounds, describe data structures."

# Initialize conversation history as JSON array
CONVERSATION="[]"
CONVERSATION=$(echo "$CONVERSATION" | jq --arg sys "$SYSTEM_PROMPT" '. + [{"role": "system", "content": $sys}]')

echo "=== HMS Multi-Model Debate ==="
echo "Models: ${MODEL_NAMES[*]}"
echo ""

# Round 1: Each model gives their opening position
echo "--- ROUND 1: Opening Positions ---"
ROUND1_PROMPT="Here is the full context about HMS:\n\n$HMS_CONTEXT\n\nYou are the FIRST to speak. Give your opening position:\n1. What is the single most important thing HMS must implement to be truly holographic?\n2. What specific algorithm or data structure would you propose?\n3. What would make HMS genuinely superior to Neo4j AND Pinecone simultaneously?\n4. What is everyone else likely to get wrong about this problem?\n\nBe concrete. Name algorithms, cite papers, describe data structures with field types. No hand-waving."

for i in "${!MODELS[@]}"; do
  echo ""
  echo ">>> ${MODEL_NAMES[$i]} (Round 1) <<<"

  MESSAGES=$(echo "$CONVERSATION" | jq --arg content "You are ${MODEL_NAMES[$i]}. $ROUND1_PROMPT" '. + [{"role": "user", "content": $content}]')

  RESPONSE=$(call_model "${MODELS[$i]}" "$MESSAGES")
  echo "$RESPONSE" | tee "$OUTPUT_DIR/r1_${MODEL_NAMES[$i]}.txt"

  # Add to conversation history
  CONVERSATION=$(echo "$CONVERSATION" | jq \
    --arg name "${MODEL_NAMES[$i]}" \
    --arg resp "$RESPONSE" \
    '. + [{"role": "assistant", "content": ("[" + $name + " Round 1]: " + $resp)}]')

  echo ""
  echo "---"
done

# Rounds 2-5: Structured debate with challenges
for round in 2 3 4 5; do
  echo ""
  echo "--- ROUND $round ---"

  case $round in
    2) ROUND_PROMPT="You've now heard all opening positions. CHALLENGE at least one other model's proposal. Point out flaws, missing pieces, or better alternatives. Then REFINE your own position based on what you've learned. What did someone else propose that you think is wrong or incomplete? What would you steal from another model's idea and improve?" ;;
    3) ROUND_PROMPT="We're halfway through. Time to get SPECIFIC about implementation. Describe the EXACT Rust data structures (struct fields, method signatures) for your most important proposal. How does it integrate with EntangledHVec's existing sorted-u32-index representation? What is the exact complexity? What test would PROVE it works (not just that it compiles)? If you're being vague, the other models will call you out." ;;
    4) ROUND_PROMPT="SYNTHESIS round. Looking at ALL proposals so far, what is the MINIMAL set of additions that would make HMS genuinely unprecedented? Not everything proposed is worth building. What should be CUT? What is essential? Prioritize ruthlessly. If you're just agreeing with the consensus, you're being lazy — find the blind spot everyone is missing." ;;
    5) ROUND_PROMPT="FINAL round. Write your contribution to the final prompt that will be given to Claude to implement HMS. This should be a SPECIFIC, ACTIONABLE section — not a summary of the debate. What EXACT capability should be built, what EXACT algorithm should be used, what EXACT test proves it works? This is your last chance to influence the final design. Make it count. Do NOT repeat what others have said unless you're adding something new." ;;
  esac

  for i in "${!MODELS[@]}"; do
    echo ""
    echo ">>> ${MODEL_NAMES[$i]} (Round $round) <<<"

    MESSAGES=$(echo "$CONVERSATION" | jq \
      --arg content "You are ${MODEL_NAMES[$i]}. Round $round of 5. $ROUND_PROMPT" \
      '. + [{"role": "user", "content": $content}]')

    RESPONSE=$(call_model "${MODELS[$i]}" "$MESSAGES")
    echo "$RESPONSE" | tee "$OUTPUT_DIR/r${round}_${MODEL_NAMES[$i]}.txt"

    CONVERSATION=$(echo "$CONVERSATION" | jq \
      --arg name "${MODEL_NAMES[$i]}" \
      --arg round "Round $round" \
      --arg resp "$RESPONSE" \
      '. + [{"role": "assistant", "content": ("[" + $name + " " + $round + "]: " + $resp)}]')

    echo ""
    echo "---"
  done
done

# Final synthesis: Ask each model to vote on the top 3 priorities
echo ""
echo "--- FINAL VOTE ---"
VOTE_PROMPT="Based on the ENTIRE debate, list your TOP 3 priorities in order. For each: one sentence describing what to build, one sentence describing the algorithm, one sentence describing the test that proves it. No fluff. This is a binding vote."

for i in "${!MODELS[@]}"; do
  echo ""
  echo ">>> ${MODEL_NAMES[$i]} (Final Vote) <<<"

  MESSAGES=$(echo "$CONVERSATION" | jq \
    --arg content "You are ${MODEL_NAMES[$i]}. $VOTE_PROMPT" \
    '. + [{"role": "user", "content": $content}]')

  RESPONSE=$(call_model "${MODELS[$i]}" "$MESSAGES")
  echo "$RESPONSE" | tee "$OUTPUT_DIR/vote_${MODEL_NAMES[$i]}.txt"

  CONVERSATION=$(echo "$CONVERSATION" | jq \
    --arg name "${MODEL_NAMES[$i]}" \
    --arg resp "$RESPONSE" \
    '. + [{"role": "assistant", "content": ("[" + $name + " VOTE]: " + $resp)}]')

  echo ""
  echo "---"
done

# Save full conversation
echo "$CONVERSATION" | jq '.' > "$OUTPUT_DIR/full_conversation.json"
echo ""
echo "=== Debate complete. Full transcript: $OUTPUT_DIR/full_conversation.json ==="
echo "=== Individual responses: $OUTPUT_DIR/ ==="
