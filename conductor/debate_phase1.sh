#!/bin/bash
# HMS Focused Debates - Phase 1 (Rounds 1-3)
# 5 parallel debates × 5 models × 3 rounds = 75 API calls

set -euo pipefail

OPENROUTER_KEY="${OPENROUTER_API_KEY}"
BASE_DIR="/Volumes/A/HMS/conductor/debate_phase1"
mkdir -p "$BASE_DIR"

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

FIRST_DEBATE_CONTEXT=$(cat <<'CTX'
## Background From First Debate (36 turns, 6 models, 5 rounds)

Key conclusions that survived demolition:
1. AtomMemory: k-sparse Modern Hopfield attractor over sharded inverted-index (64 shards × 256 dims). Overlap scan + softmax attention + top-k=64 projection per iteration. Converges in 1-3 steps. Energy E=-lse(β,overlaps) is a Lyapunov function.
2. CompositeMemory: Second inverted index storing canonical triples T = S ⊕ ρ₁(R) ⊕ ρ₂(O) with cyclic-shift role binding (odd shifts coprime with D=16384).
3. RoleAlgebra: Permutation-based role binding fixes XOR commutativity (current h⊕r⊕t can't distinguish subject from object).
4. fuzzy_structural_query: Build query from known roles → overlap scan CompositeMemory → per-composite unbind → AtomMemory cleanup → aggregate confidence.
5. TripleStore: Materialized fallback for fan-out > 40.

What was KILLED:
- Dense FHRR layer + FFT (complexity, sync bugs)
- Superposition KV plates (threshold-bundling breaks XOR-unbind)
- "Algebraic multi-hop independent of branching" (frontier capacity wall at ~100 items)
- Auto-discovery of relation compositions (dismissed as research)
- Parity bundling (incompatible with sparsity at scale)

HMS specs: D=16384 (2^14), rho=1/256, k=64 active indices, Jaccard similarity, sparse sorted-u32 index arrays, Rust, no new external crates.
CTX
)

call_model() {
  local model="$1"
  local messages="$2"
  local response
  response=$(curl -s --max-time 120 -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer $OPENROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/writerslogic/hms" \
    -H "X-Title: HMS Debate Phase 1" \
    -d "$(jq -n \
      --arg model "$model" \
      --argjson messages "$messages" \
      '{model: $model, messages: $messages, temperature: 0.8, max_tokens: 4096}')" 2>/dev/null)
  echo "$response" | jq -r '.choices[0].message.content // .error.message // "ERROR: No response"' 2>/dev/null || echo "ERROR: parse failed"
}

run_debate() {
  local debate_id="$1"
  local debate_name="$2"
  local system_prompt="$3"
  local r1_prompt="$4"
  local r2_prompt="$5"
  local r3_prompt="$6"
  local outdir="$BASE_DIR/$debate_id"
  mkdir -p "$outdir"

  echo "=== DEBATE $debate_id: $debate_name ===" | tee "$outdir/transcript.txt"

  local CONVERSATION="[]"
  CONVERSATION=$(echo "$CONVERSATION" | jq --arg sys "$system_prompt" '. + [{"role": "system", "content": $sys}]')

  local prompts=("$r1_prompt" "$r2_prompt" "$r3_prompt")
  local round_names=("Opening Positions" "Challenge & Demolish" "Concrete Implementation")

  for round in 0 1 2; do
    local rn=$((round + 1))
    echo "" | tee -a "$outdir/transcript.txt"
    echo "--- Round $rn: ${round_names[$round]} ---" | tee -a "$outdir/transcript.txt"

    for i in "${!MODELS[@]}"; do
      echo "" | tee -a "$outdir/transcript.txt"
      echo ">>> ${MODEL_NAMES[$i]} (R$rn) <<<" | tee -a "$outdir/transcript.txt"

      local MESSAGES
      MESSAGES=$(echo "$CONVERSATION" | jq \
        --arg content "You are ${MODEL_NAMES[$i]}. Round $rn of 3. ${prompts[$round]}" \
        '. + [{"role": "user", "content": $content}]')

      local RESPONSE
      RESPONSE=$(call_model "${MODELS[$i]}" "$MESSAGES")
      echo "$RESPONSE" | tee -a "$outdir/transcript.txt"

      CONVERSATION=$(echo "$CONVERSATION" | jq \
        --arg name "${MODEL_NAMES[$i]}" \
        --arg rn "R$rn" \
        --arg resp "$RESPONSE" \
        '. + [{"role": "assistant", "content": ("[" + $name + " " + $rn + "]: " + $resp)}]')

      echo "---" | tee -a "$outdir/transcript.txt"
    done
  done

  echo "$CONVERSATION" | jq '.' > "$outdir/conversation.json"
  echo "=== DEBATE $debate_id PHASE 1 COMPLETE ===" | tee -a "$outdir/transcript.txt"
}

# ============================================================
# DEBATE A: C_max + Embedding Collapse + Plate Revival
# ============================================================
run_debate "A" "C_max + Embedding Collapse + Plate Revival" \
"You are in a focused technical debate about solving the frontier capacity wall (C_max) in HMS. $FIRST_DEBATE_CONTEXT

The C_max problem: bundling N frontier nodes into 1 vector hits a noise wall at ~100 items (BSC, D=16384, k=64). This kills holographic multi-hop traversal at realistic branching factors. The first debate tabled this as 'fall back to materialized traversal.' We want to SOLVE it.

Key new ideas to evaluate:
1. Continuous-count attractor: keep raw [u16;16384] counts, never threshold to binary. Feed continuous counts to Hopfield. Does this increase capacity?
2. Attention-weighted collapse: weight each frontier node by relevance to the NEXT hop's relation before bundling. Effective frontier = relevant nodes only.
3. Resonator networks (Frady 2020): iterative factorization designed for high-item superpositions.
4. Multi-centroid frontier: maintain 4-8 centroid vectors via online k-means instead of 1 bundle.
5. Deferred thresholding: can bind/permute operate on count vectors without ever thresholding to binary?
6. Superposition plate revival: if continuous-count approach works, does it resurrect 'N items in 1 vector'?

Be BRUTAL. Derive math. Show complexity bounds. If an idea can't work, prove it can't. No hand-waving." \
\
"ROUND 1: Give your opening position. What is the most promising path to pushing C_max past 1000? Derive the capacity bound for your proposed approach. What specific data structure would you use? What is the EXACT SNR formula for continuous-count attractor vs binary threshold? Can bind/permute operate on count vectors — prove or disprove with the XOR algebra. Be specific enough to implement in Rust." \
\
"ROUND 2: Challenge at least one other model's proposal. Find the mathematical flaw. Then refine your own position. Key question nobody may have answered: if you keep continuous counts and XOR-bind two count vectors, what does the result MEAN algebraically? Is it still invertible? Does the Hopfield energy function still have a Lyapunov certificate on count vectors? If the count vector is [u16;16384] (32KB), what happens to cache performance vs the sparse [u32;64] (256B) that everything else uses?" \
\
"ROUND 3: Show the EXACT Rust data structure and algorithm. Include struct fields, method signatures, and the hot loop. Derive the complexity. What test would PROVE your approach achieves C_max > 1000? Describe the test: how many items bundled, what noise level, what recall threshold. If you can't show concrete code, you're hand-waving." &

# ============================================================
# DEBATE B: True Holographic Properties + Attention-Bundling
# ============================================================
run_debate "B" "True Holographic Properties + Attention-Bundling" \
"You are in a focused technical debate about making HMS genuinely holographic. $FIRST_DEBATE_CONTEXT

The first debate concluded 'stop calling it holographic, call it associative memory.' We reject that conclusion. We want to find what WOULD make it truly holographic.

Key ideas to evaluate:
1. The attractor IS the hologram: stored patterns create an energy landscape. The basin of attraction IS the holographic redundancy. Question: how wide is the basin for k-sparse Modern Hopfield?
2. Multi-scale representation: store each vector at multiple sparsity levels (rho=1/256, 1/64, 1/16). Coarse levels = more noise-tolerant. Pattern completion starts coarse, refines down. This IS multi-resolution holographic encoding.
3. Attention-bundling as universal primitive: replace majority-vote bundle() everywhere with attention-weighted bundle(vecs, context). Every operation becomes context-sensitive.
4. Graceful degradation: the first debate proved sparse BSC doesn't degrade gracefully under index deletion. But does the SYSTEM (sparse BSC + attractor) degrade gracefully? The attractor's basin might compensate.

The bar: what theorem must hold for 'holographic' to survive peer review? Be rigorous." \
\
"ROUND 1: What specific mathematical property must HMS demonstrate to legitimately claim 'holographic memory'? Derive the basin of attraction width for k-sparse Modern Hopfield as a function of N (stored patterns), k (sparsity), D (dimensions), and β (inverse temperature). Does multi-scale encoding (multiple rho levels) provide provable graceful degradation — derive the degradation curve equation. If attention-bundling replaces majority-vote everywhere, what changes in encoding, concept synthesis, and sequence memory?" \
\
"ROUND 2: Challenge another model's claims. Key questions: Is the basin width sufficient for practical pattern completion at N=100K atoms? Does multi-scale encoding actually help, or does it just trade sparsity for capacity? If you claim graceful degradation, show the EXACT curve: recall@10 as a function of corruption fraction, with the attractor. Derive it, don't guess it. Is attention-bundling just a weighted average (boring) or does it create genuinely new capabilities?" \
\
"ROUND 3: Concrete implementation. Show the Rust struct for multi-scale vectors. Show the attractor-based pattern completion algorithm step by step. Show the test that PROVES graceful degradation — the test must produce a smooth monotonic curve, not a cliff. What is the EXACT theorem statement that a peer reviewer would accept as proof of 'holographic' behavior? Write it as: Theorem: For any stored pattern p_i and any corruption c with ||c|| < δ(N,D,k,β), the attractor converges to p_i in at most T iterations." &

# ============================================================
# DEBATE C: Relation Composition + Binding Arity
# ============================================================
run_debate "C" "Relation Composition + Binding Arity Limits" \
"You are in a focused technical debate about relation composition and binding arity in HMS. $FIRST_DEBATE_CONTEXT

The first debate dismissed composition discovery as 'research, not product' and never addressed arity limits. We want to SOLVE both.

Key ideas to evaluate:
1. Composition as emergent attractor property: if many (father,X,Y) and (father,Y,Z) triples coexist with (grandfather,X,Z), the attractor creates a basin near father⊕father that overlaps with grandfather. No explicit rules needed.
2. Hierarchical binding for high arity: instead of flat S⊕ρ₁(R)⊕ρ₂(O)⊕ρ₃(T)⊕ρ₄(L), use binary tree: (S⊕ρ₁(R))⊕ρ₂((O⊕ρ₃(T))⊕ρ₄(L)). Depth=log(arity), not linear. Each level has 2-3 operands.
3. Role vectors optimized via coding theory: choose permutations maximizing pairwise Hamming distance. Error-correcting code families for the permutation group.
4. Composition rules as stored patterns: store grandfather_rule=bundle(father,father,grandfather) as atom. Composition becomes lookup via attractor.

Key mathematical questions: What is the max practical arity? How many supporting triples does composition discovery need? Can hierarchical binding be unbound correctly?" \
\
"ROUND 1: Derive the maximum practical arity for role-binding at D=16384, k=64. At what arity does attractor cleanup fail (recall < 0.5)? Show the math: how does the active-index count of the composite grow with arity, and when does it exceed the attractor's discrimination threshold? For composition: under what EXACT conditions does father⊕father produce a vector with Jaccard > 0.4 to grandfather? How many supporting triples are needed? Can hierarchical binding be correctly unbound — show the unbinding algebra step by step." \
\
"ROUND 2: Challenge another model. Key traps: Does hierarchical binding break commutativity WITHIN a level (good) but create NEW ambiguities between levels (bad)? If composition requires 'many supporting triples,' how many is 'many'? 10? 100? 10000? If you can't give a number, you're guessing. Is there a fundamental information-theoretic limit to how many composition rules can coexist before they interfere? Derive it." \
\
"ROUND 3: Show Rust implementation. Hierarchical binding struct with tree-structured roles. Composition discovery: given a set of stored triples, find all pairs (R1,R2) where R1⊕R1 has Jaccard > threshold with some R3. What's the complexity? What test proves composition works — show the test code. What test proves arity=6 works for events like (agent, action, patient, time, location, instrument)? Show the exact triple encoding formula, the query formula for each role, and the expected recall." &

# ============================================================
# DEBATE D: Self-Supervised Semantic Embeddings
# ============================================================
run_debate "D" "Self-Supervised Semantic Embeddings + Incremental Learning" \
"You are in a focused technical debate about making HMS generate its own semantic embeddings without external models. $FIRST_DEBATE_CONTEXT

Currently HMS encodes text via character trigrams — 'cat' and 'feline' are orthogonal. This makes the system useless for real semantic search without external embeddings. We want HMS to learn semantic similarity from usage alone.

Key ideas to evaluate:
1. Distributional semantics via VSA: context_vec = bundle(vectors of surrounding words within window). Word's semantic embedding = accumulated context bundle. 'cat' and 'feline' appear near 'pet','fur','purring' → embeddings converge.
2. Document-level LSA: bundle all words in document → doc vector. Words in similar doc vectors get similar embeddings. LSA without matrix decomposition.
3. Contrastive learning: 'cats are NOT dogs' — bind 'dog' with negation role before bundling. Pushes them apart.
4. Meta-learning: bundle of all verb contexts IS a verb template. New words classified by template proximity. Parts of speech learned without labels.
5. Polysemy: 'bank' (financial) vs 'bank' (river). Context-conditioned bundling separates senses.
6. Feedback loop: successful queries reinforce embeddings, failed queries weaken them.
7. Cross-lingual: 'chat' (French) and 'cat' (English) in similar contexts → embeddings converge.

Design a complete pipeline using only existing VSA operations. No neural network, no gradient descent, no external API." \
\
"ROUND 1: Design the complete self-supervised embedding pipeline. When memorize_text(id, text) is called, what EXACTLY happens beyond the current trigram encoding? Show: word extraction, context window, context bundling, embedding update. What data structure stores the evolving embeddings? How does it interact with the existing AtomMemory? How many memorize() calls until embeddings stabilize — derive the convergence rate. Show exact Rust structs." \
\
"ROUND 2: Challenge another model's design. Key problems: How do you handle polysemy ('bank' has multiple meanings)? If embeddings are updated incrementally, don't older vectors in the AtomMemory become stale? How do you handle the cold-start problem (first few texts have terrible embeddings)? What's the memory cost — if you maintain a semantic embedding for every word ever seen, how big does this get? Does the feedback loop create a self-reinforcing bias where popular query patterns dominate?" \
\
"ROUND 3: Show the EXACT implementation. Rust struct for SemanticMemory. The memorize_text() pipeline from raw text to updated embeddings. The query path that combines lexical AND semantic similarity. A test that proves: after memorizing 1000 documents about animals, querying 'feline' returns 'cat'-related results even though they share zero trigrams. Show the test code. What's the computational overhead per memorize_text() call — can it stay under 10ms?" &

# ============================================================
# DEBATE E: Adversarial Robustness + Production Hardening
# ============================================================
run_debate "E" "Adversarial Robustness + Production Hardening" \
"You are in a focused technical debate about production hardening HMS. $FIRST_DEBATE_CONTEXT

The first debate designed beautiful static structures but ignored production realities. We want to solve:
1. Attractor poisoning: adversary stores atoms creating trap basins near legitimate atoms. How to detect and prevent?
2. Cascading deletes: delete an atom that appears in 10K composites. Rebuild CompositeMemory? Tombstone? What's the staleness bound?
3. Encoding attack surface: adversary crafts text whose trigram encoding collides with target atom. Collision probability? Defense via salting?
4. QPS ceiling: at 10M atoms with sharded inverted index, what's max queries/second? What's the bottleneck?
5. Incremental index maintenance: insert/delete without rebuilding. What's the write amplification?
6. Memory pressure: at 10M atoms × 64 indices × 4 bytes = 2.5GB for vectors alone. Plus posting lists. How to stay under 8GB RAM?

Be practical. Show Rust code. Derive numbers from first principles." \
\
"ROUND 1: Design a concrete attractor-poisoning attack against HMS. Show: what atoms the adversary stores, how they create a trap basin, what legitimate query gets hijacked. Then design the defense. For cascading deletes: what happens when you delete atom X that appears in 10K composites? What's the correct cascade strategy? Derive the worst-case staleness bound — how many queries return wrong answers before cleanup completes? For QPS: derive the theoretical max at 10M atoms from first principles (memory bandwidth, cache sizes, overlap scan cost)." \
\
"ROUND 2: Challenge another model. Key questions: Is the poisoning defense worse than the attack (does admission control reject legitimate atoms)? If cascading deletes are lazy, can an adversary trigger unbounded staleness by rapidly inserting and deleting? Is the QPS derivation realistic or does it assume perfect cache behavior? What happens under concurrent write load — does the RwLock contention kill read throughput? Show the contention model." \
\
"ROUND 3: Show production-ready Rust code. The admission control struct. The cascading delete state machine. The memory budget calculator (given N atoms and M composites, how much RAM?). A chaos test: randomly insert/delete/query 100K operations across 8 threads, assert zero corruption and bounded latency. What monitoring metrics should HMS expose for production ops (posting list growth rate, attractor convergence time, tombstone ratio, cache hit rate)?" &

# Wait for all 5 debates to complete
echo "All 5 debates launched in parallel. Waiting..."
wait
echo ""
echo "=== ALL PHASE 1 DEBATES COMPLETE ==="
echo "Transcripts:"
for d in A B C D E; do
  lines=$(wc -l < "$BASE_DIR/$d/transcript.txt")
  echo "  Debate $d: $lines lines → $BASE_DIR/$d/transcript.txt"
done
