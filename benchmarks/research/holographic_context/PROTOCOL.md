# Frozen episode-context probe

Registered before execution on 2026-10-24. This synthetic experiment is separate
from official QA datasets and has no model, network or paid API dependency.

The transfer from [Ray and Dancy (2025)](https://arxiv.org/abs/2508.15630) is to
recover a complete episode from partial cues. The control from
[Kumar (2026)](https://arxiv.org/abs/2606.24948) is to measure downstream fact
recovery with a correct intermediate episode, separately from locating that
episode. This is an engineering probe, not a reproduction of either paper.

Freeze: development seed 15630; held-out synthetic audit seeds 21425, 21065,
24948; episode counts 64 and 256; complex dimensions 512 and 2048; 16 queries
per cue condition and corpus. No tuning or selection after audit execution.

Each episode contains nine alternating original turns and a rare Unicode fact
at turn 1. Turns 4, 6 and 8 contain place, day and version cues. Turn 8 also
contains a unique cue. Sessions repeat across two versions with conflicting
rare facts; the three split cues jointly identify one episode. The conditions
are a unique partial cue and three partial cues spread across the episode.

All readers receive at most nine original sources, five original rounds and
4096 serialized UTF-8 bytes, counted with the repository's existing compact
context encoder and exact canonical record IDs. No output text is generated.

Controls and candidate:

- Existing immediate context with the **correct** turn-8 anchor supplied.
  This reproduces the pre-fix case where that anchor is already in the seed.
- Full transitive exact context with the same correct anchor supplied.
- Exact slot/value-to-episode posting intersection, then full exact context.
- One global FHRR memory `M = sum(cue * episode)`; sum unbound cue estimates,
  rank the stored episode codebook, then full exact context. Verify the
  selected episode against every exact cue before admitting its sources.

The phase candidate uses independent unit complex codes, not oscillators or
learned cleanup. All dimensions and seeds run. Exact posting payload is uint32
episode IDs plus uint32 offsets, with token/episode identities shared by both
methods. It must fit within the phase memory vector's payload budget in every
setting. Report phase codebook working payload separately; regenerating codes
would trade that payload for computation. These are representation-byte counts,
not Python resident-memory measurements.

Primary outcomes are exact rare-source recall and episode location accuracy.
Also report abstention, returned source count, actual compact bytes and locator
latency. Latency is a Python mechanism probe, not a Rust serving benchmark.
Keep per-query rows and hashes so aggregate claims are reproducible. The
production correctness gate is the native Rust context regression suite.

Promotion requires strictly better rare-source recall than exact intersection
at the same source/byte limits and within the same sidecar byte budget, with no
incorrect episode/version sources. Ties do not justify a new sidecar. A negative
result leaves the phase probe as research only; it does not claim every possible
compressed or noisy-cue workload is solved by exact postings.
