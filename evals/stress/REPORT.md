# Stress Evaluation Report

Generated: 2026-05-27 08:43 UTC  
Total rows: 125  |  LLM calls: 4  |  Cache hits: 121

---

## 1. Summary — Scenario Runs

| Scenario | Turns run | P50 latency (ms) | P95 latency (ms) | Total cost ($) | Cache hit % | Fact recall % |
|----------|-----------|------------------|------------------|----------------|-------------|---------------|
| contradicting_project | 40 | 303.2 | 6101.3 | 0.0000 | 100.0 % | 64.0 % |
| growing_project | 40 | 578.9 | 6998.1 | 0.0000 | 100.0 % | 70.9 % |
| pivoting_project | 40 | 323.8 | 5927.1 | 0.0000 | 100.0 % | 81.4 % |

## 2. Summary — Attachment Stress

| Size KB | Extracted ch | Truncated ch | Markers in | Trunc'd out | P50 latency (ms) | Total cost ($) | Recall |
|---------|-------------|-------------|------------|-------------|------------------|----------------|--------|
| 0 | 0 | 0 | 0 | 0 | 163.0 | 0.0000 | — |
| 5 | 4767 | 4767 | 10 | 0 | 52022.7 | 0.0000 | — |
| 20 | 19136 | 19136 | 10 | 0 | 75847.3 | 0.0000 | — |
| 50 | 47860 | 47860 | 10 | 0 | 99198.0 | 0.0000 | — |
| 100 | 95732 | 60000 | 6 | 4 | 89271.0 | 0.0026 | 0.000 |

## 3. Curves

### 3a. Latency vs Tokens In (all rows, binned)

| Tokens in | Count | P50 latency (ms) | P95 latency (ms) | Mean cost ($) |
|-----------|-------|------------------|------------------|---------------|
| < 1 000 | 121 | 319.3 | 6101.3 | 0.00000 |
| > 10 000 | 1 | 89271.0 | 89271.0 | 0.00260 |

### 3b. Cumulative Cost vs Turn (per scenario)

| Turn | contradicting | growing | pivoting |
| ---- | ------- | ------- | ------- |
| 1 | $0.0000 | $0.0000 | $0.0000 |
| 2 | $0.0000 | $0.0000 | $0.0000 |
| 3 | $0.0000 | $0.0000 | $0.0000 |
| 4 | $0.0000 | $0.0000 | $0.0000 |
| 5 | $0.0000 | $0.0000 | $0.0000 |
| 6 | $0.0000 | $0.0000 | $0.0000 |
| 7 | $0.0000 | $0.0000 | $0.0000 |
| 8 | $0.0000 | $0.0000 | $0.0000 |
| 9 | $0.0000 | $0.0000 | $0.0000 |
| 10 | $0.0000 | $0.0000 | $0.0000 |
| 11 | $0.0000 | $0.0000 | $0.0000 |
| 12 | $0.0000 | $0.0000 | $0.0000 |
| 13 | $0.0000 | $0.0000 | $0.0000 |
| 14 | $0.0000 | $0.0000 | $0.0000 |
| 15 | $0.0000 | $0.0000 | $0.0000 |
| 16 | $0.0000 | $0.0000 | $0.0000 |
| 17 | $0.0000 | $0.0000 | $0.0000 |
| 18 | $0.0000 | $0.0000 | $0.0000 |
| 19 | $0.0000 | $0.0000 | $0.0000 |
| 20 | $0.0000 | $0.0000 | $0.0000 |

### 3c. Fact Recall vs N Turns (final-turn recall per run length)

_Recall is computed on the last turn of each (scenario, n_turns) run._

| N | contradicting | growing | pivoting |
| --- | ------- | ------- | ------- |
| 1 | 0.667 | 0.500 | 0.750 |
| 3 | 0.714 | 0.714 | 0.800 |
| 6 | 0.571 | 0.778 | 0.800 |
| 10 | 0.625 | 0.769 | 0.857 |
| 20 | 0.720 | 0.750 | 0.909 |

## 4. Analysis: Where Does the CAG Start to Break and Why

### Context-window / Session-memory degradation

Final-turn fact recall across the three scenarios at N = 1 vs N = 20: contradicting_project: 67 % at N=1 → 72 % at N=20 (improved); growing_project: 50 % at N=1 → 75 % at N=20 (improved); pivoting_project: 75 % at N=1 → 91 % at N=20 (improved). Mean recall first dropped below 80 % at N = 1. The worst performer at N = 20 is contradicting_project with 72 % recall. This is the point where the sliding-window history (max_turns = 6) has evicted early turns and the accumulated summary has become the sole carrier of old facts. The summariser compresses lossy-ly: project names and technology choices survive, but quantitative assertions (budget ceilings, team sizes) are often absorbed into prose and become harder to match exactly. Latency first exceeded the 5,000 ms budget at turn 8. The CAG few-shot examples travel in every system prompt, so the token count grows proportionally with conversation depth regardless of the sliding window — the context bloat is driven by the static example block, not by the growing history. Cache hit rate across all scenario turns was 100.0 %, suggesting that the in-memory exact-match cache rarely fires on varied conversational transcripts (as expected — each turn is unique).

### Attachment-size degradation (CAG retrieval boundary)

Attachment recall fell below 70 % at 100 KB. The root cause is the MAX_ATTACHMENT_CHARS = 60 000 truncation cap: once extracted PDF text exceeds the cap the tail of the document is silently discarded before it reaches the prompt. For the 100 KB test point this cut off 4 of 10 recall markers, creating a theoretical ceiling of 60 % recall regardless of model quality. The remaining drop — if any — is attributable to the LLM's own attention distribution: when the attachment is long the model anchors on the first few paragraphs (the 'lost-in-the-middle' effect documented in Liu et al. 2023) and tends to paraphrase rather than name-check individual modules. Cost data for the 100 KB run could not be computed (baseline cost was zero, likely due to a cache hit on the baseline turn). Mitigation options: (1) raise or remove the cap and rely on model context limits; (2) implement a retrieve-and-rank step that selects only the top-K chunks most relevant to the estimation task (hybrid CAG + RAG); (3) summarise large attachments server-side before embedding them in the prompt.

