# OCR cost per video

> Measured 2026-09-01 from parallel load tests (20x the same 10-second campaign
> video, full docker stack). Test video: 3 unique frames after dedup → 1 OCR call.

| Model | Per video | Per 1,000 videos |
|---|---:|---:|
| **gemini-3.7-flash** (current default) | **$0.008** | ~$8 |
| qwen3.8-max (previous default) | $0.011–0.018* | ~$11–18 |
| glm-5.3-flash (rejected — hallucinates fields) | $0.0003 | ~$0.33 |

\* qwen's higher end includes the OOM-retry re-runs from load-test run 2;
$0.011 is its clean price.

## Wider model sweep (2026-09-01, same 3 frames)

Ground truth = the 11 fields qwen3.8-max and gemini-3.7-flash independently
agree on. One OCR call each; per-video cost is the measured `cost_usd`.

| Model | Fields correct | Invented | Latency | Per video |
|---|---:|---:|---:|---:|
| **openai/gpt-5.4-mini** | 11/11 | 0 | **3.4s** | $0.0030 |
| mistralai/mistral-small-2603 | 11/11 | 0 | 4.8s | **$0.0005** |
| google/gemini-3.5-flash-lite | 11/11 | 0 | 6.3s | $0.0033 |
| anthropic/claude-haiku-4.5 | 11/11 | 0 | 7.8s | $0.0053 |
| bytedance-seed/seed-2.0-mini | 11/11 | 0 | 20.4s | $0.0014 |
| google/gemini-3.7-flash (default) | 11/11 | 0 | 10.8s | $0.0070 |
| qwen/qwen3.8-max | 11/11 | 0 | 29.5s | $0.0111 |
| qwen/qwen3.5-flash-02-23 | — | — | 29.4s | failed (no schema-valid output) |
| z-ai/glm-5.3-flash | 11/11 | **8** | 10.8s | $0.0003 |

Standouts: **gpt-5.4-mini** (fastest + perfect, 2.3x cheaper than the current
default) and **mistral-small-2603** (perfect at 1/14th the default's cost).

Caveat: this is one video with one screen layout. The 2026-08-22 50-frame
benchmark showed models diverge on icon-strip layouts (flash recall 57.6% vs
qwen 70.4%) — any model switch for production should re-run that benchmark
(needs the `frames/` set from the original machine; see `eval-2026-08-22/`).
glm-5.3-flash is rejected outright: it invents values for metrics not on
screen (e.g. likes > interactions, which is impossible).

## What actually drives the cost

It's **per OCR call, not per video**. Each call is ~$0.007–0.008 with flash,
and a call covers up to 12 frames (`OCR_BATCH_SIZE`). So:

- Short video (≤12 unique frames, like the test video): 1 call ≈ **$0.008**
- Longer recording (e.g. 60s, ~2–3 batches after dedup): 2–3 calls ≈ **$0.015–0.025**
- The 1 frame/sec sampling (`EXTRACT_FPS`) + phash dedup are what keep frame
  counts (and thus cost) low — a 60s recording of mostly-static Insights
  screens usually dedups down to well under 24 unique frames.

## Where to watch it live

Every billable call logs an `ai_call` line with its exact `cost_usd` (returned
by OpenRouter per request). The Grafana **"AI Costs (OpenRouter)"** dashboard
(http://127.0.0.1:3000, provisioned from `observability/`) sums spend, call
counts, and tokens over any time range.
