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
