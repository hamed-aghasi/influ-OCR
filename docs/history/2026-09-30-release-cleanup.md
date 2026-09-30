# 2026-09-30 — release cleanup: what changed, and the proof that outputs did not

Branch `feat/scroll-coverage-selection`, local commits only (nothing pushed; origin is at
`55e66d5`). Reference outputs: `Final_Test2_output/` (19 videos, 2026-09-29, image
`ce0982465cb6`, tagged `influ-ocr:ft2-reference`). Released image: `influ-ocr:post-cleanup` =
`f7d2818cdb45` (also `latest`), running in compose since 2026-09-30.

## Rule

No pipeline output changes: metrics JSON (keys, layout, values), schema, field names, Excel
export, prompt, config defaults, selection/aggregation logic, log messages and logger names.
Feedback that could only be applied by changing what the model sees or how consensus votes was
excluded and is listed under *Known issues & next steps* in `README.md`.

## Commits (after `176662f`, tag `pre-cleanup`)

| commit | what | touches pipeline code? |
|---|---|---|
| `176662f` | 403 stale-session fix in `ocr_processor/client.py` (drop the pooled connection on a 403; request body unchanged) | yes — retry path only |
| `bd47d41` | `HANDOFF.md`, `SESSION-*.md`, `PROXY_SUPPORT_PLAN.md` → `docs/history/` | no |
| `6f12fc6` | `PIPELINE.md` headings/citations use the package paths | no |
| `6dea396` | `README.md`: all 56 settings, real defaults, cost, layout, known issues, unused code | no |
| `428c635` | `RUNBOOK.md` | no |
| `de5ce20` | `training/README.md`: nine scripts, eval gate | no |
| `b09a3b2` | stale module names in comments/docstrings (`config.py`, `metric_coverage/__init__.py`, two training scripts, two READMEs) | comments only |
| `bdf3640` | one `.gitignore`; `Final_Test*/`, `.pytest_cache/` ignored | no |
| `a1deca0` | compose: worker `celery inspect ping` healthcheck, beat none; `.dockerignore` at every depth | no (Dockerfile untouched) |
| `a2b1851` | pyflakes: unused imports, f-string prefixes, unused local; `test_dedup` uses `importorskip` | imports only |
| `2926fb9` | `tests/test_export_columns.py`: export columns = schema fields, order pinned | test only |
| `249985e` | docstrings marking unused paths (`process_frames`, S3 `upload_file`/`delete_job_files`, `get_all_jobs(offset)`, `export_to_excel(output_path)`, `port`, `ocr_delay_seconds`) | docstrings only |
| `92fd8c3`, `5edd027` | `tools/`: `submit_batch.py`, `run_local.py`, `doctor.py`, `check_outputs.py`, `replay_in_docker.sh`, `baselines/` | no (outside the image) |

## Evidence

Baseline captured at `176662f`; every check re-run on the final tree and the final image.

| check | before | after |
|---|---|---|
| `pytest tests/` (host / in image) | 182 | 184 (two new export-column tests) |
| prompt sha256 / chars | `b4c8642bc152…` / 4695 | identical |
| response_format sha256 (sorted) / bytes | `4ea2f1ec8c54…` / 4150 | identical |
| exact OpenRouter request body, 3 synthetic frames (host, reference image, released image) | `8c60350e1c0d…`, 187,539 B | identical in all three |
| package `__all__` maps, Celery tasks + beat, 46 app module paths, logger names, 95 log literals | — | identical |
| consensus replay: `assemble_metrics` over every stored `all_frames_data` (53 documents) vs stored `summary`/`summary_quality` | 53/53 | 53/53, file identical |
| Excel export from the stored documents (in-memory store), normalised cell by cell | 19 rows × 42 cols | identical |
| live: `/export` (33 rows incl. dates), `/api/job/<id>` ×19, `/status/<id>` ×19, `/jobs` | captured on `post-403fix` | identical on `post-cleanup` |
| offline export vs live export, 19 jobs × 28 metric columns + status/frame counts | identical | — |
| selection replay of the 19 videos inside the reference image vs `tools/baselines/final_test2_frames_sent.json` | 19/19 | — |
| the same replay: final code bind-mounted into the reference image | — | identical to the reference replay (frame digests, scores, coverage decisions) |
| the same replay inside the released image | — | identical; `pip freeze` identical to the reference image |
| model files sha256 (host and image) | `fingerprint.pb 4787ef40…`, `saved_model.pb 8e82af10…`, `variables.data 199ba7ca…`, `variables.index 4b6f58b5…`, `model_metadata.json 3ada3bcb…` | identical |
| every tracked app file vs the released image | — | byte-identical; image contains no `.DS_Store`, `__pycache__`, `.pytest_cache`, logs |
| `training/eval_selection.py` (ground truth, 10 videos) | 46 frames / 0 lost | JSON byte-identical |
| `training/eval_feedback.py` (73 videos, 118 checks) | 110/118, 396 frames | JSON byte-identical |
| docs link check (current docs) | 12 dangling | 0 (history files reported separately, as written) |
| tool smokes (one video each, ~1 ¢): `run_local.py`, `submit_batch.py` via the running stack | — | frames sent identical to the reference; document keys identical |

Compose after the release: app and worker `healthy`, beat unflagged (`docker compose ps`).

## Not done (by decision; see README known issues)

`non_followers` Interactions-split confusion; seven Persian prompt labels; provider pinning;
mid-load "Views 0" relabel/retrain; UI error surfaces; CI; default credentials.

## How to re-verify later

```
tools/check_outputs.py all --out verify/x          # host, offline
tools/replay_in_docker.sh influ-ocr:latest Final_Test2 verify/replay   # needs the videos
tools/check_outputs.py compare verify/<before> verify/<after>
```
