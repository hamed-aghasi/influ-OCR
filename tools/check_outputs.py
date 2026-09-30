#!/usr/bin/env python
"""Offline output-identity harness: prove that a code change did not alter what
the pipeline produces, without sending anything to OpenRouter.

Every check writes one JSON (or text) file into --out; `compare` diffs two such
directories. Empty diff = identical. Checks that need data outside the repo
(Final_Test*/ outputs, videos, a running stack) record `{"skipped": reason}`
instead of failing, so a before/after comparison still sees every check.

    check_outputs.py all --out verify/before/host          # static, wire, aggregate, export-offline, docs-links
    check_outputs.py replay --videos Final_Test2 --out X   # run INSIDE the Docker image (tools/replay_in_docker.sh)
    check_outputs.py export-live --out X                   # against a running stack (ANALYZER_PASSWORD)
    check_outputs.py expect                                # build tools/baselines/final_test2_frames_sent.json
    check_outputs.py compare verify/before/host verify/after/host

The pipeline's only non-deterministic stage is the vision model. Everything
before it (extraction, classifier, scroll selection, Tesseract coverage) is
deterministic for a fixed Tesseract/model, and everything after it (consensus,
export) is a pure function of the stored model replies — which is what these
checks replay.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# On the host the app is a subdirectory; inside the Docker image /app *is* the
# app and tools/ may be bind-mounted anywhere.
APP = next((p for p in (REPO / "instagram_analyzer_app", REPO, Path("/app"), Path.cwd())
            if (p / "processing" / "config.py").exists()), REPO / "instagram_analyzer_app")

SETTINGS_KEYS = (
    "openrouter_model", "openrouter_reasoning_effort", "openrouter_url", "ocr_batch_size",
    "ocr_max_tokens", "ocr_max_retries", "ocr_request_timeout", "classifier_threshold",
    "classifier_image_size", "dark_frame_threshold", "frame_selection", "coverage_selection",
    "coverage_ocr_lang", "frame_interval", "convert_to_720p", "perceptual_hash_size",
    "perceptual_hash_distance",
)
EXPORT_FIXED_HEADERS = [
    "Job ID", "Campaign Date", "Campaign Name", "Product Name", "Company",
    "Filename", "File Type", "Status", "Created At", "Completed At",
    "Total Frames", "Good Frames", "Bad Frames", "Processing Time (s)",
]


# ----------------------------------------------------------------- helpers

def bootstrap(out: Path) -> None:
    """Environment the app expects, before the first `processing` import.
    No database: db_client falls back to its in-memory store."""
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SECRET_KEY", "x" * 32)
    os.environ.pop("DATABASE_URL", None)
    os.environ.setdefault("LOG_DIR", str(out / "_logs"))
    os.environ.setdefault("PROCESSING_DIR", str(out / "_processing"))
    Path(os.environ["LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PROCESSING_DIR"]).mkdir(parents=True, exist_ok=True)
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plain(obj):
    """JSON-safe copy: Paths -> file names, sets -> sorted lists, dict keys likewise."""
    if isinstance(obj, Path):
        return obj.name
    if isinstance(obj, (set, frozenset)):
        return sorted(plain(x) for x in obj)
    if isinstance(obj, dict):
        return {plain(k) if isinstance(k, Path) else str(k) if not isinstance(k, str) else k: plain(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(x) for x in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def dump(out: Path, name: str, obj) -> Path:
    path = out / name
    path.write_text(json.dumps(plain(obj), sort_keys=True, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def skipped(out: Path, name: str, reason: str) -> None:
    print(f"  {name}: SKIPPED ({reason})")
    dump(out, name, {"skipped": reason})


def git_files(*patterns: str):
    """Tracked files matching the patterns, or None when git is unavailable."""
    try:
        res = subprocess.run(["git", "ls-files", "--", *patterns], cwd=REPO, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(l for l in res.stdout.splitlines() if l)


def app_py_files():
    return sorted(p for p in APP.rglob("*.py") if "tests" not in p.relative_to(APP).parts)


# ------------------------------------------------------------------ static

def check_static(out: Path) -> None:
    from processing.config import settings
    from processing.ocr_processor import EXTRACTION_PROMPT, RESPONSE_FORMAT, _METRIC_FIELDS
    from processing.db_client.export import _METRIC_KEYS
    import processing, processing.ocr_processor, processing.db_client, processing.frame_selection, processing.metric_coverage

    schema_default = json.dumps(RESPONSE_FORMAT)
    schema_sorted = json.dumps(RESPONSE_FORMAT, sort_keys=True)
    exports = {}
    for mod in (processing, processing.ocr_processor, processing.db_client,
                processing.frame_selection, processing.metric_coverage):
        exports[mod.__name__] = {n: type(getattr(mod, n)).__name__ for n in mod.__all__}
    try:
        from celery_app import celery
        import tasks  # noqa: F401  (registers the tasks)
        celery_info = {
            "tasks": sorted(n for n in celery.tasks if not n.startswith("celery.")),
            "beat": sorted(celery.conf.beat_schedule or {}),
            "include": list(celery.conf.include or []),
        }
    except Exception as exc:  # noqa: BLE001
        celery_info = {"error": f"{type(exc).__name__}: {exc}"}

    dump(out, "static.json", {
        "prompt": {"sha256": sha(EXTRACTION_PROMPT.encode()), "chars": len(EXTRACTION_PROMPT),
                   "has_braces": "{" in EXTRACTION_PROMPT},
        "response_format": {"sha256": sha(schema_default.encode()), "bytes": len(schema_default),
                            "sha256_sorted": sha(schema_sorted.encode()), "bytes_sorted": len(schema_sorted),
                            "name": RESPONSE_FORMAT["json_schema"]["name"],
                            "strict": RESPONSE_FORMAT["json_schema"]["strict"]},
        "metric_fields": list(_METRIC_FIELDS),
        "export_metric_keys": list(_METRIC_KEYS),
        "export_headers": EXPORT_FIXED_HEADERS + [k.replace("_", " ").title() for k in _METRIC_KEYS],
        "settings": {k: getattr(settings, k) for k in SETTINGS_KEYS},
        "exports": exports,
        "celery": celery_info,
        "python": sys.version.split()[0],
    })

    digests = {}
    tracked = git_files("instagram_analyzer_app/models", "instagram_analyzer_app/templates",
                        "instagram_analyzer_app/static", "instagram_analyzer_app/alembic")
    if tracked is None:  # no git (inside the image): walk the same directories
        for sub in ("models", "templates", "static", "alembic"):
            for p in sorted((APP / sub).rglob("*")):
                if p.is_file() and ".DS_Store" not in p.name:
                    digests[str(p.relative_to(APP))] = sha(p.read_bytes())
    else:
        for rel in tracked:
            digests[rel.split("instagram_analyzer_app/", 1)[1]] = sha((REPO / rel).read_bytes())
    dump(out, "file_digests.json", digests)
    dump(out, "module_paths.json", [str(p.relative_to(APP)) for p in app_py_files()])

    name_rx = re.compile(r'(?:get_logger|job_logger)\(\s*("[^"]+"|__name__)')
    lit_rx = re.compile(r'\b(?:log|logger|LOG)\.(?:info|warning|error|exception|debug|critical)\(\s*(f?"[^"]*"|f?\'[^\']*\')')
    names, lits = {}, {}
    for p in app_py_files():
        text = p.read_text(encoding="utf-8")
        for m in name_rx.finditer(text):
            names[m.group(1)] = names.get(m.group(1), 0) + 1
        for m in lit_rx.finditer(text):
            lits[m.group(1)] = lits.get(m.group(1), 0) + 1
    (out / "logger_names.txt").write_text("".join(f"{v:4d} {k}\n" for k, v in sorted(names.items())))
    (out / "log_literals.txt").write_text("".join(f"{v:4d} {k}\n" for k, v in sorted(lits.items())))
    print(f"  static: prompt {sha(EXTRACTION_PROMPT.encode())[:12]} ({len(EXTRACTION_PROMPT)} chars), "
          f"schema {sha(schema_sorted.encode())[:12]} ({len(schema_sorted)} B), "
          f"{len(digests)} file digests, {len(lits)} log literals")


# -------------------------------------------------------------------- wire

def synthetic_frames(out: Path, n: int = 3):
    """Deterministic JPEGs (no external data): a gradient with a frame index band."""
    import numpy as np
    import cv2
    d = out / "_frames"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = np.zeros((1280, 720, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 720, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, 1280, dtype=np.uint8)[:, None]
        img[100 * (i + 1): 100 * (i + 2), :, 2] = 255
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok
        p = d / f"frame_{i:06d}.jpg"
        p.write_bytes(buf.tobytes())
        paths.append(p)
    return paths


def check_wire(out: Path, frames=None) -> None:
    from processing.config import settings
    from processing.ocr_processor import client

    paths = [Path(f) for f in frames] if frames else synthetic_frames(out)
    captured = {}

    class FakeResp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"frames": []}'}}]}

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None):
            captured.update(url=url, timeout=timeout, body=json,
                            headers={k: ("<redacted>" if k == "Authorization" else v) for k, v in (headers or {}).items()})
            return FakeResp()

    old_session, old_key = client._session, settings.openrouter_api_key
    client._session = lambda: FakeSession()
    settings.openrouter_api_key = "check-dummy-key"
    try:
        log = client.job_logger("processing.ocr_processor", "check")
        encoded = client._encode(paths, log)
        result = client._call_api(encoded, log)
    finally:
        client._session, settings.openrouter_api_key = old_session, old_key

    body = captured["body"]
    readable = json.loads(json.dumps(body))
    images = []
    for part in readable["messages"][0]["content"]:
        if part.get("type") == "image_url":
            prefix, b64 = part["image_url"]["url"].split(",", 1)
            digest = sha(base64.b64decode(b64))
            images.append(digest)
            part["image_url"]["url"] = f"{prefix},<sha256:{digest}>"
    dump(out, "wire.json", {
        "url": captured["url"], "timeout": captured["timeout"], "headers": captured["headers"],
        "body_sha256_sorted": sha(json.dumps(body, sort_keys=True).encode()),
        "body_sha256": sha(json.dumps(body).encode()),
        "body_bytes": len(json.dumps(body)),
        "frames": [(p.name, sha(p.read_bytes())) for p in paths],
        "encoded": [(name, sha(base64.b64decode(b64))) for name, b64 in encoded],
        "image_digests": images,
        "readable_body": readable,
        "parsed_result": result,
    })
    print(f"  wire: body {sha(json.dumps(body, sort_keys=True).encode())[:12]} ({len(json.dumps(body))} B), {len(paths)} frames")


# --------------------------------------------------------------- aggregate

def stored_documents():
    """Every stored metrics document with all_frames_data: (label, doc)."""
    docs = []
    ft2 = REPO / "Final_Test2_output"
    for d in sorted(p for p in ft2.glob("*") if p.is_dir()) if ft2.exists() else []:
        f = d / f"{d.name}.json"
        if f.exists():
            docs.append((f"Final_Test2_output/{d.name}", json.loads(f.read_text(encoding="utf-8"))))
    ft = REPO / "Final_Test_output"
    if ft.exists():
        for f in sorted(ft.glob("*.json")) + sorted(ft.glob("rerun-*/*.json")):
            if f.name.startswith("_"):
                continue
            docs.append((str(f.relative_to(REPO)), json.loads(f.read_text(encoding="utf-8"))))
    return [(k, d) for k, d in docs if isinstance(d, dict) and "all_frames_data" in d]


def check_aggregate(out: Path) -> None:
    from processing.ocr_processor import assemble_metrics, _aggregate

    docs = stored_documents()
    result, mismatches = {}, []
    for label, d in docs:
        m = assemble_metrics(d["all_frames_data"], total_frames=d["total_frames"],
                             unique_frames=d["unique_frames"], duplicate_frames=d["duplicate_frames"],
                             batches_total=d["ocr_batches_total"], batches_failed=d["ocr_batches_failed"],
                             frames_missing=d.get("ocr_frames_missing", 0))
        # documents written before a counter existed (ocr_frames_missing, 09-26) lack it: not comparable, not a mismatch
        top = {k: m[k] == d[k] for k in m if k in d and k not in ("extraction_date", "all_frames_data", "summary", "summary_quality")}
        ok = m["summary"] == d["summary"] and m["summary_quality"] == d["summary_quality"] and all(top.values())
        if not ok:
            mismatches.append(label)
        result[label] = {"summary": m["summary"], "summary_quality": m["summary_quality"],
                         "matches_stored": ok, "top_level": top}
    bench = REPO / "docs" / "eval-2026-08-22"
    for f in sorted(bench.glob("bench_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        s, q = _aggregate([{"metrics": v} for v in d.values()])
        result[f"docs/eval-2026-08-22/{f.name}"] = {"summary": s, "summary_quality": q}
    dump(out, "aggregate.json", result)
    print(f"  aggregate: {len(docs)} stored documents replayed, {len(docs) - len(mismatches)} match stored"
          + (f", MISMATCH: {mismatches}" if mismatches else ""))
    if mismatches:
        raise SystemExit("aggregate replay does not match stored summaries")


# ------------------------------------------------------------------ export

def normalise_xlsx(data: bytes) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data))
    ws = wb.active
    rows = [[repr(c.value) for c in row] for row in ws.iter_rows()]
    header_style = [(c.font.bold, c.font.color.rgb if c.font.color else None,
                     c.fill.fgColor.rgb if c.fill and c.fill.fill_type else None,
                     c.alignment.horizontal) for c in ws[1]]
    row_fill = [ws.cell(row=r, column=1).fill.fgColor.rgb if ws.cell(row=r, column=1).fill.fill_type else None
                for r in range(2, ws.max_row + 1)]
    widths = {k: v.width for k, v in ws.column_dimensions.items()}
    return {"sheets": wb.sheetnames, "title": ws.title, "freeze_panes": ws.freeze_panes,
            "widths": widths, "header": rows[0] if rows else [], "header_style": header_style,
            "rows": rows[1:], "row_fill": row_fill, "n_rows": len(rows) - 1}


def check_export_offline(out: Path) -> None:
    """Seed the in-memory store from the stored Final_Test2 documents and export."""
    from processing.db_client.memory import _memory_jobs, _memory_metrics, _memory_lock
    from processing.db_client import export_to_excel

    docs = [(k, d) for k, d in stored_documents() if k.startswith("Final_Test2_output/") and "_job" in d]
    if not docs:
        return skipped(out, "export_offline.json", "no Final_Test2_output documents with a _job block")
    base = datetime(2026, 9, 29, 0, 0, 0)
    with _memory_lock:
        _memory_jobs.clear()
        _memory_metrics.clear()
        for i, (label, d) in enumerate(docs):
            job = d["_job"]
            stem = label.split("/", 1)[1]
            created = base + timedelta(minutes=i)
            _memory_jobs[job["job_id"]] = {
                "id": job["job_id"], "campaign_date": date(2026, 9, 29), "campaign_name": "Final_Test2",
                "product_name": stem, "company": "final-test2", "filename": job.get("video", f"{stem}.mp4"),
                "file_type": "video", "status": job.get("status"), "created_at": created,
                "completed_at": created + timedelta(seconds=job.get("processing_time_seconds") or 0),
                "error_message": job.get("error_message"),
            }
            _memory_metrics[job["job_id"]] = {
                "job_id": job["job_id"], "total_frames": job.get("total_frames"),
                "good_frames": job.get("good_frames"), "bad_frames": job.get("bad_frames"),
                "processing_time_seconds": job.get("processing_time_seconds"),
                "metrics_json": {k: v for k, v in d.items() if k != "_job"}, "created_at": created,
            }
    data = export_to_excel()
    with _memory_lock:
        _memory_jobs.clear()
        _memory_metrics.clear()
    norm = normalise_xlsx(data)
    dump(out, "export_offline.json", norm)
    print(f"  export-offline: {norm['n_rows']} rows x {len(norm['header'])} columns")


def check_export_live(out: Path, base_url: str, user: str, password: str, trust_env: bool) -> None:
    import requests
    s = requests.Session()
    s.trust_env = trust_env
    r = s.post(f"{base_url}/login", data={"username": user, "password": password}, allow_redirects=False)
    if r.status_code != 303:
        return skipped(out, "export_live.json", f"login failed: HTTP {r.status_code}")
    r = s.get(f"{base_url}/export")
    if r.status_code != 200:
        return skipped(out, "export_live.json", f"/export HTTP {r.status_code}")
    (out / "export_live.xlsx").write_bytes(r.content)
    norm = normalise_xlsx(r.content)
    dump(out, "export_live.json", norm)
    (out / "jobs_live.html").write_text(s.get(f"{base_url}/jobs").text, encoding="utf-8")
    job_ids = [d["_job"]["job_id"] for k, d in stored_documents() if k.startswith("Final_Test2_output/") and "_job" in d]
    api = {}
    for jid in job_ids:
        api[jid] = s.get(f"{base_url}/api/job/{jid}").json()
        (out / f"status_live_{jid}.html").write_text(s.get(f"{base_url}/status/{jid}").text, encoding="utf-8")
    dump(out, "api_jobs_live.json", api)
    print(f"  export-live: {norm['n_rows']} rows, {len(job_ids)} job pages captured")


# ------------------------------------------------------------------ replay

def check_replay(out: Path, videos: Path, expect: Path, name: str = "replay.json") -> None:
    from processing.frame_extractor import extract_frames_from_video
    from processing.frame_classifier import classify_frames
    from processing.selection_pipeline import choose_ocr_frames
    from processing.metric_coverage import tesseract_available
    from processing.ocr_processor import chunk_batches

    expected = json.loads(expect.read_text(encoding="utf-8")) if expect and expect.exists() else {}
    work = Path(os.environ["PROCESSING_DIR"]) / "replay"
    result, env = {}, {}
    try:
        env["tesseract"] = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        env["tesseract"] = f"unavailable: {exc}"
    env["tesseract_available"] = tesseract_available()
    for mod in ("cv2", "tensorflow", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            env[mod] = f"unavailable: {exc}"
    failures = []
    for video in sorted(p for p in videos.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}):
        stem = video.stem
        job_dir = work / stem
        count, frames = extract_frames_from_video(video, job_dir / "frames", job_id=stem)
        cls = classify_frames(frames, output_dir=None, job_id=stem)
        good = cls["good_paths"]
        scores = {Path(f["path"]): f["confidence"] for f in cls["good_frames"]}
        choice = choose_ocr_frames([frames], good, scores, job_id=stem)
        cov = choice.coverage
        entry = {
            "extracted": count,
            "frame_digests": {p.name: sha(p.read_bytes()) for p in frames},
            "good": [p.name for p in good],
            "scores": {Path(f["path"]).name: round(f["confidence"], 6) for f in cls["good_frames"] + cls["bad_frames"]},
            "strategy": choice.selection.strategy,
            "scroll_selected": [p.name for p in choice.selection.selected],
            "scroll_duplicates": len(choice.selection.duplicates),
            "pages": choice.selection.pages,
            "coverage": None if cov is None else {
                "applied": cov.applied, "reason": cov.reason,
                "selected": [p.name for p in cov.selected], "dropped": [p.name for p in cov.dropped],
                "metrics": {p.name: sorted(m) for p, m in cov.metrics.items()},
                "values": {p.name: sorted(v) for p, v in cov.values.items()},
                "reasons": {p.name: r for p, r in cov.reasons.items()},
            },
            "dropped_as_duplicate": [p.name for p in choice.dropped_as_duplicate],
            "final": [p.name for p in choice.final],
            "batches": [len(b) for b in chunk_batches(choice.final)],
        }
        exp = expected.get(stem)
        if exp:
            checks = {
                "frames_sent": entry["final"] == exp["frames_sent"],
                "total_frames": entry["extracted"] == exp.get("total_frames"),
                "good_frames": len(entry["good"]) == exp.get("good_frames"),
                "coverage_in": exp.get("coverage_in") is None or len(entry["scroll_selected"]) == exp["coverage_in"],
                "coverage_out": exp.get("coverage_out") is None or (cov is not None and len(cov.selected) == exp["coverage_out"]),
            }
            entry["matches_expected"] = all(checks.values())
            entry["expected_checks"] = checks
            if not entry["matches_expected"]:
                failures.append((stem, checks, entry["final"], exp["frames_sent"]))
        result[stem] = entry
        print(f"  replay {stem[:8]}: {count} frames, {len(good)} good, {len(entry['scroll_selected'])} scroll, "
              f"{len(entry['final'])} sent" + ("" if not exp else ("  OK" if entry["matches_expected"] else "  MISMATCH")))
    result["env"] = env
    dump(out, name, result)
    n_exp = sum(1 for k in result if k != "env" and "matches_expected" in result[k])
    print(f"  replay: {len(result) - 1} videos, {n_exp - len(failures)}/{n_exp} match the expectation")
    for stem, checks, got, want in failures:
        print(f"    {stem}: {checks}\n      got  {got}\n      want {want}")
    if failures:
        raise SystemExit("replay does not match tools/baselines/final_test2_frames_sent.json")


def build_expectation(out_file: Path) -> None:
    """From Final_Test2_output: the frames the reference run sent, per video."""
    cov_rx = re.compile(r"Coverage: (\d+) -> (\d+) frames \((\d+) unrecognised kept\), (\d+) metrics")
    sel_rx = re.compile(r"Selection \(scroll\): (\d+) unique / (\d+) duplicates from (\d+) good frames -> (\d+) OCR batches")
    exp = {}
    for label, d in stored_documents():
        if not label.startswith("Final_Test2_output/") or "_job" not in d:
            continue
        stem = label.split("/", 1)[1]
        job = d["_job"]
        entry = {"job_id": job["job_id"], "frames_sent": [f.get("actual_frame") for f in d["all_frames_data"]],
                 "total_frames": job.get("total_frames"), "good_frames": job.get("good_frames"),
                 "bad_frames": job.get("bad_frames"), "unique_frames": d.get("unique_frames"),
                 "ocr_frames_missing": d.get("ocr_frames_missing"), "ocr_batches_failed": d.get("ocr_batches_failed")}
        log = REPO / "Final_Test2_output" / stem / "worker.log"
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            m = cov_rx.search(text)
            if m:
                entry.update(coverage_in=int(m.group(1)), coverage_out=int(m.group(2)),
                             unrecognised_kept=int(m.group(3)), coverage_metrics=int(m.group(4)))
            m = sel_rx.search(text)
            if m:
                entry.update(scroll_unique=int(m.group(1)), scroll_duplicates=int(m.group(2)),
                             batches=int(m.group(4)))
        exp[stem] = entry
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(exp, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(f"  expect: {len(exp)} videos -> {out_file.relative_to(REPO) if out_file.is_relative_to(REPO) else out_file}")


# --------------------------------------------------------------- docs links

def check_docs_links(out: Path) -> None:
    md_files = git_files("*.md", "**/*.md") or [str(p.relative_to(REPO)) for p in REPO.rglob("*.md")]
    ignored_roots = ("train/", "test-frame/", "review-rounds/", "training/runs/", "Final_Test", ".venv-train/",
                     "training/_trace_work/", "training/logs/", "models/backup_")
    # files the tools generate per run (never tracked), so a doc may name them
    generated = {"feedback.json", "review.html", "summary.json", "groups.json", "frames.html", "truth.json",
                 "clusters.json", "video_info.json", "history.json", "per_video.md", "errors.md",
                 "threshold_sweep.json", "model_metadata.json", "frame_trace.json", "classifier_conditions.json",
                 "extraction_metadata.json", "instagram_metrics.json", "classification_summary.json",
                 "campaign_summary.json", "index.html", "index.json", "frames.csv", "added.json", "labels.json",
                 "_run_report.json"}
    archival = ("docs/history/",)   # kept as written; their references describe the tree of their day
    tracked_all = git_files() or [str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file()]
    basenames = {Path(t).name for t in tracked_all}
    token_rx = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|md|yml|yaml|json|txt|html|example|tsv|ini|sh))`|\]\(([A-Za-z0-9_./\-]+)\)")
    report, missing_total = {}, 0
    for rel in sorted(md_files):
        p = REPO / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        missing, data_refs = [], []
        for m in token_rx.finditer(text):
            ref = (m.group(1) or m.group(2)).split("#", 1)[0]
            if not ref or ref.startswith("http") or "/" not in ref and "." not in ref:
                continue
            cands = [REPO / ref, p.parent / ref, APP / ref, APP / "processing" / ref, REPO / "training" / ref]
            if any(c.exists() for c in cands):
                continue
            base = Path(ref).name
            if base in generated or any(ref.startswith(r) or f"/{r}" in ref for r in ignored_roots):
                data_refs.append(ref)
                continue
            if "/" not in ref and base in basenames:   # bare file name that exists somewhere in the tree
                continue
            missing.append(ref)
        if rel.startswith(archival):
            if missing:
                report[rel] = {"archival_stale": sorted(set(missing))}
            continue
        if missing or data_refs:
            report[rel] = {"missing": sorted(set(missing)), "ignored_data": sorted(set(data_refs))}
        missing_total += len(set(missing))
    report["_summary"] = {"files": len(md_files), "missing_refs": missing_total}
    dump(out, "docs_links.json", report)
    print(f"  docs-links: {len(md_files)} markdown files, {missing_total} dangling references")


# ----------------------------------------------------------------- compare

def compare(a: Path, b: Path) -> int:
    diffs = 0

    def is_result(p: Path, root: Path) -> bool:   # _logs/, _processing/, _frames/ are scratch, not results
        return p.is_file() and not any(part.startswith("_") for part in p.relative_to(root).parts)

    for pa in sorted(p for p in a.rglob("*") if is_result(p, a)):
        rel = pa.relative_to(a)
        pb = b / rel
        if not pb.exists():
            print(f"MISSING in {b}: {rel}")
            diffs += 1
            continue
        da, db = pa.read_bytes(), pb.read_bytes()
        if rel.suffix == ".json" and "replay" in rel.name:
            ja, jb = json.loads(da), json.loads(db)
            ja.pop("env", None)
            jb.pop("env", None)
            same = ja == jb
        elif rel.name == "api_jobs_live.json":
            same = json.loads(da) == json.loads(db)
        elif rel.suffix == ".xlsx":
            continue  # bytes carry docProps timestamps; export_*.json holds the cells
        elif rel.name == "docs_links.json":
            continue  # documentation is expected to change; its own summary is reported by docs-links
        else:
            same = da == db
        if not same:
            print(f"DIFFERENT: {rel}")
            diffs += 1
    for pb in sorted(p for p in b.rglob("*") if is_result(p, b)):
        if not (a / pb.relative_to(b)).exists():
            print(f"NEW in {b}: {pb.relative_to(b)}")
            diffs += 1
    print("IDENTICAL" if diffs == 0 else f"{diffs} difference(s)")
    return 1 if diffs else 0


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("static", "wire", "aggregate", "export-offline", "export-live", "docs-links", "all", "replay"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", type=Path, required=True, help="directory for this run's files")
        if name in ("wire", "all"):
            sp.add_argument("--frames", nargs="*", help="real JPEGs instead of synthetic ones")
        if name == "export-live":
            sp.add_argument("--base-url", default="http://localhost:8010")
            sp.add_argument("--user", default=os.environ.get("ANALYZER_USER", "admin"))
            sp.add_argument("--password", default=os.environ.get("ANALYZER_PASSWORD"))
            sp.add_argument("--trust-env", action="store_true", help="honour proxy env vars (off: VPN proxies 502 localhost)")
        if name == "replay":
            sp.add_argument("--videos", type=Path, required=True)
            sp.add_argument("--expect", type=Path, default=HERE / "baselines" / "final_test2_frames_sent.json")
            sp.add_argument("--name", default="replay.json")
    sp = sub.add_parser("expect", help="build tools/baselines/final_test2_frames_sent.json from Final_Test2_output")
    sp.add_argument("--out-file", type=Path, default=HERE / "baselines" / "final_test2_frames_sent.json")
    sp = sub.add_parser("compare")
    sp.add_argument("a", type=Path)
    sp.add_argument("b", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        return compare(args.a, args.b)
    if args.cmd == "expect":
        build_expectation(args.out_file)
        return 0
    bootstrap(args.out)
    print(f"check_outputs {args.cmd} -> {args.out}   (app: {APP})")
    if args.cmd in ("static", "all"):
        check_static(args.out)
    if args.cmd in ("wire", "all"):
        check_wire(args.out, args.frames)
    if args.cmd in ("aggregate", "all"):
        check_aggregate(args.out)
    if args.cmd in ("export-offline", "all"):
        check_export_offline(args.out)
    if args.cmd in ("docs-links", "all"):
        check_docs_links(args.out)
    if args.cmd == "export-live":
        if not args.password:
            return skipped(args.out, "export_live.json", "no password (--password or ANALYZER_PASSWORD)") or 2
        check_export_live(args.out, args.base_url, args.user, args.password, args.trust_env)
    if args.cmd == "replay":
        check_replay(args.out, args.videos, args.expect, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
