#!/usr/bin/env python
"""Submit every video in a folder through the running web app, one at a time,
and collect each job's result.

    ANALYZER_PASSWORD=... tools/submit_batch.py --input videos/ --out results/ \
        --campaign-name "Spring" --company acme [--worker-container ocr-worker-1] [--retry-403 2]

For each video: log in -> POST /upload (the browser's form) -> follow the 303
to /status/<job_id> -> poll /api/job/<job_id> until the job settles -> write
<out>/<stem>/<stem>.json (the job's full metrics document plus a `_job` block:
job id, status, error, frame counts, timing, and — with --worker-container —
the worker's "OpenRouter answered ... provider= ... tokens" lines) and, with
--worker-container, <out>/<stem>/worker.log. Also <out>/_run_report.json.

Never overwrites: a video whose <stem>.json exists is skipped. A failed attempt
that is retried is renamed to <stem>.failed-<reason>.json first.

--retry-403 (needs --worker-container): when a job fails because every
OpenRouter call was refused with HTTP 403 (route blocked, see RUNBOOK.md §2),
wait until the route is back, restart the worker container if its pooled
connection is stale, and resubmit the video, up to N times.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import requests

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
TERMINAL = {"completed", "failed", "error"}


def worker_lines(container: str, job_id: str) -> list:
    p = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
    return [l for l in (p.stdout + p.stderr).splitlines() if f"[{job_id}]" in l]


def route_ok(container: str) -> bool:
    """Probe OpenRouter from the worker container: it is the one making the calls."""
    p = subprocess.run(["docker", "exec", container, "python", "-c",
                        "import requests;print(requests.get('https://openrouter.ai/api/v1/models',timeout=20).status_code)"],
                       capture_output=True, text=True)
    return p.stdout.strip() == "200"


def wait_for_route(container: str, max_minutes: int = 30) -> None:
    waited = 0
    while not route_ok(container):
        if waited == 0:
            print("  route to OpenRouter is down (VPN?); waiting...", flush=True)
        time.sleep(30)
        waited += 30
        if waited >= max_minutes * 60:
            raise SystemExit(f"route still down after {max_minutes} min; giving up")
    if waited:
        print(f"  route back after {waited}s", flush=True)


def restart_worker(container: str) -> None:
    print(f"  restarting {container} (stale pooled connection)", flush=True)
    subprocess.run(["docker", "restart", container], capture_output=True)
    for _ in range(30):
        time.sleep(5)
        p = subprocess.run(["docker", "exec", container, "celery", "-A", "celery_app.celery", "inspect", "ping"],
                           capture_output=True, text=True)
        if "pong" in p.stdout:
            return
    print("  worker did not answer celery ping after 150s", flush=True)


class Submitter:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.trust_env = args.trust_env
        r = self.s.post(f"{args.base_url}/login", data={"username": args.user, "password": args.password},
                        allow_redirects=False)
        if r.status_code != 303:
            raise SystemExit(f"login failed at {args.base_url}/login: HTTP {r.status_code}")

    def run_one(self, video: Path) -> dict:
        a, stem = self.a, video.stem
        out_dir = self.out / stem
        out_json = out_dir / f"{stem}.json"
        if out_json.exists():
            return {"video": video.name, "status": "skipped", "reason": "output exists"}
        started = time.time()
        with open(video, "rb") as f:
            r = self.s.post(f"{a.base_url}/upload",
                            files={"file": (video.name, f, "video/mp4")},
                            data={"campaign_date": a.campaign_date, "campaign_name": a.campaign_name,
                                  "product_name": a.product_name or stem, "company": a.company},
                            allow_redirects=False)
        if r.status_code != 303:
            return {"video": video.name, "status": "upload_error", "http": r.status_code, "body": r.text[:300]}
        job_id = r.headers["location"].rsplit("/", 1)[-1]
        print(f"  job {job_id}", flush=True)

        job = None
        while time.time() - started < a.timeout_minutes * 60:
            time.sleep(a.poll_seconds)
            job = self.s.get(f"{a.base_url}/api/job/{job_id}").json()
            if job.get("status") in TERMINAL:
                break
        job = job or {}
        status = job.get("status", "unknown")
        elapsed = round(time.time() - started, 1)

        out_dir.mkdir(exist_ok=True)
        lines = worker_lines(a.worker_container, job_id) if a.worker_container else []
        if a.worker_container:
            (out_dir / "worker.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        answered = [l.split("OpenRouter answered", 1)[1].strip() for l in lines if "OpenRouter answered" in l]
        problems = [l for l in lines if "WARNING" in l or "ERROR" in l]

        doc = dict(job.get("metrics_json") or {})
        doc["_job"] = {
            "job_id": job_id, "video": video.name, "status": status,
            "error_message": job.get("error_message"),
            "processing_time_seconds": job.get("processing_time_seconds"),
            "total_frames": job.get("total_frames"), "good_frames": job.get("good_frames"),
            "bad_frames": job.get("bad_frames"), "openrouter": answered, "wall_seconds": elapsed,
        }
        out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {"video": video.name, "job_id": job_id, "status": status,
                "error_message": job.get("error_message"),
                "summary_fields": len(doc.get("summary") or {}),
                "frames_missing": doc.get("ocr_frames_missing"), "batches_failed": doc.get("ocr_batches_failed"),
                "answers": answered, "problems": problems[:6], "wall_seconds": elapsed, "output": str(out_json)}

    def shelve_failed(self, video: Path, tag: str) -> None:
        d = self.out / video.stem
        for name, new in ((f"{video.stem}.json", f"{video.stem}.{tag}.json"), ("worker.log", f"worker.{tag}.log")):
            if (d / name).exists():
                (d / name).rename(d / new)


def blocked_by_403(row: dict) -> bool:
    return row.get("status") == "failed" and any("API error 403" in p for p in row.get("problems", []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of videos (or one video)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", default=os.environ.get("ANALYZER_URL", "http://localhost:8010"))
    ap.add_argument("--user", default=os.environ.get("ANALYZER_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("ANALYZER_PASSWORD"),
                    help="or ANALYZER_PASSWORD in the environment")
    ap.add_argument("--campaign-name", required=True)
    ap.add_argument("--company", required=True)
    ap.add_argument("--campaign-date", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    ap.add_argument("--product-name", default=None, help="default: the video's file name without extension")
    ap.add_argument("--poll-seconds", type=int, default=10)
    ap.add_argument("--timeout-minutes", type=int, default=30, help="per job")
    ap.add_argument("--trust-env", action="store_true",
                    help="honour proxy environment variables (off by default: VPN proxies 502 requests to localhost)")
    ap.add_argument("--worker-container", default=None,
                    help="docker container of the Celery worker, e.g. ocr-worker-1: attaches its log lines per job")
    ap.add_argument("--retry-403", type=int, default=0, metavar="N",
                    help="resubmit a job that failed on HTTP 403 up to N times (needs --worker-container)")
    a = ap.parse_args(argv)
    if not a.password:
        ap.error("--password or ANALYZER_PASSWORD is required")
    if a.retry_403 and not a.worker_container:
        ap.error("--retry-403 needs --worker-container")

    src = a.input
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"no videos in {src}")
    sub = Submitter(a)
    print(f"{len(videos)} video(s) -> {a.out}", flush=True)
    report = []
    for n, v in enumerate(videos, 1):
        print(f"[{n}/{len(videos)}] {v.name}", flush=True)
        for attempt in range(1, a.retry_403 + 2):
            if a.worker_container and a.retry_403:
                wait_for_route(a.worker_container)
            row = sub.run_one(v)
            if not blocked_by_403(row) or attempt > a.retry_403:
                break
            print(f"  attempt {attempt}: job failed on 403s; shelving and retrying", flush=True)
            sub.shelve_failed(v, f"failed-403-attempt{attempt}")
            row["retried_after_403"] = attempt
            if route_ok(a.worker_container):
                restart_worker(a.worker_container)
        report.append(row)
        print(f"  -> {row['status']}  fields={row.get('summary_fields')}  missing={row.get('frames_missing')}"
              f"  {row.get('wall_seconds')}s", flush=True)
        for ans in row.get("answers", []):
            print("     ", ans, flush=True)
        for p in row.get("problems", []):
            print("      !", p[-160:], flush=True)
        (a.out / "_run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(r["status"] == "completed" for r in report)
    print(f"DONE completed={ok} of {len(report)}", flush=True)
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
