"""The fewest frames that together show every metric: set cover plus safety nets."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from ..config import settings
from ..logger import job_logger
from .detect import evidence_values, metric_value_pairs, metrics_in_text, values_in
from .tesseract import read_lines, tesseract_available
from .text import _is_guardable

@dataclass
class Coverage:
    selected: List[Path]
    dropped: List[Path]
    metrics: Dict[Path, Set[str]] = field(default_factory=dict)
    applied: bool = True
    values: Dict[Path, Set[str]] = field(default_factory=dict)   # labelled values read per frame
    reason: str = ""
    # per frame: "cover" (needed for a metric), "backstop" (shows a labelled
    # value no other kept frame shows), "unrecognised" (no metric read, but a
    # value no other kept frame shows), "covered" (dropped: its metrics are in
    # kept frames), "blank" (dropped: nothing it shows is new)
    reasons: Dict[Path, str] = field(default_factory=dict)


def select_by_coverage(frames: Sequence[Path], job_id: Optional[str] = None,
                       detect=None) -> Coverage:
    """Keep the fewest `frames` that show every metric any of them shows.

    `frames` should already be in time order. `detect(path)` returns a set of
    metrics, or (metrics, values); injectable for tests. By default it OCRs
    the frame.
    """
    log = job_logger("processing.metric_coverage", job_id)
    frames = list(frames)
    if detect is None:
        if not tesseract_available():
            log.warning("Tesseract (%s) unavailable; metric-coverage selection skipped", settings.coverage_ocr_lang)
            return Coverage(frames, [], applied=False, reason="tesseract unavailable")
        def detect(p):
            lines = read_lines(p)
            return (metrics_in_text(lines), values_in(lines),
                    evidence_values(lines), metric_value_pairs(lines))

    metrics: Dict[Path, Set[str]] = {}
    values: Dict[Path, Set[str]] = {}
    evidence: Dict[Path, Set[str]] = {}          # values found without a recognised label
    pairs: Dict[Path, Dict[str, Set[str]]] = {}
    for p in frames:
        try:
            got = detect(p)
            if not isinstance(got, tuple):
                got = (got, set(), set())
            third = got[2] if len(got) > 2 else set()
            # A detector may still answer the old yes/no; True then means
            # "something is on screen but I cannot say what", which must keep
            # the frame rather than silently drop it.
            if isinstance(third, bool):
                third = {f"\x00unreadable:{p}"} if third else set()
            metrics[p], values[p], evidence[p] = set(got[0]), set(got[1]), set(third)
            pairs[p] = got[3] if len(got) > 3 else {}
        except Exception as exc:  # noqa: BLE001 — one unreadable frame must not drop anything
            log.warning("Coverage OCR failed on %s (%s); keeping it", p.name, exc)
            # A value unique to this frame: nothing else can cover it, so a
            # frame we could not read is always kept.
            metrics[p], values[p], evidence[p], pairs[p] = set(), set(), {f"\x00unreadable:{p}"}, {}

    # Cross-frame evidence: when one frame pairs a metric with a value of three or
    # more digits ("Likes 124"), another frame showing that same value in a
    # labelled list shows that metric too, even if OCR garbled its label there
    # (round-01, 8102e03d: "Likes 124" read as "'3 124" made two identical
    # frames both look necessary).
    for a in frames:
        for metric, vals in pairs[a].items():
            for v in vals:
                if len(re.sub(r"\D", "", v)) < 3:
                    continue
                for b in frames:
                    if b is not a and metric not in metrics[b] and v in values[b]:
                        metrics[b].add(metric)

    todo = set().union(*metrics.values()) if metrics else set()
    chosen: List[Path] = []
    order = {p: i for i, p in enumerate(frames)}
    while todo:
        best = max((p for p in frames if p not in chosen and metrics[p]),
                   # ties go to the most complete frame, then the later one: a page
                   # still loading reads fewer values, and loads top to bottom over time
                   key=lambda p: (len(metrics[p] & todo), len(metrics[p]), len(values[p]), order[p]))
        if not metrics[best] & todo:
            break
        chosen.append(best)
        todo -= metrics[best]

    # One rule for everything the set cover did not pick: keep a frame only if
    # it shows a value no kept frame shows. A missed label must not become a
    # dropped value — but "I recognised nothing here" is not by itself a reason
    # to spend an OCR slot (round-02: loading skeletons whose numbers were
    # already on a sent frame, and rates tabs with no counts at all).
    #
    # For a frame whose labels were read, the guarded values are the labelled
    # ones; for a frame with no recognised metric, they are the evidence values
    # (evidence_values above). Percentages and K/M ticks are excluded either
    # way: OCR reads small percentages inconsistently ("1.1%" as "11%"), so
    # guarding them kept a new frame per misread (round-01b: three copies of one
    # sources list), and a single digit identifies nothing.
    keep = set(chosen)
    reasons = {p: "cover" for p in chosen}
    guarded = {p: ({v for v in values[p] if _is_guardable(v)} if metrics[p] else evidence[p])
               for p in frames}
    shown = {p: values[p] | (evidence[p] if not metrics[p] else set()) for p in frames}
    while True:
        seen = set().union(*(shown[p] for p in keep)) if keep else set()
        orphans = [p for p in frames if p not in keep and guarded[p] - seen]
        if not orphans:
            break
        # among frames showing the missing value, the most complete one (round-01:
        # a half-loaded frame was kept over the fully loaded frame_000060)
        extra = max(orphans, key=lambda p: (len(guarded[p] - seen), len(metrics[p]), len(values[p]), order[p]))
        keep.add(extra)
        reasons[extra] = "backstop" if metrics[extra] else "unrecognised"
    for p in frames:
        reasons.setdefault(p, "covered" if metrics[p] else "blank")

    selected = [p for p in frames if p in keep]
    dropped = [p for p in frames if p not in keep]
    unknown_kept = sum(1 for p in selected if reasons.get(p) == "unrecognised")
    log.info("Coverage: %d -> %d frames (%d unrecognised kept), %d metrics",
             len(frames), len(selected), unknown_kept, len(set().union(*metrics.values())) if metrics else 0)
    return Coverage(selected, dropped, metrics, values=values, reasons=reasons)
