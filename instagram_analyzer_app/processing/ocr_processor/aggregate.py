"""Consensus across frame reads, and the final metrics document."""

from datetime import datetime
from typing import Dict, List, Tuple

from .schema import _METRIC_FIELDS

def _aggregate(results: List[Dict]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Consensus per metric across frames: modal value if any read repeats,
    otherwise the median. Replaces max(), which let a single inflated
    misread win (observed in production: follows read as 3154, true 31).

    The no-agreement median is median_LOW, not median_high: with exactly two
    disagreeing reads — the common case, since most metrics survive dedup on
    only one or two screens — median_high returns the larger value and is
    therefore identical to the max() this was meant to replace. Misreads
    inflate (digit repetition, see _MAX_PLAUSIBLE_VALUE) far more often than
    they deflate, so ties break downward. For 3+ reads both pick the true
    middle element and the choice is immaterial.

    Returns (summary, quality) where quality records reads/min/max and a
    disputed flag for metrics whose reads disagree materially.
    """
    from collections import Counter
    from statistics import median_low

    summary: Dict[str, float] = {}
    quality: Dict[str, Dict] = {}
    for metric in _METRIC_FIELDS:
        values = []
        for r in results:
            metrics = r.get("metrics") or {}
            v = metrics.get(metric)
            if v is not None:
                values.append(v)
        if not values:
            continue
        top, freq = Counter(values).most_common(1)[0]
        chosen = top if freq > 1 else median_low(values)
        lo, hi = min(values), max(values)
        summary[metric] = chosen
        quality[metric] = {
            "reads": len(values),
            "min": lo,
            "max": hi,
            "disputed": (hi - lo) > max(0.05 * abs(hi), 1),
        }
    return summary, quality


def assemble_metrics(
    all_results: List[Dict],
    total_frames: int,
    unique_frames: int,
    duplicate_frames: int,
    batches_total: int,
    batches_failed: int,
) -> Dict:
    summary, summary_quality = _aggregate(all_results)
    return {
        "extraction_date": datetime.now().isoformat(),
        "total_frames": total_frames,
        "unique_frames": unique_frames,
        "duplicate_frames": duplicate_frames,
        "ocr_batches_total": batches_total,
        "ocr_batches_failed": batches_failed,
        "all_frames_data": all_results,
        "summary": summary,
        "summary_quality": summary_quality,
    }
