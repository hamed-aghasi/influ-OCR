"""Which frames go to OCR: the one sequence of selection stages that production
(tasks.run_job) and every review/eval tool call, so they cannot drift apart.

    good frames -> frame selection (settings.frame_selection)
                -> metric coverage via Tesseract (settings.coverage_selection)
                -> drop frames whose content is already in another kept frame
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .config import settings
from .frame_selection import Selection, drop_duplicate_content, select_frames
from .metric_coverage import Coverage, select_by_coverage


@dataclass
class OcrChoice:
    final: List[Path]                               # what is sent to OCR, in time order
    selection: Selection                            # frame selection's output
    coverage: Optional[Coverage] = None             # None when coverage is off or had nothing to do
    dropped_as_duplicate: List[Path] = field(default_factory=list)

    @property
    def not_sent(self) -> List[Path]:
        sent = set(self.final)
        return [p for p in list(self.selection.duplicates) + list(self.selection.selected) if p not in sent]


def choose_ocr_frames(
    sequences: Sequence[Sequence[Path]],
    good: Sequence[Path],
    scores: Optional[Mapping[Path, float]] = None,
    job_id: Optional[str] = None,
) -> OcrChoice:
    selection = select_frames(sequences, good, scores)
    final = list(selection.selected)
    coverage = None
    if settings.coverage_selection and final:
        coverage = select_by_coverage(final, job_id=job_id)
        final = list(coverage.selected)
    dropped: List[Path] = []
    if selection.strategy == "scroll" and len(final) > 1:
        rank = {}
        if coverage is not None and coverage.applied:
            rank = {p: len(coverage.metrics.get(p, ())) + 0.01 * len(coverage.values.get(p, ())) for p in final}
        final, dropped = drop_duplicate_content(selection, final, rank)
        if coverage is not None:
            for p in dropped:
                coverage.reasons[p] = "duplicate"
    return OcrChoice(final=final, selection=selection, coverage=coverage, dropped_as_duplicate=dropped)
