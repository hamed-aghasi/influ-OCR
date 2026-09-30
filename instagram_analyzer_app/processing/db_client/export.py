"""Excel export of every job's summary metrics."""

import io
from pathlib import Path
from typing import Optional

from ..errors import APIError
from .jobs import get_all_jobs

_METRIC_KEYS = (
    "views",
    "accounts_reached",
    "followers",
    "non_followers",
    "interactions",
    "likes",
    "replies",
    "shares",
    "profile_visits",
    "follows",
    "links_clicks",
    "sticker_taps",
    "navigation",
    "forward",
    "next_story",
    "back",
    "exited",
    "profile_activity",
    "external_link_taps",
    "gender_men",
    "gender_women",
    "source_feed",
    "source_reels_tab",
    "source_profile",
    "source_explore",
    "source_stories",
    "source_search",
    "source_other",
)


def export_to_excel(output_path: Optional[Path] = None) -> Optional[bytes]:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    except ImportError as exc:
        raise APIError(500, "openpyxl not installed") from exc

    jobs = get_all_jobs(limit=10000)
    if not jobs:
        return None

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Campaign Results"

    headers = [
        "Job ID", "Campaign Date", "Campaign Name", "Product Name", "Company",
        "Filename", "File Type", "Status", "Created At", "Completed At",
        "Total Frames", "Good Frames", "Bad Frames", "Processing Time (s)",
    ] + [k.replace("_", " ").title() for k in _METRIC_KEYS]

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    alt_fill = PatternFill(start_color="E9EBF5", end_color="E9EBF5", fill_type="solid")
    border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = border

    for row_idx, job in enumerate(jobs, 2):
        metrics_json = job.get("metrics_json") or {}
        summary = metrics_json.get("summary", {}) if isinstance(metrics_json, dict) else {}

        row = [
            str(job.get("id", "")),
            job["campaign_date"].isoformat() if job.get("campaign_date") else "",
            job.get("campaign_name", ""),
            job.get("product_name", ""),
            job.get("company", ""),
            job.get("filename", ""),
            job.get("file_type", ""),
            job.get("status", ""),
            job["created_at"].isoformat() if job.get("created_at") else "",
            job["completed_at"].isoformat() if job.get("completed_at") else "",
            job.get("total_frames", 0),
            job.get("good_frames", 0),
            job.get("bad_frames", 0),
            job.get("processing_time_seconds", 0),
        ] + [summary.get(k, "") for k in _METRIC_KEYS]

        for col, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col, value=value)
            cell.border = border
            if row_idx % 2 == 0:
                cell.fill = alt_fill

    widths = [36, 12, 20, 20, 20, 25, 10, 12, 20, 20, 12, 12, 12, 15] + [14] * len(_METRIC_KEYS)
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = width
    ws.freeze_panes = "A2"

    if output_path:
        wb.save(output_path)
        return None

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
