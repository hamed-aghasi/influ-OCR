"""Metric-coverage selection. Detection is tested on Tesseract output captured
from real ground-truth frames; selection with an injected detector, so none of
this needs the tesseract binary."""

from pathlib import Path


# ----- detection, on real OCR output -----

REEL_DARK_EN = [  # 08165dca frame 060
    "19:06 @) Sen oN to!", "< ___—Reelinsights nw:", "oo]", "7] Q @ Vv A", "186 0 1 3 1",
    "Overview Engagement Audience", "Summary ©", "Views Accounts reached", "20,532 14,472",
    "Average watch time Follows", "4s 0", "What affects your views ©", "< O Ill",
]
STORY_EN_LIST = [  # 57e39f36 frame 390
    "12:53 440° oo / Full & | ==", "Likes 10", "Replies 0", "Shares 0", "Sticker taps 2",
    "Navigation 14,207", "Forward 5,500", "Next story 4,836", "Exited 2,036", "Back 1835",
]
STORY_FA = [  # 7cb649a2 frame 105
    "اشتراک‌گذاری‌ها 7", "پیمایش 68,408", "بازارسال 54,169", "برگشت 4,808", "داستان بعدی 2,818",
    "فعالیت نمایه © 535", "بازدیدهای تمایه 535",
]
INTERACTIONS_SPLIT_HEADER_OFFSCREEN = [  # 1caae563 frame 333
    "19:31 8 ۰ Goal wat", "lil = 18.3K wi", "@ Followers 100.0%", "® Non-followers 0.0%",
    "Likes 45", "Replies 3", "Shares 3", "Navigation 20,549",
]
VIEWS_SPLIT = [  # 1caae563 frame 321
    "22,273", "* Followers 99.7%", "® Non-followers 0.3%", "Accounts reached 16,838", "Interactions ©",
]
TILE_WITHOUT_FOLLOWS_VALUE = [  # c70ca2f3 frame 066: Follows shows "--"
    "213 9 4 142 7", "Overview Engagement Audience", "Views Accounts reached", "21,756 15,735",
    "Average watch time Follows", "4s =",
]
RATES_ONLY = [  # 08165dca frame 090: status bar sits two lines above the tabs
    "19:06 78 Sen om ts 0", "+ 86۵۱ هم ونوا", "Overview Engagement Audience",
    "W Save rate 0.0%", "g Repost rate -", "Q Comment rate 0.0%", "How long people watched your reel ©",
]


def test_reel_summary_and_icon_strip():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(REEL_DARK_EN) == {
        "likes", "replies", "shares", "views", "accounts_reached", "follows"}


def test_story_list_english_and_persian():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(STORY_EN_LIST) == {
        "likes", "replies", "shares", "sticker_taps", "navigation", "forward", "next_story", "exited", "back"}
    # OCR read "نمایه" as "تمایه" in the last line; profile_activity still counts
    assert metrics_in_text(STORY_FA) >= {"shares", "navigation", "forward", "back", "next_story",
                                         "profile_activity"}


def test_profile_visits_is_not_read_as_views():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(["Profile visits 20"]) == {"profile_visits"}
    assert metrics_in_text(["بازدیدهای نمایه 535"]) == {"profile_visits"}


def test_interactions_split_is_not_the_followers_metric():
    from processing.metric_coverage import metrics_in_text

    found = metrics_in_text(INTERACTIONS_SPLIT_HEADER_OFFSCREEN)
    assert "followers" not in found and "non_followers" not in found
    assert {"likes", "replies", "shares", "navigation"} <= found
    assert {"followers", "non_followers", "accounts_reached"} <= metrics_in_text(VIEWS_SPLIT)


def test_value_line_must_have_a_value_per_label():
    from processing.metric_coverage import metrics_in_text

    found = metrics_in_text(TILE_WITHOUT_FOLLOWS_VALUE)
    assert "follows" not in found                    # "4s =" is the watch time, not Follows
    assert {"views", "accounts_reached"} <= found


def test_status_bar_is_not_the_icon_strip_and_rates_are_not_metrics():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(RATES_ONLY) == set()


# siah tulani, test-frame-output-v2: the Views-over-time filter buttons, then
# whatever the chart puts on the next line. Exact Tesseract output, per frame.
FILTER_ROW_CASES = {
    168: ["All Followers Non-followers", "200K eR ETT"],
    237: ["All Followers Non-followers ۰", "2۳0 \u200fبت بت بت بت بت بت\u200e"],
    264: ["All Followers Nor-followers", "24.4"],
    300: ["All Followers Non-followers", "29 ۳"],
}


def test_filter_buttons_are_not_the_followers_split():
    from processing.metric_coverage import metrics_in_text, values_in

    for frame, lines in FILTER_ROW_CASES.items():
        assert metrics_in_text(lines) == set(), frame
        assert values_in(lines) == set(), frame      # the axis / tooltip is not a guarded value
    persian = ["همه دنبال‌کنندگان غیردنبال‌کنندگان", "۵۰ هزار"]
    assert metrics_in_text(persian) == set() and values_in(persian) == set()


def test_the_real_split_still_counts_next_to_the_filter_row():
    from processing.metric_coverage import metrics_in_text

    lines = ["Views over time ©", "All Followers Non-followers", "200K",
             "* Followers 99.7%", "® Non-followers 0.3%", "Accounts reached 16,838"]
    assert metrics_in_text(lines) == {"followers", "non_followers", "accounts_reached"}


def test_chart_headers_are_not_values():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(["Views over time ©", "30K", "15K"]) == set()


def test_values_backstop_ignores_rates_age_and_country():
    from processing.metric_coverage import values_in

    assert values_in(["Skip rate 68.0%", "19:31 5", "Age 18-24 36.6%", "Country Iran 97.1%"]) == set()
    assert values_in(["Views Accounts reached", "20,532 14,472"]) == {"20532", "14472"}
    # the gender split is a schema metric now, so its values are guarded
    assert values_in(["Men 70.4%", "Women 29.6%"]) == {"70.4%", "29.6%"}


# 4eb4b131 frame 084, exact Tesseract output: label over its bar, value on the
# next line after bar glyphs, and "4.1%" misread as "41%" (fine — only the
# vision model reads values).
GENDER_STORY = ["Profile visits 6", "Follows 0", "Audience ©", "Gender Country Age",
                "Men", "\u200fاه\u200e 95.9%", "Women", "= 41%"]


def test_gender_split_is_detected():
    from processing.metric_coverage import metrics_in_text

    assert metrics_in_text(GENDER_STORY) == {"profile_visits", "follows", "gender_men", "gender_women"}
    assert metrics_in_text(["Men 95.9%", "Women 4.1%"]) == {"gender_men", "gender_women"}
    assert metrics_in_text(["مردان ۷۰٫۴٪", "زنان ۲۹٫۶٪"]) == {"gender_men", "gender_women"}


def test_gender_labels_do_not_match_inside_other_words():
    from processing.metric_coverage import metrics_in_text

    # "men" is inside "comments" / "engagement"; Persian "زن" inside "بازنشرها" (reposts)
    assert metrics_in_text(["Comments 0", "Overview Engagement Audience", "بازنشرها 1"]) == {"replies"}
    assert metrics_in_text(["Men", "Women"]) == set()          # label with no value
    assert metrics_in_text(["Men", "Profile visits 6"]) == {"profile_visits"}  # value belongs to the next label


# ----- selection, with an injected detector -----

def _frames(n):
    return [Path(f"frame_{i * 3:06d}.jpg") for i in range(n)]


def test_set_cover_keeps_the_fewest_frames():
    from processing.metric_coverage import select_by_coverage

    f = _frames(5)
    table = {
        f[0]: ({"likes", "views"}, set(), True),
        f[1]: ({"likes", "views", "follows"}, set(), True),
        f[2]: ({"views"}, set(), True),
        f[3]: ({"navigation"}, set(), True),
        f[4]: ({"navigation", "back"}, set(), True),
    }
    cov = select_by_coverage(f, detect=table.__getitem__)
    assert cov.selected == [f[1], f[4]]
    assert cov.dropped == [f[0], f[2], f[3]]
    assert [cov.reasons[p] for p in f] == ["covered", "cover", "covered", "covered", "cover"]


def test_unrecognised_frame_with_a_number_is_kept_and_blank_one_dropped():
    from processing.metric_coverage import select_by_coverage

    f = _frames(3)
    table = {f[0]: ({"views"}, set(), True), f[1]: (set(), set(), True), f[2]: (set(), set(), False)}
    cov = select_by_coverage(f, detect=table.__getitem__)
    assert cov.selected == [f[0], f[1]]
    assert [cov.reasons[p] for p in f] == ["cover", "unrecognised", "blank"]


def test_value_backstop_keeps_a_frame_whose_number_is_nowhere_else():
    """Label detection says frame 1 adds nothing, but it shows a labelled value
    (a mis-paired read) that no kept frame shows: it must stay."""
    from processing.metric_coverage import select_by_coverage

    f = _frames(2)
    table = {f[0]: ({"views", "likes"}, {"20532"}, True), f[1]: ({"views"}, {"20532", "3154"}, True)}
    cov = select_by_coverage(f, detect=table.__getitem__)
    assert cov.selected == f
    assert cov.reasons[f[1]] == "backstop"


def test_a_failing_detector_drops_nothing():
    from processing.metric_coverage import select_by_coverage

    f = _frames(2)

    def boom(p):
        raise RuntimeError("tesseract crashed")

    assert select_by_coverage(f, detect=boom).selected == f


def test_missing_tesseract_leaves_the_selection_alone(monkeypatch):
    from processing import metric_coverage as mc

    monkeypatch.setattr(mc, "tesseract_available", lambda: False)
    f = _frames(3)
    cov = mc.select_by_coverage(f)
    assert cov.selected == f and not cov.applied


def test_run_job_applies_coverage_only_when_enabled(tmp_path, monkeypatch):
    import tasks
    from processing.frame_selection import Selection
    from processing.metric_coverage import Coverage

    frames = [tmp_path / f"frame_{i:06d}.jpg" for i in range(3)]
    monkeypatch.setattr(tasks, "update_job_status", lambda *a, **k: True)
    monkeypatch.setattr(tasks, "update_job_progress", lambda *a, **k: True)
    monkeypatch.setattr(tasks.settings, "processing_dir", tmp_path / "processing")
    monkeypatch.setattr(tasks, "extract_frames_from_video", lambda *a, **k: (3, frames))
    monkeypatch.setattr(tasks, "classify_frames", lambda *a, **k: {
        "good_paths": frames, "bad_frames": [], "good_frames": []})
    monkeypatch.setattr(tasks, "select_frames", lambda *a, **k: Selection(selected=frames, duplicates=[]))
    monkeypatch.setattr(tasks, "select_by_coverage",
                        lambda paths, job_id=None: Coverage(selected=paths[:1], dropped=paths[1:]))
    sent = {}
    monkeypatch.setattr(tasks.finalize_job, "delay", lambda *a, **k: None)
    monkeypatch.setattr(tasks, "chunk_batches", lambda paths: sent.setdefault("paths", paths) and [paths])
    monkeypatch.setattr(tasks, "chord", lambda *a, **k: type("C", (), {"delay": lambda self: None})())

    monkeypatch.setattr(tasks.settings, "coverage_selection", False)
    tasks.run_job("j-off", str(tmp_path / "v.mp4"), "video")
    assert sent.pop("paths") == frames

    monkeypatch.setattr(tasks.settings, "coverage_selection", True)
    tasks.run_job("j-on", str(tmp_path / "v.mp4"), "video")
    assert sent.pop("paths") == frames[:1]
