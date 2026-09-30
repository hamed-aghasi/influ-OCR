"""The Excel export's metric columns are the schema's metric fields, in the
order users have seen since the export was written. Both lists are maintained
by hand in different files; this is the only thing that keeps them aligned."""


EXPORT_ORDER = (
    "views", "accounts_reached", "followers", "non_followers",
    "interactions", "likes", "replies", "shares",
    "profile_visits", "follows", "links_clicks", "sticker_taps",
    "navigation", "forward", "next_story", "back", "exited",
    "profile_activity", "external_link_taps",
    "gender_men", "gender_women",
    "source_feed", "source_reels_tab", "source_profile", "source_explore",
    "source_stories", "source_search", "source_other",
)


def test_export_columns_are_exactly_the_schema_metric_fields():
    from processing.db_client.export import _METRIC_KEYS
    from processing.ocr_processor import _METRIC_FIELDS

    assert len(_METRIC_KEYS) == len(set(_METRIC_KEYS)), "duplicate export column"
    assert set(_METRIC_KEYS) == set(_METRIC_FIELDS)


def test_export_column_order_is_pinned():
    from processing.db_client.export import _METRIC_KEYS

    # changing this changes every spreadsheet users have downloaded so far
    assert tuple(_METRIC_KEYS) == EXPORT_ORDER
