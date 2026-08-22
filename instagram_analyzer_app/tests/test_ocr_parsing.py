"""OCR response parsing: structured-output envelope, truncation, bad payloads."""

import json


def test_parse_valid_envelope():
    from processing.gemini_processor import _parse_content

    payload = json.dumps(
        {
            "frames": [
                {"frame_index": 0, "metrics": {"views": 1234, "likes": 30}},
                {"frame_index": 1, "metrics": {"shares": 4}, "metadata": {"language": "fa"}},
            ]
        }
    )
    results = _parse_content(payload)
    assert results is not None
    assert len(results) == 2
    assert results[0]["metrics"]["views"] == 1234
    assert results[1]["metadata"]["language"] == "fa"


def test_parse_truncated_json_fails_whole_batch():
    """A response cut off by max_tokens must return None (batch retry),
    never a partial list that silently undercounts."""
    from processing.gemini_processor import _parse_content

    truncated = '{"frames": [{"frame_index": 0, "metrics": {"views": 1234}}, {"frame_ind'
    assert _parse_content(truncated) is None


def test_parse_rejects_wrong_shape():
    from processing.gemini_processor import _parse_content

    # Old-style bare array (pre-structured-outputs shape) is no longer accepted.
    assert _parse_content('[{"frame_index": 0}]') is None
    # Schema violations fail the batch instead of being kept as raw dicts.
    assert _parse_content('{"frames": [{"frame_index": -1}]}') is None
    assert _parse_content("not json at all") is None


def test_parse_empty_frames_is_valid():
    from processing.gemini_processor import _parse_content

    assert _parse_content('{"frames": []}') == []


def test_response_format_declares_frames_schema():
    from processing.gemini_processor import RESPONSE_FORMAT

    assert RESPONSE_FORMAT["type"] == "json_schema"
    schema = RESPONSE_FORMAT["json_schema"]["schema"]
    assert "frames" in schema["properties"]


def test_implausible_metric_values_dropped():
    """Observed in production: model hallucinated follows=4949...(100 digits).
    Absurd values must become None, not win the max() aggregation."""
    from processing.gemini_processor import Metrics

    m = Metrics(views=241099, follows=int("49" * 50), likes=-5, shares=3)
    assert m.views == 241099
    assert m.follows is None
    assert m.likes is None
    assert m.shares == 3


def test_percentage_fields_bounded():
    from processing.gemini_processor import Metrics

    m = Metrics(followers=53.5, non_followers=146.5)
    assert m.followers == 53.5
    assert m.non_followers is None
