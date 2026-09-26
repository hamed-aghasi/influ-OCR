"""OCR response parsing: structured-output envelope, truncation, bad payloads."""

import json


def test_parse_valid_envelope():
    from processing.ocr_processor import _parse_content

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
    from processing.ocr_processor import _parse_content

    truncated = '{"frames": [{"frame_index": 0, "metrics": {"views": 1234}}, {"frame_ind'
    assert _parse_content(truncated) is None


def test_parse_rejects_wrong_shape():
    from processing.ocr_processor import _parse_content

    # A bare array IS accepted (see test_parse_accepts_a_bare_array_envelope):
    # the provider drops the envelope even in strict mode. Real schema
    # violations still fail the batch instead of being kept as raw dicts.
    assert _parse_content('{"frames": [{"frame_index": -1}]}') is None
    assert _parse_content("not json at all") is None


def test_parse_accepts_a_bare_array_envelope():
    """Final_Test 2026-09-26: qwen3.8-max-0902 answered a single frame with
    `[{...}]` instead of `{"frames": [...]}` — strict mode is not enforced by
    that provider. The old parser rejected it and the batch was lost."""
    from processing.ocr_processor import _parse_content

    results = _parse_content('[{"frame_index": 0, "metrics": {"views": 4616, "interactions": 19}}]')
    assert results is not None and len(results) == 1
    assert results[0]["metrics"]["views"] == 4616


def test_parse_normalises_noisy_keys():
    """Final_Test 0d12e688: every key came back as `": views"`. pydantic ignored
    them all, the frame read as 28 nulls, and the job completed with nothing."""
    from processing.ocr_processor import _parse_content

    results = _parse_content(
        '{"frames": [{": frame_index": 0, ": metrics": {": views": 21727, ": interactions": 30},'
        ' " metadata ": {": language": "en"}}]}'
    )
    assert results is not None
    assert results[0]["frame_index"] == 0
    assert results[0]["metrics"]["views"] == 21727 and results[0]["metrics"]["interactions"] == 30
    assert results[0]["metadata"]["language"] == "en"


def test_all_null_batch_is_a_failed_answer():
    from processing.ocr_processor import _all_null

    assert _all_null([{"frame_index": 0, "metrics": None}, {"frame_index": 1, "metrics": {"views": None}}])
    assert not _all_null([{"frame_index": 0, "metrics": {"views": None}}, {"frame_index": 1, "metrics": {"views": 3}}])
    assert not _all_null([])   # nothing sent is not the same as nothing read


def test_parse_empty_frames_is_valid():
    from processing.ocr_processor import _parse_content

    assert _parse_content('{"frames": []}') == []


def test_response_format_declares_frames_schema():
    from processing.ocr_processor import RESPONSE_FORMAT

    assert RESPONSE_FORMAT["type"] == "json_schema"
    schema = RESPONSE_FORMAT["json_schema"]["schema"]
    assert "frames" in schema["properties"]


def test_implausible_metric_values_dropped():
    """Observed in production: model hallucinated follows=4949...(100 digits).
    Absurd values must become None, not win the max() aggregation."""
    from processing.ocr_processor import Metrics

    m = Metrics(views=241099, follows=int("49" * 50), likes=-5, shares=3)
    assert m.views == 241099
    assert m.follows is None
    assert m.likes is None
    assert m.shares == 3


def test_percentage_fields_bounded():
    from processing.ocr_processor import Metrics

    m = Metrics(followers=53.5, non_followers=146.5)
    assert m.followers == 53.5
    assert m.non_followers is None
