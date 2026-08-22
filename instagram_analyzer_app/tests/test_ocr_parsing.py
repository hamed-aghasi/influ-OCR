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
