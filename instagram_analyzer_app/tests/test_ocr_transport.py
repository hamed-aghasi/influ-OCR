"""Transport-level behaviour of the OpenRouter call: strict-schema compliance,
malformed envelopes, truncation splitting, and connection reuse."""

import json

import pytest


class _Resp:
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _completion(frames, finish_reason="stop"):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": json.dumps({"frames": frames})},
            }
        ]
    }


# ----- strict schema -----

def test_response_format_is_strict_compliant():
    from processing.ocr_processor import RESPONSE_FORMAT

    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    schema = RESPONSE_FORMAT["json_schema"]["schema"]

    def walk(node, path="root"):
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties") or {}
            assert node.get("additionalProperties") is False, f"{path} is not closed"
            assert set(node.get("required") or []) == set(props), f"{path} required != properties"
        for key, value in node.items():
            assert key not in {"minimum", "maximum", "default", "format"}, f"{path}.{key} unsupported"
            walk(value, f"{path}.{key}")

    walk(schema)


def test_strict_schema_keeps_the_frames_envelope():
    from processing.ocr_processor import RESPONSE_FORMAT

    schema = RESPONSE_FORMAT["json_schema"]["schema"]
    assert "frames" in schema["properties"]


def test_actual_frame_is_not_requested_from_the_model():
    """It is filled in locally; asking for it costs tokens on every frame."""
    from processing.ocr_processor import RESPONSE_FORMAT

    schema = RESPONSE_FORMAT["json_schema"]["schema"]
    frame = schema["$defs"]["FrameResult"]
    assert "actual_frame" not in frame["properties"]
    assert "actual_frame" not in frame["required"]


def test_wire_schema_carries_no_prose():
    """pydantic copies class docstrings into the schema as `description`,
    which then ships to the model on every request."""
    from processing.ocr_processor import RESPONSE_FORMAT

    assert "description" not in json.dumps(RESPONSE_FORMAT["json_schema"]["schema"])


def test_client_side_validation_survives_schema_relaxation():
    """minimum/ge is stripped from the wire schema but must still be enforced."""
    from processing.ocr_processor import _parse_content

    assert _parse_content('{"frames": [{"frame_index": -1}]}') is None


# ----- malformed envelopes -----

@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "provider returned an error", "code": 502}},  # 200 + error
        {"choices": []},                                                   # empty choices
        {},                                                                # no choices key
        None,                                                              # body isn't JSON
    ],
    ids=["provider-error", "empty-choices", "no-choices", "not-json"],
)
def test_malformed_envelope_is_retried_not_raised(payload, monkeypatch):
    from processing import ocr_processor as gp

    monkeypatch.setattr(gp.settings, "openrouter_api_key", "k")
    monkeypatch.setattr(gp.settings, "ocr_max_retries", 2)
    monkeypatch.setattr(gp.time, "sleep", lambda *_: None)

    calls = []

    class _Session:
        def post(self, *a, **k):
            calls.append(1)
            return _Resp(payload, text="upstream exploded")

    monkeypatch.setattr(gp, "_session", lambda: _Session())

    result = gp._call_api([("a.jpg", "b64")], gp.job_logger(__name__))

    assert result is None          # failed, but as a return value
    assert len(calls) == 2         # and it used its retries instead of raising


def test_valid_envelope_returns_frames(monkeypatch):
    from processing import ocr_processor as gp

    monkeypatch.setattr(gp.settings, "openrouter_api_key", "k")
    monkeypatch.setattr(gp.time, "sleep", lambda *_: None)

    class _Session:
        def post(self, *a, **k):
            return _Resp(_completion([{"frame_index": 0, "metrics": {"views": 12}}]))

    monkeypatch.setattr(gp, "_session", lambda: _Session())

    result = gp._call_api([("a.jpg", "b64")], gp.job_logger(__name__))
    assert result is not None
    assert result[0]["metrics"]["views"] == 12


# ----- truncation splitting -----

def test_truncation_splits_the_batch_instead_of_resending(monkeypatch):
    from processing import ocr_processor as gp

    monkeypatch.setattr(gp.settings, "openrouter_api_key", "k")
    monkeypatch.setattr(gp.settings, "ocr_max_retries", 5)
    monkeypatch.setattr(gp.time, "sleep", lambda *_: None)

    seen_sizes = []

    class _Session:
        def post(self, *a, **k):
            n = sum(1 for part in k["json"]["messages"][0]["content"] if part["type"] == "image_url")
            seen_sizes.append(n)
            if n > 2:
                return _Resp(_completion([], finish_reason="length"))
            return _Resp(_completion([{"frame_index": i, "metrics": {"views": 10 + i}} for i in range(n)]))

    monkeypatch.setattr(gp, "_session", lambda: _Session())

    encoded = [(f"{i}.jpg", "b64") for i in range(4)]
    result = gp._call_api(encoded, gp.job_logger(__name__))

    assert seen_sizes == [4, 2, 2]  # full batch truncated, then two halves
    assert result is not None
    # second half's indices are rebased onto the original batch
    assert [r["frame_index"] for r in result] == [0, 1, 2, 3]


def test_truncation_on_a_single_frame_gives_up(monkeypatch):
    from processing import ocr_processor as gp

    monkeypatch.setattr(gp.settings, "openrouter_api_key", "k")
    monkeypatch.setattr(gp.time, "sleep", lambda *_: None)

    calls = []

    class _Session:
        def post(self, *a, **k):
            calls.append(1)
            return _Resp(_completion([], finish_reason="length"))

    monkeypatch.setattr(gp, "_session", lambda: _Session())

    assert gp._call_api([("only.jpg", "b64")], gp.job_logger(__name__)) is None
    assert len(calls) == 1  # no point retrying an unsplittable truncation


# ----- connection reuse -----

def test_session_is_reused_within_a_thread():
    from processing.ocr_processor import _session

    assert _session() is _session()


def test_each_thread_gets_its_own_session():
    import threading

    from processing.ocr_processor import _session

    seen = []
    main = _session()
    t = threading.Thread(target=lambda: seen.append(_session()))
    t.start()
    t.join()

    assert seen[0] is not main
