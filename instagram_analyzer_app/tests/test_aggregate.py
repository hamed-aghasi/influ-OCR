def test_aggregator_takes_max_per_metric():
    from processing.gemini_processor import _aggregate

    results = [
        {"metrics": {"likes": 10, "views": 100}},
        {"metrics": {"likes": 25, "views": 105}},
        {"metrics": {"shares": 4}},
        {"metrics": None},
    ]
    summary = _aggregate(results)
    assert summary["likes"] == 25
    assert summary["views"] == 105
    assert summary["shares"] == 4
    assert "follows" not in summary


def test_aggregator_empty_inputs():
    from processing.gemini_processor import _aggregate

    assert _aggregate([]) == {}
    assert _aggregate([{}]) == {}
    assert _aggregate([{"metrics": {}}]) == {}
