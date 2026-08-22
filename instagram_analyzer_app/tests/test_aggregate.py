def test_aggregator_consensus_per_metric():
    from processing.gemini_processor import _aggregate

    results = [
        {"metrics": {"likes": 10, "views": 100}},
        {"metrics": {"likes": 25, "views": 105}},
        {"metrics": {"shares": 4}},
        {"metrics": None},
    ]
    summary, quality = _aggregate(results)
    # Two distinct reads -> median_high (equals the old max() behavior here).
    assert summary["likes"] == 25
    assert summary["views"] == 105
    assert summary["shares"] == 4
    assert "follows" not in summary
    assert quality["shares"]["reads"] == 1


def test_aggregator_outlier_outvoted():
    """Production case: follows read as 31 on two frames and 3154 on one.
    max() crowned 3154; consensus must pick 31 and flag the dispute."""
    from processing.gemini_processor import _aggregate

    results = [{"metrics": {"follows": v}} for v in (31, 31, 3154)]
    summary, quality = _aggregate(results)
    assert summary["follows"] == 31
    assert quality["follows"]["disputed"] is True
    assert quality["follows"]["max"] == 3154


def test_aggregator_median_when_all_distinct():
    from processing.gemini_processor import _aggregate

    results = [{"metrics": {"views": v}} for v in (100, 90, 4000)]
    summary, quality = _aggregate(results)
    assert summary["views"] == 100  # median, not the 4000 outlier
    assert quality["views"]["disputed"] is True


def test_aggregator_consistent_reads_not_disputed():
    from processing.gemini_processor import _aggregate

    results = [{"metrics": {"views": v}} for v in (100, 100, 101)]
    summary, quality = _aggregate(results)
    assert summary["views"] == 100
    assert quality["views"]["disputed"] is False


def test_aggregator_empty_inputs():
    from processing.gemini_processor import _aggregate

    assert _aggregate([]) == ({}, {})
    assert _aggregate([{}]) == ({}, {})
    assert _aggregate([{"metrics": {}}]) == ({}, {})
