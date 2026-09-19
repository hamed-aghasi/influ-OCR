def test_settings_load():
    from processing.config import Settings

    s = Settings()
    assert len(s.secret_key) >= 16
    # Tied to the deployed classifier (run 2026-09-19-relabel), which holds
    # recall 1.000 up to 0.30. Rolling the model back means changing this too.
    assert s.classifier_threshold == 0.30
    assert s.ocr_batch_size == 12
    # Chosen by training/eval_selection.py: loses no ground-truth metric.
    assert s.frame_selection == "scroll"
    assert s.coverage_selection is True
    assert s.broker_url.startswith("redis://")


def test_database_configured_flag():
    from processing.config import Settings

    s = Settings(database_url="postgresql://x")
    assert s.database_configured is True

    s2 = Settings(database_url=None)
    assert s2.database_configured is False
