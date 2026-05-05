def test_settings_load():
    from processing.config import Settings

    s = Settings()
    assert len(s.secret_key) >= 16
    assert s.classifier_threshold == 0.65
    assert s.ocr_batch_size == 50
    assert s.broker_url.startswith("redis://")


def test_database_configured_flag():
    from processing.config import Settings

    s = Settings(database_url="postgresql://x")
    assert s.database_configured is True

    s2 = Settings(database_url=None)
    assert s2.database_configured is False
