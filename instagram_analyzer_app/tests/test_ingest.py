"""MinIO ingest: key-convention parsing and claim/settle/reject moves."""


class _FakePaginator:
    def __init__(self, store):
        self.store = store

    def paginate(self, Bucket=None, Prefix=""):
        yield {
            "Contents": [
                {"Key": k, "Size": len(v)}
                for k, v in sorted(self.store.items())
                if k.startswith(Prefix)
            ]
        }


class FakeS3:
    def __init__(self, objects):
        self.store = dict(objects)

    def copy_object(self, Bucket=None, Key=None, CopySource=None):
        self.store[Key] = self.store[CopySource["Key"]]

    def delete_object(self, Bucket=None, Key=None):
        self.store.pop(Key, None)

    def get_paginator(self, name):
        return _FakePaginator(self.store)

    def download_file(self, bucket, key, dest):
        with open(dest, "wb") as f:
            f.write(self.store[key])


def test_parse_object_key_valid():
    from processing.ingest import parse_object_key

    meta = parse_object_key("incoming/nike/summer-sale/airmax/story1.mp4")
    assert meta == {
        "company": "nike",
        "campaign_name": "summer-sale",
        "product_name": "airmax",
        "filename": "story1.mp4",
        "ext": ".mp4",
        "file_type": "video",
    }
    assert parse_object_key("incoming/co/camp/prod/shots.zip")["file_type"] == "zip"
    assert parse_object_key("incoming/co/camp/prod/frame.PNG")["file_type"] == "image"


def test_parse_object_key_rejects_bad_shapes():
    from processing.ingest import parse_object_key

    assert parse_object_key("incoming/nike/summer/story1.mp4") is None  # too shallow
    assert parse_object_key("incoming/a/b/c/d/story1.mp4") is None  # too deep
    assert parse_object_key("incoming/nike/summer/airmax/notes.txt") is None  # bad ext
    assert parse_object_key("processed/nike/summer/airmax/story1.mp4") is None  # wrong prefix
    assert parse_object_key("incoming/nike//airmax/story1.mp4") is None  # empty segment


def test_list_incoming_filters_prefix_and_folders():
    from processing.ingest import list_incoming

    fake = FakeS3(
        {
            "incoming/a/b/c/f.mp4": b"xx",
            "incoming/a/b/c/": b"",  # folder marker
            "processed/a/b/c/old.mp4": b"yy",
        }
    )
    objs = list_incoming(client=fake)
    assert [o["key"] for o in objs] == ["incoming/a/b/c/f.mp4"]


def test_claim_settle_and_reject_moves():
    from processing.ingest import claim_object, reject_object, settle_object

    fake = FakeS3({"incoming/a/b/c/f.mp4": b"data", "incoming/a/b/c/bad.txt": b"junk"})

    claimed = claim_object("incoming/a/b/c/f.mp4", client=fake)
    assert claimed == "ingesting/a/b/c/f.mp4"
    assert "incoming/a/b/c/f.mp4" not in fake.store

    settle_object(claimed, success=True, client=fake)
    assert "processed/a/b/c/f.mp4" in fake.store
    assert claimed not in fake.store

    reject_object("incoming/a/b/c/bad.txt", client=fake)
    assert "failed/a/b/c/bad.txt" in fake.store
    assert "incoming/a/b/c/bad.txt" not in fake.store


def test_settle_failure_goes_to_failed_prefix():
    from processing.ingest import settle_object

    fake = FakeS3({"ingesting/a/b/c/f.mp4": b"data"})
    settle_object("ingesting/a/b/c/f.mp4", success=False, client=fake)
    assert "failed/a/b/c/f.mp4" in fake.store
    assert "ingesting/a/b/c/f.mp4" not in fake.store
