def test_app_routes_register():
    """If the app imports cleanly and the expected paths are registered,
    most wiring (settings, error handler, templates) is also healthy."""
    from main import app

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    expected = {
        "/", "/login", "/logout", "/upload", "/jobs", "/export", "/health",
        "/status/{job_id}",
        "/api/job/{job_id}",
        "/api/job/{job_id}/metrics",
        "/api/job/{job_id}/metrics/download",
    }
    missing = expected - paths
    assert not missing, f"Missing routes: {missing}"


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "timestamp" in body


def test_login_required_redirects():
    from fastapi.testclient import TestClient

    from main import app

    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
