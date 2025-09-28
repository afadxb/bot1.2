from premarket import loader_finviz


def test_http_get_falls_back_to_urllib(monkeypatch):
    def fail_with_stub(_url: str) -> str:
        raise loader_finviz.requests.RequestException(
            "Network access disabled in test environment"
        )

    def succeed_with_urllib(url: str) -> str:
        assert url == "https://example.com/export"
        return "ok"

    monkeypatch.setattr(loader_finviz, "_fetch_with_requests", fail_with_stub)
    monkeypatch.setattr(loader_finviz, "_fetch_with_urllib", succeed_with_urllib)

    result = loader_finviz._http_get("https://example.com/export")

    assert result == "ok"


def test_download_csv_overwrites_existing_file(tmp_path, monkeypatch):
    out_path = tmp_path / "raw" / "2024-01-01" / "finviz_elite.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("old", encoding="utf-8")

    def fake_http_get(url: str) -> str:
        assert url == "https://example.com/export"
        return "new"

    monkeypatch.setattr(loader_finviz, "_http_get", fake_http_get)

    result = loader_finviz.download_csv("https://example.com/export", out_path, use_cache=True)

    assert result == out_path
    assert out_path.read_text(encoding="utf-8") == "new"
