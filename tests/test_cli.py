from premarket import __main__ as cli


def test_main_invokes_service(monkeypatch):
    called = {}

    def fake_scan():
        called["ran"] = True

    monkeypatch.setattr(cli, "premarket_scan", fake_scan)

    cli.main()

    assert called["ran"] is True
