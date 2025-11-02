"""Tests for the CLI entry point."""

from __future__ import annotations

import sys

from pathlib import Path

from premarket import __main__ as cli


def _write_dummy_config(tmp_path):
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("{}", encoding="utf-8")
    return config_path


def test_main_trims_commented_output_dir_env(monkeypatch, tmp_path):
    captured: dict[str, Path] = {}

    def fake_run(params):
        captured["output_dir"] = params.output_base_dir
        return 0

    monkeypatch.setenv(
        "PREMARKET_OUT_DIR",
        'data/watchlists"      # auto-appends YYYY-MM-DD',
    )
    monkeypatch.setattr(cli.orchestrate, "run", fake_run)

    # ensure the config path resolves without touching the real filesystem
    config_path = _write_dummy_config(tmp_path)
    monkeypatch.setenv("PREMARKET_CONFIG_PATH", str(config_path))

    exit_code = cli.main([])

    assert exit_code == 0
    assert captured["output_dir"] == Path("data/watchlists")

def test_root_main_run_delegates_to_cli(monkeypatch):
    from premarket import __main__ as cli
    import importlib

    captured: dict[str, object] = {}

    def fake_main(args=None):
        captured["argv"] = args
        return 42

    monkeypatch.setattr(cli, "main", fake_main)

    # reload the convenience script to ensure it picks up the patched CLI
    if "premarket_script" in sys.modules:
        del sys.modules["premarket_script"]
    main_script = importlib.import_module("premarket_script")

    exit_code = main_script.run(["--demo"])

    assert exit_code == 42
    assert captured["argv"] == ["--demo"]


def test_main_schedule_delegates_to_scheduler(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_parse(spec):
        captured["spec"] = spec
        return ["parsed"]

    def fake_run_schedule(params, times, timezone=None, runs=None):
        captured["params"] = params
        captured["times"] = times
        captured["timezone"] = timezone
        captured["runs"] = runs
        return 17

    monkeypatch.setattr(cli.scheduler, "parse_schedule", fake_parse)
    monkeypatch.setattr(cli.scheduler, "run_schedule", fake_run_schedule)

    config_path = _write_dummy_config(tmp_path)
    monkeypatch.setenv("PREMARKET_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PREMARKET_SCHEDULE", "07:45")
    monkeypatch.setenv("PREMARKET_SCHEDULE_RUNS", "3")

    exit_code = cli.main([])

    assert exit_code == 17
    assert captured["spec"] == "07:45"
    assert captured["times"] == ["parsed"]
    assert captured["timezone"] == cli.utils.DEFAULT_TZ_NAME
    assert captured["runs"] == 3

