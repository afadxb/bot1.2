"""Tests for utility helpers."""

from __future__ import annotations

from premarket import utils


def test_safe_int_handles_finviz_suffixes():
    assert utils.safe_int("850K") == 850_000
    assert utils.safe_int("1.4M") == 1_400_000
    assert utils.safe_int("2.5B") == 2_500_000_000


def test_safe_int_handles_percent_strings():
    assert utils.safe_int("65%") == 65
    assert utils.safe_int(" 99.73% ") == 99


def test_env_str_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    assert utils.env_str("SOME_KEY", "fallback") == "fallback"


def test_env_str_ignores_blank_and_comment_only_values(monkeypatch):
    monkeypatch.setenv("EMPTY_VALUE", "   ")
    assert utils.env_str("EMPTY_VALUE", "fallback") == "fallback"

    monkeypatch.setenv("COMMENT_ONLY", "   # just a comment")
    assert utils.env_str("COMMENT_ONLY", "fallback") == "fallback"


def test_env_str_strips_inline_comment_when_separated(monkeypatch):
    monkeypatch.setenv("WITH_COMMENT", "logs/premarket.log  # trailing note")
    assert utils.env_str("WITH_COMMENT") == "logs/premarket.log"


def test_env_str_preserves_hash_in_value(monkeypatch):
    url = "https://example.com/path/#fragment"
    monkeypatch.setenv("URL_VALUE", url)
    assert utils.env_str("URL_VALUE") == url

def test_env_str_trims_wrapping_quotes(monkeypatch):
    monkeypatch.setenv("QUOTED", '"logs/premarket.log"')
    assert utils.env_str("QUOTED") == "logs/premarket.log"


def test_env_str_discards_dangling_quote_after_comment(monkeypatch):
    monkeypatch.setenv("DANGLING", 'data/watchlists"      # auto-appends')
    assert utils.env_str("DANGLING") == "data/watchlists"

