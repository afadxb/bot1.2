from premarket import news_ai


def test_score_signal_prefers_recent_news():
    payload = {"freshness_hours": 1.5, "category": "finnhub"}
    stale_payload = {"freshness_hours": 30.0, "category": "finviz"}

    fresh_score = news_ai.score_signal(payload, horizon_hours=24)
    stale_score = news_ai.score_signal(stale_payload, horizon_hours=24)

    assert 0.0 <= stale_score <= 1.0
    assert 0.0 <= fresh_score <= 1.0
    assert fresh_score > stale_score


def test_score_batch_handles_missing_values():
    payloads = {
        "AAA": {"freshness_hours": None, "category": None},
        "BBB": {"freshness_hours": "3", "category": "finnhub"},
    }
    scores = news_ai.score_batch(payloads, horizon_hours=12)
    assert set(scores) == {"AAA", "BBB"}
    assert scores["AAA"] == 0.0
    assert scores["BBB"] > 0.0


def test_summarize_scores_reports_leader():
    scores = {"AAA": 0.9, "BBB": 0.4, "CCC": 0.7}
    summary = news_ai.summarize_scores(scores, strong_threshold=0.6)

    expected_avg = round(sum(scores.values()) / len(scores), 3)
    assert summary["average"] == expected_avg
    assert summary["leader"]["symbol"] == "AAA"
    assert set(summary["strong_symbols"]) == {"AAA", "CCC"}
    assert summary["top_signals"][0]["symbol"] == "AAA"
