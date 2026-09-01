import pytest

from us_open_model.data import is_completed_score
from us_open_model.kalshi import compare_match


@pytest.mark.parametrize("score", ["6-4 6-3", "7-6(5) 4-6 6-2", "6-0 6-0"])
def test_completed_scores(score: str) -> None:
    assert is_completed_score(score)


@pytest.mark.parametrize("score", ["6-4 RET", "W/O", "WO", "DEF", "ABD"])
def test_incomplete_scores(score: str) -> None:
    assert not is_completed_score(score)


def test_kalshi_pairing_and_devig() -> None:
    markets = [
        {
            "event_ticker": "KXATPMATCH-TEST",
            "ticker": "KXATPMATCH-TEST-A",
            "yes_sub_title": "Player One",
            "yes_bid_dollars": "0.58",
            "yes_ask_dollars": "0.62",
            "volume_fp": "100.0",
            "rules_primary": "US Open men's singles match",
        },
        {
            "event_ticker": "KXATPMATCH-TEST",
            "ticker": "KXATPMATCH-TEST-B",
            "yes_sub_title": "Player Two",
            "yes_bid_dollars": "0.38",
            "yes_ask_dollars": "0.42",
            "volume_fp": "80.0",
            "rules_primary": "US Open men's singles match",
        },
    ]
    comparison = compare_match("Player One", "Player Two", markets, model_p1=0.55)
    assert comparison is not None
    assert comparison["player1"]["midpoint"] == pytest.approx(0.60)
    assert comparison["player1"]["de_vig_probability"] == pytest.approx(0.60)
    assert comparison["model_minus_market_p1"] == pytest.approx(-0.05)
