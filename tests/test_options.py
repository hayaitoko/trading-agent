"""Single-leg options: contract identity, marks, Black–Scholes pricing, the
×100 P&L book, expiry/assignment, and the chain provider (fail-loud + offline)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from trading_agent.instruments.options import (
    OptionContract,
    OptionQuote,
    OptionRight,
    OptionsBook,
    black_scholes_price,
    mark_price,
)
from trading_agent.instruments.options_chain import (
    AlpacaOptionChainProvider,
    OptionChainProvider,
    OptionsProviderError,
)


def _call(strike: float = 150.0, expiry: date = date(2026, 6, 19)) -> OptionContract:
    return OptionContract("AAPL", expiry, strike, OptionRight.CALL)


def _put(strike: float = 150.0, expiry: date = date(2026, 6, 19)) -> OptionContract:
    return OptionContract("AAPL", expiry, strike, OptionRight.PUT)


# --- contract identity -------------------------------------------------------


def test_occ_symbol_format() -> None:
    c = OptionContract("AAPL", date(2024, 1, 19), 150.0, OptionRight.CALL)
    assert c.occ_symbol == "AAPL  240119C00150000"


def test_occ_round_trip() -> None:
    c = OptionContract("AAPL", date(2024, 1, 19), 152.5, OptionRight.PUT)
    parsed = OptionContract.from_occ(c.occ_symbol)
    assert parsed == c


def test_occ_round_trip_short_root() -> None:
    c = OptionContract("F", date(2026, 12, 18), 12.0, OptionRight.CALL)
    assert OptionContract.from_occ(c.occ_symbol) == c


def test_from_occ_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        OptionContract.from_occ("NOPE")


def test_intrinsic_call_and_put() -> None:
    assert _call(150).intrinsic(170.0) == pytest.approx(20.0)
    assert _call(150).intrinsic(140.0) == 0.0
    assert _put(150).intrinsic(130.0) == pytest.approx(20.0)
    assert _put(150).intrinsic(160.0) == 0.0


def test_is_expired() -> None:
    c = _call(expiry=date(2026, 6, 19))
    assert c.is_expired(date(2026, 6, 18)) is False
    assert c.is_expired(date(2026, 6, 19)) is True
    assert c.is_expired(date(2026, 6, 20)) is True


# --- mark mechanism ----------------------------------------------------------


def test_mark_price_mid_of_two_sided() -> None:
    assert mark_price(1.0, 1.4, None) == pytest.approx(1.2)


def test_mark_price_falls_back_to_last_then_single_side() -> None:
    assert mark_price(None, None, 2.5) == 2.5
    assert mark_price(1.0, None, None) == 1.0
    assert mark_price(None, 1.5, None) == 1.5
    assert mark_price(None, None, None) is None


def test_option_quote_mark_property() -> None:
    q = OptionQuote(_call(), bid=2.0, ask=2.2)
    assert q.mark == pytest.approx(2.1)


# --- Black–Scholes pricing ---------------------------------------------------


def test_bs_call_above_intrinsic_with_time_value() -> None:
    price = black_scholes_price(OptionRight.CALL, 100.0, 100.0, 1.0, 0.02, 0.2)
    assert price > 0.0
    # ATM call has only time value (intrinsic 0) and should be a sane magnitude.
    assert 5.0 < price < 12.0


def test_bs_zero_time_returns_intrinsic() -> None:
    assert black_scholes_price(OptionRight.CALL, 120.0, 100.0, 0.0, 0.02, 0.2) == pytest.approx(20.0)
    assert black_scholes_price(OptionRight.PUT, 80.0, 100.0, 0.0, 0.0, 0.2) == pytest.approx(20.0)


def test_bs_put_call_parity() -> None:
    import math

    spot, strike, t, r, vol = 100.0, 95.0, 0.5, 0.03, 0.25
    call = black_scholes_price(OptionRight.CALL, spot, strike, t, r, vol)
    put = black_scholes_price(OptionRight.PUT, spot, strike, t, r, vol)
    # C - P == S - K*e^{-rt}
    assert (call - put) == pytest.approx(spot - strike * math.exp(-r * t), abs=1e-6)


# --- book: position + P&L with ×100 multiplier -------------------------------


def test_buy_option_uses_multiplier_for_cash_and_position() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    result = book.buy(c, 2, 3.0)  # 2 contracts @ $3.00 → 2*3*100 = $600
    assert result["status"] == "FILLED"
    assert book.cash == pytest.approx(99_400.0)
    pos = book.position(c)
    assert pos.quantity == 2
    assert pos.avg_price == pytest.approx(3.0)


def test_buy_rejected_when_unaffordable() -> None:
    book = OptionsBook(initial_balance=500.0)
    result = book.buy(_call(), 2, 3.0)  # needs $600
    assert result["status"] == "REJECTED"
    assert book.position(_call()) is None
    assert book.cash == 500.0


def test_winning_long_call_realizes_pnl_times_multiplier() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    book.buy(c, 1, 2.0)
    sell = book.sell(c, 1, 5.0)  # +3.00 * 100 = +300
    assert sell["status"] == "FILLED"
    assert book.position(c) is None
    assert book.get_realized_pnl() == pytest.approx(300.0)
    assert book.get_win_loss() == (1, 0)
    assert book.cash == pytest.approx(100_000.0 + 300.0)


def test_losing_long_call_realizes_negative_pnl() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    book.buy(c, 1, 4.0)
    book.sell(c, 1, 1.0)  # -3.00 * 100 = -300
    assert book.get_realized_pnl() == pytest.approx(-300.0)
    assert book.get_win_loss() == (0, 1)


def test_partial_close_books_one_event_proportionally() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    book.buy(c, 4, 2.0)
    book.sell(c, 1, 5.0)  # close 1 of 4 → +3 * 100
    assert book.get_realized_pnl() == pytest.approx(300.0)
    assert book.get_win_loss() == (1, 0)
    assert book.position(c).quantity == 3


def test_account_value_marks_to_market() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    book.buy(c, 2, 3.0)  # cash 99_400, holding 2 @ mark 4.5 → 2*4.5*100 = 900
    val = book.account_value({c.occ_symbol: 4.5})
    assert val == pytest.approx(99_400.0 + 900.0)


# --- writing (short) options -------------------------------------------------


def test_writing_requires_allow_short() -> None:
    book = OptionsBook(initial_balance=100_000.0)  # allow_short defaults False
    assert book.sell(_call(), 1, 3.0)["status"] == "REJECTED"
    assert book.position(_call()) is None


def test_write_then_buy_to_close_short() -> None:
    book = OptionsBook(initial_balance=100_000.0, allow_short=True)
    c = _put()
    book.sell(c, 1, 4.0)  # write 1 put, collect $400
    assert book.cash == pytest.approx(100_400.0)
    assert book.position(c).quantity == -1
    book.buy(c, 1, 1.5)  # buy back cheaper → +2.50 * 100 = +250
    assert book.position(c) is None
    assert book.get_realized_pnl() == pytest.approx(250.0)
    assert book.get_win_loss() == (1, 0)


# --- expiry / assignment -----------------------------------------------------


def test_long_call_expires_itm_settles_to_intrinsic() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call(strike=150.0)
    book.buy(c, 1, 2.0)  # paid $200
    book.settle_expiry(c, underlying_price=160.0)  # intrinsic 10 → +$1000 cash
    assert book.position(c) is None
    # realized = (10 - 2) * 100 = +800
    assert book.get_realized_pnl() == pytest.approx(800.0)
    assert book.cash == pytest.approx(100_000.0 - 200.0 + 1_000.0)


def test_long_call_expires_otm_worthless() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call(strike=150.0)
    book.buy(c, 1, 2.0)
    book.settle_expiry(c, underlying_price=140.0)  # OTM → worth 0, lose premium
    assert book.position(c) is None
    assert book.get_realized_pnl() == pytest.approx(-200.0)
    assert book.get_win_loss() == (0, 1)


def test_short_put_otm_expires_keeps_full_premium() -> None:
    book = OptionsBook(initial_balance=100_000.0, allow_short=True)
    c = _put(strike=150.0)
    book.sell(c, 1, 4.0)  # collect $400
    book.settle_expiry(c, underlying_price=160.0)  # OTM put → worthless
    assert book.position(c) is None
    assert book.get_realized_pnl() == pytest.approx(400.0)  # full premium kept
    assert book.cash == pytest.approx(100_400.0)


def test_short_put_itm_assignment_books_loss() -> None:
    book = OptionsBook(initial_balance=100_000.0, allow_short=True)
    c = _put(strike=150.0)
    book.sell(c, 1, 4.0)  # collect $400
    book.settle_expiry(c, underlying_price=130.0)  # ITM put, intrinsic 20 → assigned
    assert book.position(c) is None
    # realized = (4 - 20) * 100 = -1600
    assert book.get_realized_pnl() == pytest.approx(-1600.0)
    # cash: +400 (premium) - 2000 (assignment pay) = -1600 vs start
    assert book.cash == pytest.approx(100_000.0 - 1_600.0)


def test_expire_settles_only_expired_contracts() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    near = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
    far = OptionContract("AAPL", date(2026, 12, 18), 150.0, OptionRight.CALL)
    book.buy(near, 1, 2.0)
    book.buy(far, 1, 5.0)
    settled = book.expire(date(2026, 6, 19), {"AAPL": 160.0})
    assert settled == [near.occ_symbol]
    assert book.position(near) is None
    assert book.position(far) is not None  # not yet expired


def test_expire_skips_contract_with_no_underlying_price() -> None:
    book = OptionsBook(initial_balance=100_000.0)
    c = _call()
    book.buy(c, 1, 2.0)
    settled = book.expire(date(2026, 6, 19), {})  # no AAPL price → untouched
    assert settled == []
    assert book.position(c) is not None


# --- chain provider: real data only, fail loud, offline seam -----------------


def test_provider_satisfies_protocol() -> None:
    assert isinstance(AlpacaOptionChainProvider(api_key="k", secret_key="s"), OptionChainProvider)


def test_provider_fails_loud_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    provider = AlpacaOptionChainProvider()
    with pytest.raises(OptionsProviderError, match="keys missing"):
        provider.get_chain("AAPL")


class _FakeOptionDataClient:
    """A stand-in for OptionHistoricalDataClient returning a canned chain."""

    def __init__(self, chain: dict) -> None:
        self._chain = chain
        self.last_request = None

    def get_option_chain(self, request):  # noqa: ANN001 - duck-typed seam
        self.last_request = request
        return self._chain


def _snapshot(bid, ask, last):
    return SimpleNamespace(
        latest_quote=SimpleNamespace(bid_price=bid, ask_price=ask),
        latest_trade=SimpleNamespace(price=last),
    )


def test_provider_parses_injected_chain_offline() -> None:
    c = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
    fake = _FakeOptionDataClient({c.occ_symbol: _snapshot(2.0, 2.4, 2.2)})
    provider = AlpacaOptionChainProvider(data_client=fake)
    chain = provider.get_chain("AAPL")
    assert len(chain) == 1
    quote = chain[0]
    assert quote.contract == c
    assert quote.mark == pytest.approx(2.2)  # mid of 2.0/2.4


def test_provider_get_quote_filters_to_contract() -> None:
    want = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
    other = OptionContract("AAPL", date(2026, 6, 19), 155.0, OptionRight.CALL)
    fake = _FakeOptionDataClient(
        {want.occ_symbol: _snapshot(2.0, 2.4, 2.2), other.occ_symbol: _snapshot(1.0, 1.2, 1.1)}
    )
    provider = AlpacaOptionChainProvider(data_client=fake)
    quote = provider.get_quote(want)
    assert quote is not None
    assert quote.contract.strike == 150.0


def test_provider_wraps_client_errors() -> None:
    class _Boom:
        def get_option_chain(self, request):  # noqa: ANN001
            raise RuntimeError("network down")

    provider = AlpacaOptionChainProvider(data_client=_Boom())
    with pytest.raises(OptionsProviderError, match="option chain failed"):
        provider.get_chain("AAPL")
