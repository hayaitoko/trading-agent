from trading_agent.sentiment.tickers import extract_tickers


def test_cashtag_extracted():
    assert extract_tickers("loaded up on $TSLA today") == ["TSLA"]


def test_bareword_extracted():
    assert extract_tickers("AAPL earnings beat") == ["AAPL"]


def test_common_word_filtered():
    assert extract_tickers("THE CEO sold shares") == []


def test_mixed_cashtag_and_bareword():
    result = extract_tickers("$NVDA vs AMD, which one?")
    assert set(result) == {"NVDA", "AMD"}


def test_cashtag_not_double_counted_as_bareword():
    assert extract_tickers("$GME GME again") == ["GME"]


def test_empty_text():
    assert extract_tickers("") == []


def test_lowercase_ignored():
    assert extract_tickers("buy tsla soon") == []


def test_too_long_ignored():
    assert extract_tickers("MSFTAB is not real") == []
