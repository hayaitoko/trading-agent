import re

# Cashtagged mentions like $TSLA are high-confidence. Barewords like AAPL are
# noisier, so we filter against COMMON_WORDS to drop English words that happen
# to match 1-5 uppercase letters.
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_BAREWORD_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Frequent English/forum tokens that match the bareword pattern but are not tickers.
# Not exhaustive. Extend as false positives appear in real post data.
COMMON_WORDS: frozenset[str] = frozenset({
    "A", "I", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "ALL", "AND", "ANY", "ARE", "BUT", "CAN", "FOR", "GET", "GOT", "HAS",
    "HAD", "HER", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW", "NOT", "NOW",
    "OUR", "OUT", "SEE", "SHE", "THE", "TOO", "TWO", "WAS", "WAY", "WHO",
    "WHY", "YES", "YET", "YOU",
    "ATH", "ATL", "CEO", "CFO", "DD", "DM", "EOD", "EPS", "ETF", "FOMO",
    "FUD", "GDP", "HODL", "IMO", "IPO", "IRA", "IRS", "LOL", "OTM", "PE",
    "PR", "PT", "QQQ", "RIP", "ROI", "SEC", "SPY", "TLDR", "USA", "WSB",
    "YOLO", "YTD",
})


def extract_tickers(text: str) -> list[str]:
    """Return unique upper-cased ticker symbols mentioned in text.

    Cashtagged forms ($TSLA) are kept verbatim. Barewords are kept only if they
    are not in COMMON_WORDS (so TSLA stays, but THE/CEO/YOLO are dropped).
    """
    cashtagged = set(_CASHTAG_RE.findall(text))
    barewords = {
        match for match in _BAREWORD_RE.findall(text)
        if match not in COMMON_WORDS and match not in cashtagged
    }
    return sorted(cashtagged | barewords)
