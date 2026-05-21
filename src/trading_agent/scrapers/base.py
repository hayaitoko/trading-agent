from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from trading_agent.models import Post


class Scraper(ABC):
    source_name: str

    @abstractmethod
    def poll(self, since: datetime | None = None) -> AsyncIterator[Post]: ...
