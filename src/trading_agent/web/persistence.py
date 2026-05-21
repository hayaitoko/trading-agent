import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker

QuoteFn = Callable[[str], Decimal]

SECRET_KEYS: tuple[str, ...] = (
    "reddit_client_id",
    "reddit_client_secret",
    "reddit_user_agent",
    "stocktwits_token",
    "investopedia_username",
    "investopedia_password",
)


@dataclass
class AccountSpec:
    id: str
    name: str
    starting_cash: Decimal
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "starting_cash": str(self.starting_cash),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccountSpec":
        return cls(
            id=data["id"],
            name=data["name"],
            starting_cash=Decimal(data["starting_cash"]),
            enabled=bool(data.get("enabled", True)),
        )


def load_specs(path: Path) -> list[AccountSpec]:
    if not path.exists():
        return []
    return [AccountSpec.from_dict(d) for d in json.loads(path.read_text())]


def save_specs(specs: list[AccountSpec], path: Path) -> None:
    path.write_text(json.dumps([s.to_dict() for s in specs], indent=2))


def load_secrets(path: Path) -> dict[str, str]:
    base = {k: "" for k in SECRET_KEYS}
    if path.exists():
        loaded = json.loads(path.read_text())
        base.update({k: str(v) for k, v in loaded.items() if k in SECRET_KEYS})
    return base


def save_secrets(secrets: dict[str, str], path: Path) -> None:
    sanitized = {k: secrets.get(k, "") for k in SECRET_KEYS}
    path.write_text(json.dumps(sanitized, indent=2))


def build_account(spec: AccountSpec, quote_fn: QuoteFn) -> Account:
    broker = MockBroker(cash=spec.starting_cash, quote_fn=quote_fn)
    return Account(
        id=spec.id,
        name=spec.name,
        broker=broker,
        starting_cash=spec.starting_cash,
        enabled=spec.enabled,
    )


def specs_from_accounts(accounts: dict[str, Account]) -> list[AccountSpec]:
    return [
        AccountSpec(
            id=a.id,
            name=a.name,
            starting_cash=a.starting_cash,
            enabled=a.enabled,
        )
        for a in accounts.values()
    ]
