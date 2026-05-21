from dataclasses import dataclass, field
from pathlib import Path

from trading_agent.accounts import Account
from trading_agent.web.persistence import (
    QuoteFn,
    build_account,
    load_secrets,
    load_specs,
    save_secrets,
    save_specs,
    specs_from_accounts,
)


@dataclass
class AppState:
    accounts_path: Path
    secrets_path: Path
    quote_fn: QuoteFn
    accounts: dict[str, Account] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    notes_dir: Path | None = None

    def hydrate(self) -> None:
        self.secrets = load_secrets(self.secrets_path)
        self.accounts = {}
        for spec in load_specs(self.accounts_path):
            account = build_account(spec, self.quote_fn)
            self.accounts[account.id] = account

    def add_account(self, account: Account) -> None:
        if account.id in self.accounts:
            raise ValueError(f"duplicate account id: {account.id}")
        self.accounts[account.id] = account
        self._persist_accounts()

    def remove_account(self, account_id: str) -> None:
        del self.accounts[account_id]
        self._persist_accounts()

    def toggle_account(self, account_id: str) -> Account:
        account = self.accounts[account_id]
        account.enabled = not account.enabled
        self._persist_accounts()
        return account

    def update_secrets(self, secrets: dict[str, str]) -> None:
        self.secrets = {**self.secrets, **secrets}
        save_secrets(self.secrets, self.secrets_path)

    def _persist_accounts(self) -> None:
        save_specs(specs_from_accounts(self.accounts), self.accounts_path)
