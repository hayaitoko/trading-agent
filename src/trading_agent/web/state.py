from dataclasses import dataclass, field

from trading_agent.accounts import Account


@dataclass
class AppState:
    accounts: dict[str, Account] = field(default_factory=dict)

    def add_account(self, account: Account) -> None:
        if account.id in self.accounts:
            raise ValueError(f"duplicate account id: {account.id}")
        self.accounts[account.id] = account
