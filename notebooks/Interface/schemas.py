# fraud_system/interface/schemas.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransactionRequest:
    user_id: str
    amount: float
    merchant_id: str
    timestamp: str
    currency: str = "INR"
    device_id: Optional[str] = None
    location: Optional[str] = None


@dataclass
class TransactionResponse:
    transaction_id: str
    decision: str
    reason: list[str]
