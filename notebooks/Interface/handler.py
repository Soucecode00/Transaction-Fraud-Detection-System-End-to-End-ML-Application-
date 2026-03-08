# fraud_system/interface/handler.py

import uuid
from validator import TransactionValidator
from schemas import TransactionResponse
from authorize import authorize_transaction


def handle_transaction(raw_txn: dict) -> TransactionResponse:
    """
    Interface layer entrypoint
    """

    # 1. Validate & normalize input
    txn = TransactionValidator.validate(raw_txn)

    # 2. Generate transaction ID
    transaction_id = str(uuid.uuid4())

    # 3. Call core fraud system
    decision, reason = authorize_transaction(
        transaction_id=transaction_id,
        transaction=txn
    )

    # 4. Build response
    return TransactionResponse(
        transaction_id=transaction_id,
        decision=decision,
        reason=reason
    )
