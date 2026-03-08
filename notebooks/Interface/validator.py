# fraud_system/interface/validator.py

from schemas import TransactionRequest


class TransactionValidator:

    REQUIRED_FIELDS = ["user_id", "amount", "merchant_id", "timestamp"]

    @staticmethod
    def validate(txn: dict) -> TransactionRequest:
        for field in TransactionValidator.REQUIRED_FIELDS:
            if field not in txn:
                raise ValueError(f"Missing required field: {field}")

        if txn["amount"] <= 0:
            raise ValueError("Transaction amount must be positive")

        return TransactionRequest(
            user_id=str(txn["user_id"]),
            amount=float(txn["amount"]),
            merchant_id=str(txn["merchant_id"]),
            timestamp=str(txn["timestamp"]),
            currency=txn.get("currency", "INR"),
            device_id=txn.get("device_id"),
            location=txn.get("location"),
        )
