from handler import handle_transaction


def demo_request() -> None:
    """
    Simple CLI-style demo for the Interface layer.

    In a real deployment this layer would typically sit behind an HTTP API
    (FastAPI / Flask / Django, etc.). For now we just construct a sample
    raw transaction dict and run it through the full interface → authorize
    pipeline.
    """

    raw_txn = {
        "user_id": "user_123",
        "amount": 1000,
        "merchant_id": "merchant_456",
        "timestamp": "2026-02-12T10:00:00Z",
        "currency": "INR",
        "device_id": "device_abc",
        "location": "IN",
    }

    response = handle_transaction(raw_txn)

    print("--- Interface Layer Demo ---")
    print(f"Transaction ID : {response.transaction_id}")
    print(f"Decision       : {response.decision}")
    print("Reasons:")
    for r in response.reason:
        print(f"  - {r}")


if __name__ == "__main__":
    demo_request()


