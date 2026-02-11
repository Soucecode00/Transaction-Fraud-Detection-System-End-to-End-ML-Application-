from typing import List, Tuple

from schemas import TransactionRequest


Decision = str
ReasonList = List[str]


def _run_basic_rules(txn: TransactionRequest) -> Tuple[Decision, ReasonList]:
    """
    Very simple, fast rule engine that can run inline.

    In a real system this would likely call into a dedicated Rule_Engine
    module and include many more checks (velocity, device risk, lists, etc.).
    """

    reasons: ReasonList = []
    decision: Decision = "APPROVE"

    # Example hard guardrail: non-positive amounts are invalid
    if txn.amount <= 0:
        return "DECLINE", ["Invalid transaction amount (<= 0)"]

    # Simple amount-based tiers (illustrative thresholds)
    if txn.amount > 100_000:
        decision = "DECLINE"
        reasons.append("Amount exceeds maximum allowed limit")
    elif txn.amount > 50_000:
        decision = "REVIEW"
        reasons.append("High-value transaction requires manual review")

    # Example: basic geo / device heuristics (placeholders)
    if txn.location is None:
        reasons.append("Missing transaction location")
    if txn.device_id is None:
        reasons.append("Missing device identifier")

    if not reasons:
        reasons.append("Passed basic rules")

    return decision, reasons


def authorize_transaction(
    transaction_id: str,
    transaction: TransactionRequest,
) -> Tuple[Decision, ReasonList]:
    """
    Top-level authorization function called by the Interface handler.

    Responsibilities:
    - Orchestrate calls to rules / ML / feature layers
    - Aggregate a single decision string + list of human-readable reasons
    """

    # 1. Run basic rules (fast, deterministic)
    rule_decision, rule_reasons = _run_basic_rules(transaction)

    # 2. (Placeholder) Call out to ML / feature layer if needed
    #    For now we only rely on rules. Later you can:
    #    - Build a single-txn feature builder
    #    - Load the trained XGBoost model + threshold
    #    - Combine probability + rules to form a final decision

    # Right now, we just return the rule-based decision.
    decision = rule_decision
    reasons = [f"[rules] {r}" for r in rule_reasons]

    # You could also attach metadata like transaction_id here if desired.
    return decision, reasons


