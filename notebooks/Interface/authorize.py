from typing import List, Tuple

from schemas import TransactionRequest
from Rule_Engine.Engine import run_rules


Decision = str
ReasonList = List[str]


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

    # 1. Run rule engine (fast, deterministic, centralised logic)
    rule_decision, rule_reasons = run_rules(transaction)

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


