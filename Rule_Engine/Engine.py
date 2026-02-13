"""
Rule Engine for transactional fraud decisions.

This module encapsulates all rule-based logic in one place so that:
- Rules are easy to read, test, and extend
- The Interface / authorization layer can call a single `run_rules` function
- The ML engine can sit on top of (or alongside) these rules in a hybrid system

Rules operate on the Interface-layer `TransactionRequest` schema and return:
- A high-level decision: "APPROVE", "REVIEW", or "DECLINE"
- A list of human-readable reasons explaining which rules fired
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from Interface.schemas import TransactionRequest


Decision = str
ReasonList = List[str]


@dataclass
class RuleResult:
    """
    Outcome of a single rule evaluation.

    Attributes
    ----------
    name : str
        Identifier for the rule (used in logs / explanations).
    hit : bool
        Whether the rule triggered for this transaction.
    decision : Optional[str]
        Suggested decision from this rule:
        - "DECLINE" for hard blocks
        - "REVIEW" for manual review / extra checks
        - "APPROVE" for explicit approvals (rare)
        - None if this rule doesn't directly affect the decision
    reasons : list[str]
        Human-readable explanations for why the rule triggered.
    severity : int
        Higher severity means the rule is more critical. Used as a
        tiebreaker when multiple rules suggest different decisions.
    """

    name: str
    hit: bool
    decision: Optional[Decision]
    reasons: ReasonList
    severity: int = 0


Rule = Callable[[TransactionRequest], RuleResult]


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def _amount_non_positive_rule(txn: TransactionRequest) -> RuleResult:
    """
    Hard guardrail: transaction amount must be positive.
    """
    if txn.amount is None or txn.amount <= 0:
        return RuleResult(
            name="amount_non_positive",
            hit=True,
            decision="DECLINE",
            reasons=["Transaction amount must be positive"],
            severity=100,
        )

    return RuleResult(
        name="amount_non_positive",
        hit=False,
        decision=None,
        reasons=[],
    )


def _max_amount_rule(txn: TransactionRequest) -> RuleResult:
    """
    Hard limit: decline if amount exceeds a configured maximum.

    NOTE: The 100_000 threshold is illustrative. In a real system, this
    would come from configuration or a policy management system.
    """
    if txn.amount is not None and txn.amount > 100_000:
        return RuleResult(
            name="max_amount_limit",
            hit=True,
            decision="DECLINE",
            reasons=["Amount exceeds maximum allowed limit (100,000)"],
            severity=90,
        )

    return RuleResult(
        name="max_amount_limit",
        hit=False,
        decision=None,
        reasons=[],
    )


def _high_value_review_rule(txn: TransactionRequest) -> RuleResult:
    """
    High-value transactions should be sent for manual review.
    """
    if txn.amount is not None and 50_000 < txn.amount <= 100_000:
        return RuleResult(
            name="high_value_review",
            hit=True,
            decision="REVIEW",
            reasons=["High-value transaction requires additional scrutiny"],
            severity=50,
        )

    return RuleResult(
        name="high_value_review",
        hit=False,
        decision=None,
        reasons=[],
    )


def _missing_location_rule(txn: TransactionRequest) -> RuleResult:
    """
    Missing location information is suspicious but not always a hard block.
    """
    if txn.location is None:
        return RuleResult(
            name="missing_location",
            hit=True,
            decision="REVIEW",
            reasons=["Missing transaction location"],
            severity=20,
        )

    return RuleResult(
        name="missing_location",
        hit=False,
        decision=None,
        reasons=[],
    )


def _missing_device_rule(txn: TransactionRequest) -> RuleResult:
    """
    Missing device identifier is suspicious but not always a hard block.
    """
    if txn.device_id is None:
        return RuleResult(
            name="missing_device_id",
            hit=True,
            decision="REVIEW",
            reasons=["Missing device identifier"],
            severity=20,
        )

    return RuleResult(
        name="missing_device_id",
        hit=False,
        decision=None,
        reasons=[],
    )


# Register all rules here so the engine can run them in sequence.
ALL_RULES: List[Rule] = [
    _amount_non_positive_rule,
    _max_amount_rule,
    _high_value_review_rule,
    _missing_location_rule,
    _missing_device_rule,
]


# ---------------------------------------------------------------------------
# Rule engine orchestration
# ---------------------------------------------------------------------------

def run_rules(txn: TransactionRequest) -> Tuple[Decision, ReasonList]:
    """
    Evaluate all rules against a transaction and produce a single decision.

    Combination logic:
    - Start with default decision: "APPROVE"
    - Evaluate all rules and collect RuleResult objects
    - Any rule that hits contributes its reasons
    - Decision priority:
        DECLINE > REVIEW > APPROVE
      with `severity` as a tiebreaker between rules suggesting the same
      decision type.

    Returns
    -------
    decision : str
        One of "APPROVE", "REVIEW", "DECLINE"
    reasons : list[str]
        Combined, human-readable explanations for all triggered rules.
    """

    # Evaluate all rules
    results: List[RuleResult] = [rule(txn) for rule in ALL_RULES]

    # Default decision if nothing triggers
    final_decision: Decision = "APPROVE"
    final_reasons: ReasonList = []

    # Collect reasons from all hit rules
    for res in results:
        if res.hit and res.reasons:
            final_reasons.extend(res.reasons)

    # Determine the strongest suggested decision
    decision_priority = {"DECLINE": 2, "REVIEW": 1, "APPROVE": 0, None: -1}
    best_score = decision_priority[final_decision]
    best_severity = -1

    for res in results:
        if not res.hit or res.decision is None:
            continue

        score = decision_priority.get(res.decision, -1)

        # Choose the decision with the highest priority, break ties by severity
        if score > best_score or (score == best_score and res.severity > best_severity):
            best_score = score
            best_severity = res.severity
            final_decision = res.decision

    if not final_reasons:
        final_reasons.append("No rules triggered; default APPROVE decision")

    return final_decision, final_reasons


__all__ = ["Decision", "ReasonList", "RuleResult", "run_rules"]

