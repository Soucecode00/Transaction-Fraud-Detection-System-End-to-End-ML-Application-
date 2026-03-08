# Feature_Layer/__init__.py

"""
Feature Layer: Transforms Interface-layer transactions into ML-compatible features.

This layer bridges the gap between:
- Interface schema (user_id, amount, merchant_id, timestamp, etc.)
- ML model schema (step, type, balances, etc.)
"""

from .transaction_adapter import FeatureAdapter

__all__ = ["FeatureAdapter"]


