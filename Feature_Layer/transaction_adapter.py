# Feature_Layer/transaction_adapter.py

"""
Feature Adaptation Layer: Transforms Interface-layer transactions into ML-compatible features.

This module bridges the gap between:
- Interface schema: user_id, amount, merchant_id, timestamp, etc.
- ML model schema: step, type, amount, oldbalanceOrg, newbalanceOrig, etc.
"""

import pandas as pd
import sys
import os
from typing import Optional, Dict, Tuple, List
from datetime import datetime

# Add parent directory to path to import Interface schemas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Interface.schemas import TransactionRequest
except ImportError:
    # Fallback if import fails
    TransactionRequest = None


class FeatureAdapter:
    """
    Transforms Interface-layer TransactionRequest into ML-compatible features.
    
    This is the bridge between your API schema and the ML model's expected input.
    The ML model expects PaySim-style features (step, type, balances, etc.),
    but the Interface layer receives simpler transaction data.
    """
    
    # Required fields for ML feature engineering (PaySim schema)
    REQUIRED_ML_FIELDS = [
        "step",           # Time step (derived from timestamp)
        "type",           # Transaction type: "TRANSFER" or "CASH_OUT"
        "amount",         # Transaction amount (direct mapping)
        "oldbalanceOrg",  # Origin account balance before transaction
        "newbalanceOrig", # Origin account balance after transaction
        "oldbalanceDest", # Destination account balance before transaction
        "newbalanceDest" # Destination account balance after transaction
    ]
    
    @staticmethod
    def validate_can_transform(txn: TransactionRequest) -> Tuple[bool, List[str]]:
        """
        Check if we have enough data to transform to ML features.
        
        This validates that we can derive all required ML fields from the
        TransactionRequest. In production, this would check:
        - Can we fetch balance data from DB/cache?
        - Can we map timestamp to step?
        - Can we infer transaction type?
        
        Args:
            txn: TransactionRequest from Interface layer
        
        Returns:
            Tuple of (can_transform: bool, missing_fields: List[str])
            - can_transform: True if we can build ML features
            - missing_fields: List of fields that are missing/unavailable
        """
        missing = []
        
        # Basic validation: check required Interface fields exist
        if txn is None:
            return False, ["TransactionRequest is None"]
        
        if txn.amount is None or txn.amount <= 0:
            missing.append("amount (must be positive)")
        
        if not txn.timestamp:
            missing.append("timestamp (required for step conversion)")
        
        # In production, you'd also check:
        # - Can we fetch balance data? (DB/cache availability)
        # - Do we have user_id to look up account balances?
        # - Can we infer transaction type from merchant_id?
        
        # For now, we assume we can derive everything (with placeholders if needed)
        # In production, add real validation here:
        #   if not can_fetch_balance_data(txn.user_id):
        #       missing.append("balance_data (cannot fetch from DB)")
        
        can_transform = len(missing) == 0
        return can_transform, missing
    
    @staticmethod
    def _timestamp_to_step(timestamp: str) -> int:
        """
        Convert timestamp string to step (time step used by ML model).
        
        In production, you'd have a proper mapping from your timestamp format
        to the step encoding used during training.
        
        Args:
            timestamp: ISO format timestamp string
        
        Returns:
            step: Integer time step (0-743 for PaySim, representing hours)
        """
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # Simplified: use hour of day as step (0-23)
            # In production, map to your actual step encoding
            step = dt.hour
            return step
        except (ValueError, AttributeError):
            # Fallback: default to noon (step 12)
            return 12
    
    @staticmethod
    def _infer_transaction_type(txn: TransactionRequest) -> str:
        """
        Infer transaction type from transaction data.
        
        The ML model only works with "TRANSFER" and "CASH_OUT" types.
        In production, you'd:
        - Look up merchant_id in a database
        - Check transaction metadata
        - Use business rules to determine type
        
        Args:
            txn: TransactionRequest
        
        Returns:
            "TRANSFER" or "CASH_OUT"
        """
        # Simplified logic: use amount as heuristic
        # High amounts more likely to be TRANSFER
        if txn.amount > 10000:
            return "TRANSFER"
        else:
            return "CASH_OUT"
        
        # In production, replace with:
        #   merchant_type = lookup_merchant_type(txn.merchant_id)
        #   return merchant_type or "TRANSFER"  # default
    
    @staticmethod
    def _fetch_balance_data(
        txn: TransactionRequest,
        balance_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Fetch or compute balance data for ML features.
        
        In production, this would:
        - Query database/cache for user account balances
        - Look up merchant/destination account balances
        - Handle missing data gracefully
        
        Args:
            txn: TransactionRequest
            balance_data: Optional pre-fetched balance data
        
        Returns:
            Dict with keys: oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
        """
        if balance_data is not None:
            # Use provided balance data (from DB/cache)
            return {
                "oldbalanceOrg": balance_data.get("oldbalanceOrg", txn.amount * 2.0),
                "newbalanceOrig": balance_data.get("newbalanceOrig", 0.0),
                "oldbalanceDest": balance_data.get("oldbalanceDest", 0.0),
                "newbalanceDest": balance_data.get("newbalanceDest", txn.amount),
            }
        
        # Placeholder logic (REPLACE WITH REAL DB CALLS IN PRODUCTION)
        # In production, you'd do:
        #   user_balance = db.get_account_balance(txn.user_id)
        #   merchant_balance = db.get_account_balance(txn.merchant_id)
        
        # For now, use simple heuristics:
        oldbalanceOrg = txn.amount * 2.0  # Assume user has 2x transaction amount
        newbalanceOrig = oldbalanceOrg - txn.amount  # Deduct transaction
        oldbalanceDest = 0.0  # Assume destination starts at 0
        newbalanceDest = txn.amount  # Destination receives the amount
        
        return {
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }
    
    @staticmethod
    def transform_to_ml_features(
        txn: TransactionRequest,
        balance_data: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Transform TransactionRequest to PaySim-style DataFrame for ML.
        
        This is the core transformation function that converts your Interface
        schema to the schema expected by build_features() and the ML model.
        
        Args:
            txn: Validated transaction from Interface layer
            balance_data: Optional dict with balance information:
                {
                    "oldbalanceOrg": float,
                    "newbalanceOrig": float,
                    "oldbalanceDest": float,
                    "newbalanceDest": float
                }
                If None, uses placeholder logic (for demo only)
        
        Returns:
            DataFrame with columns matching PaySim schema:
            - step: int
            - type: str ("TRANSFER" or "CASH_OUT")
            - amount: float
            - oldbalanceOrg: float
            - newbalanceOrig: float
            - oldbalanceDest: float
            - newbalanceDest: float
        
        Raises:
            ValueError: If transformation fails
        """
        if txn is None:
            raise ValueError("TransactionRequest cannot be None")
        
        # 1. Convert timestamp to step
        step = FeatureAdapter._timestamp_to_step(txn.timestamp)
        
        # 2. Infer transaction type
        txn_type = FeatureAdapter._infer_transaction_type(txn)
        
        # 3. Get balance data (from DB/cache in production)
        balances = FeatureAdapter._fetch_balance_data(txn, balance_data)
        
        # 4. Build DataFrame matching PaySim schema
        df_ml = pd.DataFrame([{
            "step": step,
            "type": txn_type,
            "amount": float(txn.amount),
            "oldbalanceOrg": float(balances["oldbalanceOrg"]),
            "newbalanceOrig": float(balances["newbalanceOrig"]),
            "oldbalanceDest": float(balances["oldbalanceDest"]),
            "newbalanceDest": float(balances["newbalanceDest"]),
        }])
        
        return df_ml
    
    @staticmethod
    def get_required_ml_fields() -> List[str]:
        """
        Get list of required fields for ML feature engineering.
        
        Returns:
            List of field names required by build_features()
        """
        return FeatureAdapter.REQUIRED_ML_FIELDS.copy()

