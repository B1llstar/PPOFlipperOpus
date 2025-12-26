#!/usr/bin/env python3
"""
Tax Logging Module for GE Trading

Tracks and reports taxes paid on GE transactions.
The Grand Exchange applies a 1% tax on all sell transactions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

logger = logging.getLogger("TaxLog")

# Tax constants
GE_TAX_RATE = 0.01  # 1% tax on sells

# Global tax tracking
_tax_records: List[Dict] = []
_total_tax_paid: float = 0.0


def log_tax_payment(
    item: str,
    quantity: int,
    sell_price: float,
    tax_amount: float,
    agent_id: Optional[int] = None
):
    """
    Log a tax payment from a sell transaction.

    Args:
        item: Item name or ID
        quantity: Quantity sold
        sell_price: Price per unit
        tax_amount: Tax paid (should be ~1% of gross)
        agent_id: Optional agent identifier
    """
    global _total_tax_paid

    record = {
        "timestamp": datetime.now().isoformat(),
        "item": str(item),
        "quantity": quantity,
        "sell_price": sell_price,
        "gross": quantity * sell_price,
        "tax": tax_amount,
        "agent_id": agent_id
    }

    _tax_records.append(record)
    _total_tax_paid += tax_amount

    logger.debug(
        f"Tax: {tax_amount:,.0f} GP on {quantity}x {item} @ {sell_price:,.0f}"
    )


def log_tax_summary():
    """Log a summary of taxes paid."""
    if not _tax_records:
        logger.info("No tax payments recorded")
        return

    total_gross = sum(r["gross"] for r in _tax_records)
    n_transactions = len(_tax_records)
    avg_tax = _total_tax_paid / n_transactions if n_transactions > 0 else 0

    logger.info("=== Tax Summary ===")
    logger.info(f"Total transactions: {n_transactions:,}")
    logger.info(f"Total gross sales: {total_gross:,.0f} GP")
    logger.info(f"Total tax paid: {_total_tax_paid:,.0f} GP")
    logger.info(f"Average tax per transaction: {avg_tax:,.0f} GP")
    logger.info(f"Effective tax rate: {(_total_tax_paid / total_gross * 100):.2f}%")


def create_tax_report(output_path: str = "tax_report.json") -> Dict:
    """
    Create a detailed tax report.

    Args:
        output_path: Path to save the JSON report

    Returns:
        Report dictionary
    """
    if not _tax_records:
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_transactions": 0,
                "total_gross": 0,
                "total_tax": 0,
                "effective_rate": 0
            },
            "records": []
        }
    else:
        total_gross = sum(r["gross"] for r in _tax_records)
        n_transactions = len(_tax_records)

        # Group by item
        by_item: Dict[str, Dict] = {}
        for record in _tax_records:
            item = record["item"]
            if item not in by_item:
                by_item[item] = {
                    "transactions": 0,
                    "quantity": 0,
                    "gross": 0,
                    "tax": 0
                }
            by_item[item]["transactions"] += 1
            by_item[item]["quantity"] += record["quantity"]
            by_item[item]["gross"] += record["gross"]
            by_item[item]["tax"] += record["tax"]

        # Group by agent
        by_agent: Dict[int, Dict] = {}
        for record in _tax_records:
            agent_id = record.get("agent_id", 0)
            if agent_id not in by_agent:
                by_agent[agent_id] = {
                    "transactions": 0,
                    "gross": 0,
                    "tax": 0
                }
            by_agent[agent_id]["transactions"] += 1
            by_agent[agent_id]["gross"] += record["gross"]
            by_agent[agent_id]["tax"] += record["tax"]

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_transactions": n_transactions,
                "total_gross": total_gross,
                "total_tax": _total_tax_paid,
                "effective_rate": _total_tax_paid / total_gross if total_gross > 0 else 0
            },
            "by_item": by_item,
            "by_agent": {str(k): v for k, v in by_agent.items()},
            "recent_records": _tax_records[-100:]  # Last 100 transactions
        }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Tax report saved to {output_path}")

    return report


def get_total_tax_paid() -> float:
    """Get total tax paid across all transactions."""
    return _total_tax_paid


def get_tax_records() -> List[Dict]:
    """Get all tax records."""
    return _tax_records.copy()


def reset_tax_tracking():
    """Reset tax tracking (for new training runs)."""
    global _tax_records, _total_tax_paid
    _tax_records = []
    _total_tax_paid = 0.0
    logger.info("Tax tracking reset")


def calculate_tax(gross_amount: float) -> float:
    """
    Calculate tax for a given gross amount.

    Args:
        gross_amount: Gross sale proceeds

    Returns:
        Tax amount (1% of gross)
    """
    return gross_amount * GE_TAX_RATE


def calculate_net_proceeds(gross_amount: float) -> float:
    """
    Calculate net proceeds after tax.

    Args:
        gross_amount: Gross sale proceeds

    Returns:
        Net proceeds (99% of gross)
    """
    return gross_amount * (1 - GE_TAX_RATE)
