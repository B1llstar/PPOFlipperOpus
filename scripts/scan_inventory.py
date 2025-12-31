#!/usr/bin/env python3
# Run with: python3 scripts/scan_inventory.py
"""
Inventory Scanner CLI Tool

Scans and displays inventory/bank state from Firestore for PPO inference.

Usage:
    python scripts/scan_inventory.py                  # Full summary
    python scripts/scan_inventory.py --gold           # Just gold
    python scripts/scan_inventory.py --bank           # Bank items
    python scripts/scan_inventory.py --portfolio      # Portfolio utilization breakdown
    python scripts/scan_inventory.py --positions      # Active PPO positions
    python scripts/scan_inventory.py --limits         # Position limits status
    python scripts/scan_inventory.py --items          # All tradeable items
    python scripts/scan_inventory.py --state          # Full JSON state for inference
    python scripts/scan_inventory.py --watch          # Continuously monitor state
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from firebase.inventory_scanner import InventoryScanner
from firebase.position_sizer import PositionSizer
from firebase.position_tracker import PositionTracker
from config.firebase_config import get_service_account_path, DEFAULT_ACCOUNT_ID


def setup_logging(verbose: bool = False):
    """Set up logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def print_separator(title: str = None):
    """Print a separator line."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
    else:
        print('-'*60)


def display_summary(scanner: InventoryScanner):
    """Display a summary of current state."""
    scanner.print_summary()


def display_gold(scanner: InventoryScanner):
    """Display current gold balance."""
    gold = scanner.get_gold()
    print(f"\nGold Balance: {gold:,} GP")

    # Show account status for context
    status = scanner.get_account_status()
    print(f"Account: {status['account_id']}")
    print(f"Plugin Online: {'Yes' if status['is_online'] else 'No'}")
    if status['last_heartbeat']:
        print(f"Last Heartbeat: {status['last_heartbeat']}")


def display_bank(scanner: InventoryScanner, min_value: int = 0, limit: int = 50):
    """Display bank contents sorted by value."""
    print_separator("Bank Contents")

    bank = scanner.get_bank_accounting()
    print(f"Total Items: {bank.get('item_count', 0):,}")
    print(f"Total Value: {bank.get('total_value', 0):,} GP")
    print(f"Tradeable Items: {bank.get('tradeable_count', 0)}")
    print()

    items = scanner.get_bank_items_by_value(min_value=min_value)
    if not items:
        print("  (no items with value >= {min_value})")
        return

    print(f"Top {min(limit, len(items))} items by value:")
    print(f"{'Item Name':<30} {'Qty':>10} {'Price':>12} {'Total':>15}")
    print('-'*70)

    for item in items[:limit]:
        name = item.get('item_name', 'Unknown')[:29]
        qty = item.get('quantity', 0)
        price = item.get('price_each', 0)
        total = item.get('total_value', 0)
        print(f"{name:<30} {qty:>10,} {price:>12,} {total:>15,}")

    if len(items) > limit:
        print(f"\n  ... and {len(items) - limit} more items")


def display_inventory(scanner: InventoryScanner):
    """Display current inventory contents."""
    print_separator("Inventory Contents")

    inv = scanner.get_inventory_accounting()
    print(f"Gold in Inventory: {inv.get('gold', 0):,}")
    print(f"Items: {len(inv.get('items', {}))}")
    print(f"Free Slots: {inv.get('free_slots', 28)}")
    print()

    items = inv.get('items', {})
    if not items:
        print("  (inventory empty)")
        return

    print(f"{'Item Name':<30} {'Qty':>10} {'Value':>15}")
    print('-'*58)

    for item_id, item_data in items.items():
        name = item_data.get('item_name', f'Item {item_id}')[:29]
        qty = item_data.get('quantity', 0)
        value = item_data.get('total_value', 0)
        print(f"{name:<30} {qty:>10,} {value:>15,}")


def display_ge_slots(scanner: InventoryScanner):
    """Display GE slot states."""
    print_separator("GE Slot Status")

    ge = scanner.get_ge_slots()
    print(f"Slots Available: {ge.get('slots_available', 0)}")
    print(f"Buy Slots Used: {ge.get('buy_slots_used', 0)}")
    print(f"Sell Slots Used: {ge.get('sell_slots_used', 0)}")
    print()

    slots = ge.get('slots', {})
    for i in range(1, 9):
        slot = slots.get(str(i))
        if slot is None:
            print(f"  Slot {i}: Empty")
        else:
            slot_type = slot.get('type', 'unknown')
            item_name = slot.get('item_name', f"Item {slot.get('item_id', '?')}")
            status = slot.get('status', 'unknown')
            qty = slot.get('quantity', 0)
            filled = slot.get('filled_quantity', 0)
            price = slot.get('price', 0)
            print(f"  Slot {i}: [{slot_type.upper()}] {item_name} - {filled}/{qty} @ {price:,} ({status})")


def display_portfolio(scanner: InventoryScanner, top_n: int = 20):
    """Display portfolio utilization breakdown."""
    print_separator("Portfolio Utilization")

    util = scanner.get_portfolio_utilization()

    total = util.get('total_portfolio_value', 0)
    gold = util.get('gold', 0)
    gold_pct = util.get('gold_percent', 0)
    items_value = util.get('items_value', 0)
    items_pct = util.get('items_percent', 0)
    orders_value = util.get('active_orders_value', 0)
    orders_pct = (orders_value / total * 100) if total > 0 else 0

    print(f"Total Portfolio Value: {total:,} GP")
    print()
    print(f"  Gold (liquid):       {gold:>15,} GP  ({gold_pct:>5.1f}%)")
    print(f"  Items (bank+inv):    {items_value:>15,} GP  ({items_pct:>5.1f}%)")
    print(f"  Active GE Orders:    {orders_value:>15,} GP  ({orders_pct:>5.1f}%)")
    print()

    # Show by item breakdown
    items = util.get('by_item', [])
    if items:
        print(f"Top {min(top_n, len(items))} Items by Value:")
        print(f"{'Item Name':<30} {'Total Value':>15} {'% Portfolio':>12} {'Location':<20}")
        print('-'*80)

        for item in items[:top_n]:
            name = item.get('item_name', 'Unknown')[:29]
            total_val = item.get('total_value', 0)
            pct = item.get('percent_of_portfolio', 0)

            # Build location string
            locations = []
            if item.get('in_inventory', 0) > 0:
                locations.append(f"inv:{item['in_inventory']:,}")
            if item.get('in_bank', 0) > 0:
                locations.append(f"bank:{item['in_bank']:,}")
            if item.get('in_ge_orders', 0) > 0:
                locations.append(f"ge:{item['in_ge_orders']:,}")
            location = ', '.join(locations) if locations else '-'

            print(f"{name:<30} {total_val:>15,} {pct:>11.2f}% {location:<20}")

        if len(items) > top_n:
            print(f"\n  ... and {len(items) - top_n} more items")

    # Summary bar
    print()
    bar_width = 50
    gold_bar = int(gold_pct / 100 * bar_width)
    items_bar = int(items_pct / 100 * bar_width)
    orders_bar = bar_width - gold_bar - items_bar

    print("Portfolio Composition:")
    print(f"  [{'$' * gold_bar}{'#' * items_bar}{'O' * orders_bar}]")
    print(f"   $ = Gold ({gold_pct:.1f}%)  # = Items ({items_pct:.1f}%)  O = GE Orders ({orders_pct:.1f}%)")


def display_limits(scanner: InventoryScanner, max_position_pct: float = 10.0):
    """Display position limits status."""
    sizer = PositionSizer(
        scanner,
        max_position_pct=max_position_pct,
        max_single_order_pct=5.0,
        min_cash_reserve_pct=10.0,
        max_total_exposure_pct=80.0
    )
    sizer.print_limits_status()


def display_positions(scanner: InventoryScanner):
    """Display active PPO positions (tradeable items)."""
    tracker = PositionTracker(scanner.client, scanner.account_id)
    tracker.print_positions()


def display_items(scanner: InventoryScanner, limit: int = 30):
    """Display all tradeable items from Firestore."""
    print_separator("Tradeable Items in Database")

    items = scanner.get_all_tradeable_items()
    print(f"Total tradeable items: {len(items):,}")
    print()

    # Sort by ge_limit descending
    sorted_items = sorted(items.items(), key=lambda x: x[1].get('ge_limit', 0), reverse=True)

    print(f"{'ID':<8} {'Name':<30} {'GE Limit':>10} {'Value':>10} {'Members':>8}")
    print('-'*70)

    for item_id, item_data in sorted_items[:limit]:
        name = item_data.get('name', 'Unknown')[:29]
        ge_limit = item_data.get('ge_limit', 0)
        value = item_data.get('value', 0)
        members = 'Yes' if item_data.get('members', False) else 'No'
        print(f"{item_id:<8} {name:<30} {ge_limit:>10,} {value:>10,} {members:>8}")

    if len(items) > limit:
        print(f"\n  ... and {len(items) - limit} more items")


def display_full_state(scanner: InventoryScanner, pretty: bool = True):
    """Display full inference state as JSON."""
    state = scanner.get_inference_state()

    if pretty:
        print(json.dumps(state, indent=2, default=str))
    else:
        print(json.dumps(state, default=str))


def watch_state(scanner: InventoryScanner, interval: int = 10):
    """Continuously monitor state changes."""
    print(f"Watching state (updating every {interval}s, Ctrl+C to stop)...")
    print()

    last_gold = None
    last_slots = None

    try:
        while True:
            state = scanner.get_inference_state()

            # Check for changes
            gold = state['account']['gold']
            slots = state['ge_slots']['slots_available']
            is_online = state['account']['is_online']

            timestamp = time.strftime('%H:%M:%S')

            if gold != last_gold or slots != last_slots:
                print(f"[{timestamp}] Gold: {gold:,} | Slots: {slots}/8 | Online: {is_online}")

                if last_gold is not None and gold != last_gold:
                    diff = gold - last_gold
                    print(f"           Gold changed: {diff:+,}")

                last_gold = gold
                last_slots = slots
            else:
                print(f"[{timestamp}] No changes (Online: {is_online})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inventory Scanner - View inventory/bank state from Firestore"
    )
    parser.add_argument(
        "--account", "-a",
        default=DEFAULT_ACCOUNT_ID,
        help=f"Account ID (default: {DEFAULT_ACCOUNT_ID})"
    )
    parser.add_argument("--gold", "-g", action="store_true", help="Show gold balance")
    parser.add_argument("--bank", "-b", action="store_true", help="Show bank contents")
    parser.add_argument("--portfolio", "-p", action="store_true", help="Show portfolio utilization breakdown")
    parser.add_argument("--positions", action="store_true", help="Show active PPO positions (tradeable items)")
    parser.add_argument("--limits", "-l", action="store_true", help="Show position limits status")
    parser.add_argument("--max-position", type=float, default=10.0, help="Max position %% per item (default: 10)")
    parser.add_argument("--inventory", "-i", action="store_true", help="Show inventory")
    parser.add_argument("--ge", action="store_true", help="Show GE slot status")
    parser.add_argument("--items", action="store_true", help="Show all tradeable items")
    parser.add_argument("--state", "-s", action="store_true", help="Show full state as JSON")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuously monitor state")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")
    parser.add_argument("--limit", type=int, default=50, help="Limit items displayed")
    parser.add_argument("--min-value", type=int, default=1000, help="Minimum item value for bank display")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        scanner = InventoryScanner(
            service_account_path=get_service_account_path(),
            account_id=args.account
        )

        # Determine what to display
        specific_view = args.gold or args.bank or args.portfolio or args.positions or args.limits or args.inventory or args.ge or args.items or args.state or args.watch

        if args.watch:
            watch_state(scanner, args.interval)
        elif args.state:
            display_full_state(scanner)
        elif args.gold:
            display_gold(scanner)
        elif args.bank:
            display_bank(scanner, min_value=args.min_value, limit=args.limit)
        elif args.portfolio:
            display_portfolio(scanner, top_n=args.limit)
        elif args.positions:
            display_positions(scanner)
        elif args.limits:
            display_limits(scanner, max_position_pct=args.max_position)
        elif args.inventory:
            display_inventory(scanner)
        elif args.ge:
            display_ge_slots(scanner)
        elif args.items:
            display_items(scanner, limit=args.limit)
        else:
            # Default: show summary
            display_summary(scanner)

        scanner.shutdown()

    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
