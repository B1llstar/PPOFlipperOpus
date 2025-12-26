import unittest
from ppo_websocket_integration import validate_trade
from decimal import Decimal

class TestTradeValidation(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.obs = {
            'prices': {
                'item1': 1000,
                'item2': 500,
            },
            'inventory': {
                'item1': 10,
                'item2': 5,
            },
            'gp': 10000
        }

    def test_buy_validation_profit_rule(self):
        # Test case where profit is exactly 2%
        action = {
            'type': 'buy',
            'item': 'item1',
            'price': 980,  # Current price 1000, 2% profit after 1% tax
            'quantity': 1
        }
        is_valid, reason = validate_trade(action, self.obs)
        self.assertTrue(is_valid, f"Trade should be valid: {reason}")

        # Test case where profit is less than 2%
        action['price'] = 990
        is_valid, reason = validate_trade(action, self.obs)
        self.assertFalse(is_valid, "Trade should be invalid due to insufficient profit margin")

    def test_buy_validation_affordability(self):
        # Test case where trade is affordable
        action = {
            'type': 'buy',
            'item': 'item1',
            'price': 900,
            'quantity': 10
        }
        is_valid, reason = validate_trade(action, self.obs)
        self.assertTrue(is_valid, f"Trade should be valid: {reason}")

        # Test case where trade is not affordable
        action['quantity'] = 12
        is_valid, reason = validate_trade(action, self.obs)
        self.assertFalse(is_valid, "Trade should be invalid due to insufficient funds")

    def test_sell_validation_inventory(self):
        # Test case with sufficient inventory
        action = {
            'type': 'sell',
            'item': 'item1',
            'price': 1000,
            'quantity': 5
        }
        is_valid, reason = validate_trade(action, self.obs)
        self.assertTrue(is_valid, f"Trade should be valid: {reason}")

        # Test case with insufficient inventory
        action['quantity'] = 15
        is_valid, reason = validate_trade(action, self.obs)
        self.assertFalse(is_valid, "Trade should be invalid due to insufficient inventory")

    def test_tax_calculation(self):
        # Test correct tax calculation and profit margin with tax
        action = {
            'type': 'buy',
            'item': 'item1',
            'price': 970,  # Current price 1000
            'quantity': 1
        }
        # Tax will be 10 (1% of 1000), so need price of 970 or less for 2% profit
        is_valid, reason = validate_trade(action, self.obs)
        self.assertTrue(is_valid, f"Trade should be valid with correct tax calculation: {reason}")

        action['price'] = 971
        is_valid, reason = validate_trade(action, self.obs)
        self.assertFalse(is_valid, "Trade should be invalid when tax makes profit below 2%")

    def test_hold_validation(self):
        # Test that hold actions are always valid
        action = {
            'type': 'hold',
            'item': 'item1',
            'price': 1000,
            'quantity': 1
        }
        is_valid, reason = validate_trade(action, self.obs)
        self.assertTrue(is_valid, "Hold actions should always be valid")

if __name__ == '__main__':
    unittest.main()