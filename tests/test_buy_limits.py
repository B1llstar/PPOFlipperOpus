import unittest
import time
import json
from shared_knowledge import SharedKnowledgeRepository

class TestBuyLimits(unittest.TestCase):
    def setUp(self):
        # Create simple mapping dictionaries for testing
        self.id_to_name = {
            123: "Test Item",
            456: "Test Item 2"
        }
        self.name_to_id = {
            "Test Item": 123,
            "Test Item 2": 456
        }
        
        # Create test buy_limits.json
        self.test_limits = {
            "Test Item": 100,
            "Test Item 2": 50
        }
        with open('buy_limits.json', 'w') as f:
            json.dump(self.test_limits, f)
        
        self.repo = SharedKnowledgeRepository(self.id_to_name, self.name_to_id)

    def test_buy_limit_tracking(self):
        current_time = int(time.time())
        
        # Should allow purchases within limit
        self.assertTrue(self.repo._check_buy_limit(123, 50, current_time))
        self.repo.record_trade(1, "Test Item", "buy", 100, 50, 0, timestamp=current_time)
        
        # Should allow up to limit
        self.assertTrue(self.repo._check_buy_limit(123, 50, current_time))
        self.repo.record_trade(1, "Test Item", "buy", 100, 50, 0, timestamp=current_time)
        
        # Should reject purchase exceeding limit
        self.assertFalse(self.repo._check_buy_limit(123, 1, current_time))
        
        # Should allow purchase after 4 hours
        future_time = current_time + (4 * 60 * 60) + 1
        self.assertTrue(self.repo._check_buy_limit(123, 50, future_time))
        
    def test_remaining_limit(self):
        current_time = int(time.time())
        
        # Full limit available initially
        self.assertEqual(self.repo.get_remaining_limit(123), 100)
        
        # Record some purchases
        self.repo.record_trade(1, "Test Item", "buy", 100, 30, 0, timestamp=current_time)
        self.assertEqual(self.repo.get_remaining_limit(123), 70)
        
        self.repo.record_trade(1, "Test Item", "buy", 100, 20, 0, timestamp=current_time)
        self.assertEqual(self.repo.get_remaining_limit(123), 50)
        
        # Check limit resets after 4 hours
        future_time = current_time + (4 * 60 * 60) + 1
        self.repo._clean_old_purchases(123, future_time)
        self.assertEqual(self.repo.get_remaining_limit(123), 100)
        
    def test_no_limit_items(self):
        # Test items not in buy_limits.json have no restrictions
        self.assertTrue(self.repo._check_buy_limit(999, 999999, int(time.time())))
        self.assertIsNone(self.repo.get_remaining_limit(999))

    def test_env_buy_limit_reset(self):
        # Create a dummy items dictionary
        items = {
            "Test Item": {"base_price": 100, "buy_limit": 100, "min_price": 50, "max_price": 150},
            "Test Item 2": {"base_price": 50, "buy_limit": 50, "min_price": 25, "max_price": 75}
        }

        # Initialize GrandExchangeEnv with a small reset interval
        from ge_env import GrandExchangeEnv
        env = GrandExchangeEnv(items=items, buy_limit_reset_ticks=10, starting_gp=1000000)

        # Step the environment and place buy orders
        obs = env.reset()
        item_to_buy = "Test Item"
        buy_quantity = 20

        # Place buy orders over several steps within the reset interval
        for i in range(5):
            action = {'type': 'buy', 'item': item_to_buy, 'price': 100, 'quantity': buy_quantity}
            obs, reward, done, info = env.step(action)
            # Check that buy limit is accumulating
            self.assertEqual(obs['buy_limits'][item_to_buy], (i + 1) * buy_quantity)

        # Step beyond the reset interval
        for i in range(6):
            action = {'type': 'hold', 'item': item_to_buy, 'price': 0, 'quantity': 0}
            obs, reward, done, info = env.step(action)

        # Place a buy order after the reset interval and check if the buy limit is reset
        action = {'type': 'buy', 'item': item_to_buy, 'price': 100, 'quantity': buy_quantity}
        obs, reward, done, info = env.step(action)
        # The buy limit should be reset and show only the quantity from the new buy order
        self.assertEqual(obs['buy_limits'][item_to_buy], buy_quantity)

        # Verify log messages (optional, but good for debugging)
        with open('ge_env.log', 'r') as f:
            log_content = f.read()
            self.assertIn(f"Tick {env.tick}: Buy limit for {item_to_buy} reset after 10 ticks.", log_content)
            self.assertIn(f"Tick {env.tick}: After buy order for {item_to_buy}, buy limit is now: {buy_quantity}", log_content)

if __name__ == '__main__':
    unittest.main()