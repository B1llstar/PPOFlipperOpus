import unittest
import logging
from ge_env import GrandExchangeEnv
from volume_analysis import VolumeAnalyzer
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO)

class TestGEEnvVolumeIntegration(unittest.TestCase):
    def setUp(self):
        # Set up mock volume analyzer
        self.volume_analyzer = MagicMock()
        self.volume_metrics = {
            'Dragon bones': {
                'recent_volume': 50000,
                'volume_momentum': 0.15,
                'buy_sell_imbalance': 0.2,
                'market_activity': 0.75
            },
            'Nature rune': {
                'recent_volume': 500,  # Low volume
                'volume_momentum': -0.3,  # Declining volume
                'buy_sell_imbalance': -0.4,  # Heavy selling
                'market_activity': 0.2
            }
        }
        
        # Mock volume metrics response
        def mock_get_metrics(analyzer, item):
            return self.volume_metrics.get(item, {})
        
        with patch('ge_env.get_volume_metrics_for_item', side_effect=mock_get_metrics):
            # Initialize environment with test items
            self.items = {
                'Dragon bones': {
                    'base_price': 3000,
                    'buy_limit': 100,
                    'min_price': 2000,
                    'max_price': 4000
                },
                'Nature rune': {
                    'base_price': 300,
                    'buy_limit': 1000,
                    'min_price': 200,
                    'max_price': 400
                }
            }
            self.env = GrandExchangeEnv(
                items=self.items,
                starting_gp=1000000,
                volume_analyzer=self.volume_analyzer
            )

    def test_volume_validation_buy_order(self):
        """Test that buy orders are rejected for low volume items"""
        # Try to buy low-volume Nature runes
        action = {
            'type': 'buy',
            'item': 'Nature rune',
            'price': 300,
            'quantity': 100
        }
        _, _, _, info = self.env.step(action)
        self.assertIn('Volume too low', info['msg'])
        
        # Try to buy good volume Dragon bones
        action = {
            'type': 'buy',
            'item': 'Dragon bones',
            'price': 3000,
            'quantity': 10
        }
        _, _, _, info = self.env.step(action)
        self.assertNotIn('Volume too low', info.get('msg', ''))

    def test_volume_based_rewards(self):
        """Test that rewards are adjusted based on volume metrics"""
        # Buy high volume item with positive momentum
        action = {
            'type': 'buy',
            'item': 'Dragon bones',
            'price': 3000,
            'quantity': 10
        }
        obs, reward, _, _ = self.env.step(action)
        first_reward = reward

        # Buy low volume item with negative momentum
        action = {
            'type': 'buy',
            'item': 'Nature rune',
            'price': 300,
            'quantity': 100
        }
        obs, reward, _, _ = self.env.step(action)
        second_reward = reward

        # First reward should be higher due to better volume metrics
        self.assertGreater(first_reward, second_reward)

    def test_volume_metrics_in_obs(self):
        """Test that volume metrics are included in observations"""
        obs = self.env._get_obs()
        
        # Check volume metrics exist in observation
        self.assertIn('volume_metrics', obs)
        
        # Check all expected metrics are present
        metrics = obs['volume_metrics']
        self.assertIn('volume', metrics)
        self.assertIn('momentum', metrics)
        self.assertIn('imbalance', metrics)
        self.assertIn('activity', metrics)
        
        # Verify metrics are normalized
        for item in self.items:
            self.assertGreaterEqual(metrics['volume'][item], 0.0)
            self.assertLessEqual(metrics['volume'][item], 1.0)
            self.assertGreaterEqual(metrics['momentum'][item], -1.0)
            self.assertLessEqual(metrics['momentum'][item], 1.0)

    def test_volume_trend_validation(self):
        """Test that orders are validated against volume trends"""
        # Try to buy with heavy selling pressure
        action = {
            'type': 'buy',
            'item': 'Nature rune',
            'price': 300,
            'quantity': 100
        }
        _, _, _, info = self.env.step(action)
        self.assertIn('selling pressure', info['msg'].lower())
        
        # Try to buy with declining volume
        self.volume_metrics['Dragon bones']['volume_momentum'] = -0.3
        action = {
            'type': 'buy',
            'item': 'Dragon bones',
            'price': 3000,
            'quantity': 10
        }
        _, _, _, info = self.env.step(action)
        self.assertIn('declining volume', info['msg'].lower())

if __name__ == '__main__':
    unittest.main()