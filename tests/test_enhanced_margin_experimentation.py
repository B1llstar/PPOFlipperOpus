import unittest
import time
from enhanced_margin_experimentation import EnhancedMarginExperimentation

class TestEnhancedMarginExperimentation(unittest.TestCase):
    def setUp(self):
        self.margin_exp = EnhancedMarginExperimentation()
        self.test_item_id = 1234
        
    def test_initial_margins(self):
        """Test initial margin ranges are correct"""
        min_margin, max_margin = self.margin_exp.calculate_dynamic_margin(self.test_item_id, 'buy')
        self.assertAlmostEqual(max_margin, 0.40)  # Should start at 40%
        self.assertGreaterEqual(min_margin, 0.02)  # Should not go below 2%
        
    def test_success_rate_tracking(self):
        """Test success rate tracking with action types"""
        # Track some buy successes
        self.margin_exp.update_success_rate(0.10, 'buy', True, 1000, "Test success")
        self.margin_exp.update_success_rate(0.10, 'buy', True, 1100, "Test success")
        self.margin_exp.update_success_rate(0.10, 'buy', False, 1200, "Test failure")
        
        # Check success rate calculation
        success_rate = self.margin_exp._get_success_rate('buy')
        self.assertAlmostEqual(success_rate, 2/3)
        
    def test_adaptive_wait_time(self):
        """Test adaptive wait time calculation with neural network input"""
        # Test without neural network input
        wait_time = self.margin_exp.get_adaptive_wait_time(self.test_item_id)
        self.assertGreaterEqual(wait_time, 15)  # Minimum wait time
        self.assertLessEqual(wait_time, 600)  # Maximum wait time (10 minutes)
        
        # Test with neural network input
        wait_time_nn = self.margin_exp.get_adaptive_wait_time(self.test_item_id, nn_wait_steps=8)
        self.assertGreaterEqual(wait_time_nn, 30)  # Higher minimum with NN
        self.assertLessEqual(wait_time_nn, 600)  # Same maximum
        
    def test_order_timeout(self):
        """Test order timeout handling"""
        # Create a test order
        order_id = "test_order_1"
        self.margin_exp.track_order_attempt(order_id, self.test_item_id, 'buy', 1000, 0.10)
        
        # Check normal trading timeout (10 minutes)
        self.assertFalse(self.margin_exp.handle_order_timeout(order_id, self.test_item_id, is_experiment=False))
        
        # Simulate time passing
        self.margin_exp.order_attempts[order_id]['start_time'] -= 601  # 10 minutes + 1 second
        self.assertTrue(self.margin_exp.handle_order_timeout(order_id, self.test_item_id, is_experiment=False))
        
        # Check experiment timeout (5 minutes)
        order_id = "test_order_2"
        self.margin_exp.track_order_attempt(order_id, self.test_item_id, 'buy', 1000, 0.10)
        self.margin_exp.order_attempts[order_id]['start_time'] -= 301  # 5 minutes + 1 second
        self.assertTrue(self.margin_exp.handle_order_timeout(order_id, self.test_item_id, is_experiment=True))
        
    def test_order_relisting(self):
        """Test order relisting logic"""
        # Add some success/failure history
        self.margin_exp.update_success_rate(0.40, 'buy', False, 1000, "Initial fail")
        self.margin_exp.update_success_rate(0.30, 'buy', False, 1000, "Second fail")
        self.margin_exp.update_success_rate(0.20, 'buy', True, 1000, "Success at lower margin")
        
        # Test relisting with low success rate
        new_price, new_margin = self.margin_exp.relist_order(self.test_item_id, 1000, 0.40)
        self.assertLess(new_margin, 0.40)  # Should reduce margin after failures
        self.assertGreaterEqual(new_margin, 0.02)  # Should not go below minimum
        
    def test_volume_based_adjustments(self):
        """Test volume-based margin adjustments"""
        # Mock volume data
        def mock_volume_data(item_id):
            return {'recent_volume': 25000, 'momentum_1h': 0.5}
        self.margin_exp._fetch_volume_data = mock_volume_data
        
        min_margin, max_margin = self.margin_exp.calculate_dynamic_margin(self.test_item_id, 'buy')
        self.assertLess(max_margin, 0.40)  # Should be reduced due to volume
        self.assertGreaterEqual(min_margin, 0.02)  # Should not go below minimum

if __name__ == '__main__':
    unittest.main()