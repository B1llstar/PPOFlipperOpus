import asyncio
import logging
import unittest
import os
import time
import json
from unittest.mock import MagicMock, patch, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_real_trading_config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_real_trading_config")

# Import required modules
from config import ENV_KWARGS, PPO_KWARGS
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration

class TestRealTradingConfig(unittest.TestCase):
    """Test cases for the Real Trading Mode configuration."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test data
        self.items = {
            "Fire rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Water rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Air rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            }
        }
        
        # Create item lists
        self.item_list = list(self.items.keys())
        self.price_ranges = {item: (self.items[item]['min_price'], self.items[item]['max_price']) for item in self.item_list}
        self.buy_limits = {item: self.items[item]['buy_limit'] for item in self.item_list}
        
        # Create mock objects
        self.real_time_client = MagicMock(spec=RealTimeGrandExchangeClient)
        self.volume_analyzer = MagicMock()
        self.shared_knowledge = MagicMock()
        
        # Save original ENV_KWARGS
        self.original_env_kwargs = ENV_KWARGS.copy()
    
    def tearDown(self):
        """Tear down the test environment."""
        # Restore original ENV_KWARGS
        global ENV_KWARGS
        ENV_KWARGS = self.original_env_kwargs
    
    def test_env_config_training_mode(self):
        """Test environment configuration in training mode."""
        logger.info("Testing environment configuration in training mode")
        
        # Create environment with training mode (default)
        env = GrandExchangeEnv(
            items=self.items,
            real_trading_mode=False,
            **{k: v for k, v in ENV_KWARGS.items() if k != 'items'}
        )
        
        # Verify that the environment is in training mode
        self.assertFalse(env.real_trading_mode)
        self.assertIsNone(env.real_trading_client)
        
        # Verify that the environment uses simulated market data
        self.assertTrue(hasattr(env, '_fluctuate_prices'))
        
        # Verify that the environment uses simulated order fulfillment
        self.assertTrue(hasattr(env, '_fulfill_orders'))
    
    def test_env_config_real_trading_mode(self):
        """Test environment configuration in real trading mode."""
        logger.info("Testing environment configuration in real trading mode")
        
        # Create environment with real trading mode
        env = GrandExchangeEnv(
            items=self.items,
            real_trading_mode=True,
            real_trading_client=self.real_time_client,
            volume_analyzer=self.volume_analyzer,
            **{k: v for k, v in ENV_KWARGS.items() if k != 'items'}
        )
        
        # Verify that the environment is in real trading mode
        self.assertTrue(env.real_trading_mode)
        self.assertEqual(env.real_trading_client, self.real_time_client)
        
        # Verify that the environment has the volume analyzer
        self.assertEqual(env.volume_analyzer, self.volume_analyzer)
    
    def test_agent_config_training_mode(self):
        """Test agent configuration in training mode."""
        logger.info("Testing agent configuration in training mode")
        
        # Create agent with training mode (default)
        agent = PPOAgent(
            item_list=self.item_list,
            price_ranges=self.price_ranges,
            buy_limits=self.buy_limits,
            device="cpu",
            **{k: v for k, v in PPO_KWARGS.items() if k not in ['item_list', 'price_ranges', 'buy_limits', 'device']}
        )
        
        # Verify that the agent does not have volume analyzer or shared knowledge
        self.assertIsNone(agent.volume_analyzer)
        self.assertIsNone(agent.shared_knowledge)
    
    def test_agent_config_real_trading_mode(self):
        """Test agent configuration in real trading mode."""
        logger.info("Testing agent configuration in real trading mode")
        
        # Create agent with real trading mode
        agent = PPOAgent(
            item_list=self.item_list,
            price_ranges=self.price_ranges,
            buy_limits=self.buy_limits,
            device="cpu",
            volume_analyzer=self.volume_analyzer,
            shared_knowledge=self.shared_knowledge,
            **{k: v for k, v in PPO_KWARGS.items() if k not in ['item_list', 'price_ranges', 'buy_limits', 'device']}
        )
        
        # Verify that the agent has volume analyzer and shared knowledge
        self.assertEqual(agent.volume_analyzer, self.volume_analyzer)
        self.assertEqual(agent.shared_knowledge, self.shared_knowledge)
    
    def test_env_kwargs_override(self):
        """Test overriding ENV_KWARGS for real trading mode."""
        logger.info("Testing ENV_KWARGS override for real trading mode")
        
        # Create a copy of ENV_KWARGS
        env_kwargs = dict(ENV_KWARGS)
        
        # Override for real trading mode
        env_kwargs["items"] = self.items
        env_kwargs["real_trading_mode"] = True
        env_kwargs["real_trading_client"] = self.real_time_client
        env_kwargs["volume_analyzer"] = self.volume_analyzer
        
        # Create environment with overridden kwargs
        env = GrandExchangeEnv(**env_kwargs)
        
        # Verify that the environment is in real trading mode
        self.assertTrue(env.real_trading_mode)
        self.assertEqual(env.real_trading_client, self.real_time_client)
        self.assertEqual(env.volume_analyzer, self.volume_analyzer)
    
    def test_ppo_kwargs_override(self):
        """Test overriding PPO_KWARGS for real trading mode."""
        logger.info("Testing PPO_KWARGS override for real trading mode")
        
        # Create a copy of PPO_KWARGS
        ppo_kwargs = dict(PPO_KWARGS)
        
        # Add real trading mode specific parameters
        ppo_kwargs["item_list"] = self.item_list
        ppo_kwargs["price_ranges"] = self.price_ranges
        ppo_kwargs["buy_limits"] = self.buy_limits
        ppo_kwargs["device"] = "cpu"
        ppo_kwargs["volume_analyzer"] = self.volume_analyzer
        ppo_kwargs["shared_knowledge"] = self.shared_knowledge
        
        # Create agent with overridden kwargs
        agent = PPOAgent(**ppo_kwargs)
        
        # Verify that the agent has volume analyzer and shared knowledge
        self.assertEqual(agent.volume_analyzer, self.volume_analyzer)
        self.assertEqual(agent.shared_knowledge, self.shared_knowledge)
    
    def test_config_switching(self):
        """Test switching between training and real trading configurations."""
        logger.info("Testing switching between training and real trading configurations")
        
        # Create training environment
        training_env = GrandExchangeEnv(
            items=self.items,
            real_trading_mode=False,
            **{k: v for k, v in ENV_KWARGS.items() if k != 'items'}
        )
        
        # Create training agent
        training_agent = PPOAgent(
            item_list=self.item_list,
            price_ranges=self.price_ranges,
            buy_limits=self.buy_limits,
            device="cpu",
            **{k: v for k, v in PPO_KWARGS.items() if k not in ['item_list', 'price_ranges', 'buy_limits', 'device']}
        )
        
        # Verify training configuration
        self.assertFalse(training_env.real_trading_mode)
        self.assertIsNone(training_env.real_trading_client)
        self.assertIsNone(training_agent.volume_analyzer)
        self.assertIsNone(training_agent.shared_knowledge)
        
        # Create real trading environment
        real_env = GrandExchangeEnv(
            items=self.items,
            real_trading_mode=True,
            real_trading_client=self.real_time_client,
            volume_analyzer=self.volume_analyzer,
            **{k: v for k, v in ENV_KWARGS.items() if k != 'items'}
        )
        
        # Create real trading agent
        real_agent = PPOAgent(
            item_list=self.item_list,
            price_ranges=self.price_ranges,
            buy_limits=self.buy_limits,
            device="cpu",
            volume_analyzer=self.volume_analyzer,
            shared_knowledge=self.shared_knowledge,
            **{k: v for k, v in PPO_KWARGS.items() if k not in ['item_list', 'price_ranges', 'buy_limits', 'device']}
        )
        
        # Verify real trading configuration
        self.assertTrue(real_env.real_trading_mode)
        self.assertEqual(real_env.real_trading_client, self.real_time_client)
        self.assertEqual(real_env.volume_analyzer, self.volume_analyzer)
        self.assertEqual(real_agent.volume_analyzer, self.volume_analyzer)
        self.assertEqual(real_agent.shared_knowledge, self.shared_knowledge)
        
        # Test loading weights from training to real trading
        with patch.object(real_agent, 'load_actor') as mock_load_actor, \
             patch.object(real_agent, 'load_critic') as mock_load_critic:
            
            # Mock the existence of model files
            with patch('os.path.exists', return_value=True):
                # Load weights
                real_agent.load_actor("actor_best.pth")
                real_agent.load_critic("critic_best.pth")
                
                # Verify that the load methods were called
                mock_load_actor.assert_called_once_with("actor_best.pth")
                mock_load_critic.assert_called_once_with("critic_best.pth")

class TestAsyncRealTradingConfig(unittest.TestCase):
    """Test cases for the asynchronous aspects of Real Trading Mode configuration."""
    
    async def async_setup(self):
        """Set up the test environment asynchronously."""
        # Create test data
        self.items = {
            "Fire rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Water rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Air rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            }
        }
        
        # Create mock objects
        self.websocket_integration = MagicMock(spec=PPOWebSocketIntegration)
        self.websocket_integration.connect = AsyncMock(return_value=True)
        self.websocket_integration.disconnect = AsyncMock(return_value=None)
        self.websocket_integration.system_status = AsyncMock(return_value=True)
        
        self.real_time_client = MagicMock(spec=RealTimeGrandExchangeClient)
        self.real_time_client.start = AsyncMock(return_value=None)
        self.real_time_client.stop = AsyncMock(return_value=None)
        self.real_time_client.update_market_data = AsyncMock(return_value=True)
        
        # Create a mock for GrandExchangeEnv
        self.env_mock = MagicMock(spec=GrandExchangeEnv)
        self.env_mock.reset.return_value = {
            'gp': 1000000,
            'inventory': {'Fire rune': 0, 'Water rune': 0, 'Air rune': 0},
            'prices': {'Fire rune': 5, 'Water rune': 5, 'Air rune': 5}
        }
        self.env_mock.step.return_value = (
            {
                'gp': 999995,
                'inventory': {'Fire rune': 1, 'Water rune': 0, 'Air rune': 0},
                'prices': {'Fire rune': 5, 'Water rune': 5, 'Air rune': 5}
            },
            0.0,
            False,
            {}
        )
        self.env_mock.update_real_market_data = AsyncMock(return_value=None)
        self.env_mock._update_real_orders = MagicMock()
        
        # Create a mock for PPOAgent
        self.agent_mock = MagicMock(spec=PPOAgent)
        self.agent_mock.sample_action.return_value = (
            {
                'type': 'buy',
                'item': 'Fire rune',
                'price': 5,
                'quantity': 1
            },
            None
        )
    
    async def async_teardown(self):
        """Tear down the test environment asynchronously."""
        pass
    
    async def test_real_trading_integration(self):
        """Test the integration of all components in real trading mode."""
        logger.info("Testing real trading integration")
        
        # Set up the test environment
        await self.async_setup()
        
        # Create a mock for asyncio.sleep to avoid waiting
        with patch('asyncio.sleep', AsyncMock(return_value=None)):
            # Create a mock for the trading loop to exit after one iteration
            def side_effect(*args, **kwargs):
                # Raise KeyboardInterrupt after one iteration to exit the loop
                self.env_mock.step.side_effect = KeyboardInterrupt()
                return self.env_mock.step.return_value
            
            self.env_mock.step = MagicMock(side_effect=side_effect)
            
            # Run a simplified version of the real trading loop
            try:
                # Connect to WebSocket server
                await self.websocket_integration.connect()
                
                # Start real-time client
                await self.real_time_client.start()
                
                # Send system status event
                await self.websocket_integration.system_status(
                    "real_trading_started",
                    "Real trading mode started"
                )
                
                # Reset environment
                obs = self.env_mock.reset()
                
                # Hook agent and environment to WebSocket integration
                self.websocket_integration.hook_agent(self.agent_mock, agent_id="real_trading_agent")
                self.websocket_integration.hook_env(self.env_mock, env_id="real_trading_env")
                
                # Main trading loop
                while True:
                    # Sample action from agent
                    action, _ = self.agent_mock.sample_action(obs)
                    
                    # Execute action in environment
                    next_obs, reward, done, info = self.env_mock.step(action)
                    
                    # Update observation
                    obs = next_obs
                    
                    # Wait for next trading interval
                    await asyncio.sleep(1)  # 1 second for testing
                    
                    # Update market data
                    await self.env_mock.update_real_market_data()
                    
                    # Check for any completed orders
                    self.env_mock._update_real_orders()
            except KeyboardInterrupt:
                pass
            finally:
                # Clean up
                self.websocket_integration.unhook_agent("real_trading_agent")
                self.websocket_integration.unhook_env("real_trading_env")
                await self.real_time_client.stop()
                await self.websocket_integration.disconnect()
            
            # Verify that the WebSocket integration was connected
            self.websocket_integration.connect.assert_called_once()
            
            # Verify that the real-time client was started
            self.real_time_client.start.assert_called_once()
            
            # Verify that the environment was reset
            self.env_mock.reset.assert_called_once()
            
            # Verify that the agent was hooked to the WebSocket integration
            self.websocket_integration.hook_agent.assert_called_once()
            
            # Verify that the environment was hooked to the WebSocket integration
            self.websocket_integration.hook_env.assert_called_once()
            
            # Verify that the agent's sample_action was called
            self.agent_mock.sample_action.assert_called_once()
            
            # Verify that the environment's step was called
            self.env_mock.step.assert_called_once()
            
            # Verify that the WebSocket integration was disconnected
            self.websocket_integration.disconnect.assert_called_once()
            
            # Verify that the real-time client was stopped
            self.real_time_client.stop.assert_called_once()
        
        # Tear down the test environment
        await self.async_teardown()

def run_tests():
    """Run the synchronous tests."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add the tests
    suite.addTest(unittest.makeSuite(TestRealTradingConfig))
    
    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)

async def run_async_tests():
    """Run the asynchronous tests."""
    test_case = TestAsyncRealTradingConfig()
    await test_case.test_real_trading_integration()

if __name__ == "__main__":
    # Run the synchronous tests
    run_tests()
    
    # Run the asynchronous tests
    asyncio.run(run_async_tests())