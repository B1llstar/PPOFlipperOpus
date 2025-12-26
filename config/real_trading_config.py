import logging
import json
import os
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_trading_config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("real_trading_config")

# Import config
try:
    from config import ENV_KWARGS
    logger.info("Successfully imported config")
except ImportError as e:
    logger.error(f"Error importing config: {str(e)}")
    raise

class RealTradingConfig:
    """
    Configuration manager for real-time trading mode.
    
    This class handles enabling/disabling real trading mode and
    configuring the marketplace connection settings.
    """
    
    def __init__(self, config_path: str = "real_trading_config.json"):
        """
        Initialize the real trading configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Default configuration
        self.default_config = {
            "real_trading_mode": False,
            "websocket_url": "ws://localhost:8080/marketplace/ws",
            "max_slots": 8,
            "experiment_timeout": 300,  # 5 minutes in seconds
            "final_timeout": 600,       # 10 minutes in seconds
            "update_interval": 300,     # 5 minutes in seconds
            "agent_id": "ppo_agent"
        }
        
        # Ensure all default keys exist in config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dict: Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return {}
        else:
            logger.info(f"Configuration file {self.config_path} not found, using defaults")
            return {}
    
    def _save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def enable_real_trading(self) -> bool:
        """
        Enable real trading mode.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Update config
        self.config["real_trading_mode"] = True
        
        # Update ENV_KWARGS
        ENV_KWARGS["real_trading_mode"] = True
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info("Real trading mode enabled")
        else:
            logger.error("Failed to enable real trading mode")
        
        return result
    
    def disable_real_trading(self) -> bool:
        """
        Disable real trading mode.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Update config
        self.config["real_trading_mode"] = False
        
        # Update ENV_KWARGS
        ENV_KWARGS["real_trading_mode"] = False
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info("Real trading mode disabled")
        else:
            logger.error("Failed to disable real trading mode")
        
        return result
    
    def is_real_trading_enabled(self) -> bool:
        """
        Check if real trading mode is enabled.
        
        Returns:
            bool: True if enabled, False otherwise
        """
        return self.config.get("real_trading_mode", False)
    
    def set_websocket_url(self, url: str) -> bool:
        """
        Set the WebSocket URL for the marketplace.
        
        Args:
            url: WebSocket URL
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update config
        self.config["websocket_url"] = url
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info(f"WebSocket URL set to {url}")
        else:
            logger.error(f"Failed to set WebSocket URL to {url}")
        
        return result
    
    def get_websocket_url(self) -> str:
        """
        Get the WebSocket URL for the marketplace.
        
        Returns:
            str: WebSocket URL
        """
        return self.config.get("websocket_url", self.default_config["websocket_url"])
    
    def set_max_slots(self, max_slots: int) -> bool:
        """
        Set the maximum number of order slots.
        
        Args:
            max_slots: Maximum number of slots
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate input
        if max_slots <= 0:
            logger.error(f"Invalid max_slots value: {max_slots}")
            return False
        
        # Update config
        self.config["max_slots"] = max_slots
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info(f"Max slots set to {max_slots}")
        else:
            logger.error(f"Failed to set max slots to {max_slots}")
        
        return result
    
    def get_max_slots(self) -> int:
        """
        Get the maximum number of order slots.
        
        Returns:
            int: Maximum number of slots
        """
        return self.config.get("max_slots", self.default_config["max_slots"])
    
    def set_timeouts(self, experiment_timeout: int, final_timeout: int) -> bool:
        """
        Set the timeouts for orders.
        
        Args:
            experiment_timeout: Timeout for experimentation phase in seconds
            final_timeout: Timeout for final phase in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate input
        if experiment_timeout <= 0 or final_timeout <= 0:
            logger.error(f"Invalid timeout values: experiment={experiment_timeout}, final={final_timeout}")
            return False
        
        # Update config
        self.config["experiment_timeout"] = experiment_timeout
        self.config["final_timeout"] = final_timeout
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info(f"Timeouts set to: experiment={experiment_timeout}s, final={final_timeout}s")
        else:
            logger.error(f"Failed to set timeouts to: experiment={experiment_timeout}s, final={final_timeout}s")
        
        return result
    
    def get_experiment_timeout(self) -> int:
        """
        Get the timeout for experimentation phase.
        
        Returns:
            int: Timeout in seconds
        """
        return self.config.get("experiment_timeout", self.default_config["experiment_timeout"])
    
    def get_final_timeout(self) -> int:
        """
        Get the timeout for final phase.
        
        Returns:
            int: Timeout in seconds
        """
        return self.config.get("final_timeout", self.default_config["final_timeout"])
    
    def set_update_interval(self, interval: int) -> bool:
        """
        Set the update interval for market data.
        
        Args:
            interval: Update interval in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate input
        if interval <= 0:
            logger.error(f"Invalid update interval: {interval}")
            return False
        
        # Update config
        self.config["update_interval"] = interval
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info(f"Update interval set to {interval}s")
        else:
            logger.error(f"Failed to set update interval to {interval}s")
        
        return result
    
    def get_update_interval(self) -> int:
        """
        Get the update interval for market data.
        
        Returns:
            int: Update interval in seconds
        """
        return self.config.get("update_interval", self.default_config["update_interval"])
    
    def set_agent_id(self, agent_id: str) -> bool:
        """
        Set the agent ID for the marketplace.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update config
        self.config["agent_id"] = agent_id
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info(f"Agent ID set to {agent_id}")
        else:
            logger.error(f"Failed to set agent ID to {agent_id}")
        
        return result
    
    def get_agent_id(self) -> str:
        """
        Get the agent ID for the marketplace.
        
        Returns:
            str: Agent ID
        """
        return self.config.get("agent_id", self.default_config["agent_id"])
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration settings.
        
        Returns:
            Dict: All configuration settings
        """
        return self.config.copy()
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all configuration settings to defaults.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Update config
        self.config = self.default_config.copy()
        
        # Update ENV_KWARGS
        ENV_KWARGS["real_trading_mode"] = self.config["real_trading_mode"]
        
        # Save config
        result = self._save_config()
        
        if result:
            logger.info("Configuration reset to defaults")
        else:
            logger.error("Failed to reset configuration to defaults")
        
        return result


# Example usage
if __name__ == "__main__":
    # Create config manager
    config_manager = RealTradingConfig()
    
    # Print current configuration
    print("Current configuration:")
    print(json.dumps(config_manager.get_all_config(), indent=4))
    
    # Check if real trading is enabled
    if config_manager.is_real_trading_enabled():
        print("Real trading mode is enabled")
    else:
        print("Real trading mode is disabled")
    
    # Enable real trading mode
    config_manager.enable_real_trading()
    
    # Print updated configuration
    print("\nUpdated configuration:")
    print(json.dumps(config_manager.get_all_config(), indent=4))