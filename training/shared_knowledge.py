import numpy as np
import time
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from training.volume_analysis import VolumeAnalyzer

# Set up module-level logger
logger = logging.getLogger("shared_knowledge")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("shared_knowledge.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(fh)
logger.propagate = False
logger.info("shared_knowledge logger initialized (imported)")

class SharedKnowledgeRepository:
    """
    Centralized repository for sharing trading knowledge across multiple agents.
    Consolidates experiences, metrics, and insights to accelerate collective learning.
    """
    
    def __init__(self):
        """
        Initialize the SharedKnowledgeRepository.
        Item ID and name mappings are loaded from 'endpoints/mapping.txt'.
        """
        # Import os for file operations
        import os
        
        # Load mappings from mapping.txt
        name_to_id_map = {}
        id_to_name_map = {}
        
        try:
            with open('endpoints/mapping.txt', 'r') as f:
                mapping_data = json.load(f)
                
                # Process each item in the mapping
                for item in mapping_data:
                    if 'name' in item and 'id' in item:
                        name = item['name']
                        item_id = item['id']
                        name_to_id_map[name] = item_id
                        id_to_name_map[str(item_id)] = name
            logger.info(f"Successfully loaded {len(id_to_name_map)} items from endpoints/mapping.txt")
        except FileNotFoundError:
            logger.error("endpoints/mapping.txt not found. Item mappings will be empty.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from endpoints/mapping.txt. Item mappings will be empty.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading item mappings: {e}")

        # Store mappings
        self.id_to_name_map = id_to_name_map
        self.name_to_id_map = name_to_id_map

        # Shared volume analyzer (single source of truth for market data)
        self.volume_analyzer = VolumeAnalyzer(self.id_to_name_map, self.name_to_id_map)
        
        # Consolidated trade history across all agents
        self.trade_history = {}  # item_id -> list of trade records
        
        # Aggregated profit metrics
        self.profit_metrics = {}  # item_id -> aggregated profit metrics
        
        # Success rate tracking
        self.success_rates = {}  # item_id -> success rate
        
        # Agent specialization tracking
        self.agent_specializations = {}  # agent_id -> {item_id -> performance score}
        
        # Consensus trading signals
        self.consensus_signals = {}  # item_id -> consensus signal
        
        # Maximum history length to prevent memory issues
        self.max_history_length = 1000  # Maximum trades to store per item
        
        # Buy limit tracking
        self.buy_limits = {}  # item_id -> buy limit
        self.purchase_history = {}  # item_id -> list of (timestamp, quantity) tuples
        
        # Load buy limits from json
        try:
            with open('buy_limits.json', 'r') as f:
                name_to_limit = json.load(f)
                for item_name, limit in name_to_limit.items():
                    if item_name in self.name_to_id_map:
                        item_id = self.name_to_id_map[item_name]
                        self.buy_limits[item_id] = limit
        except Exception as e:
            logger.error(f"Error loading buy limits: {e}")
        
        # Load trade history from archives
        # Method is implemented but skips loading archive data since we only care about current episode
        self.load_trade_history_from_archives()
            
        logger.info(f"SharedKnowledgeRepository initialized with {len(id_to_name_map)} items and {len(self.buy_limits)} buy limits")
    
    def record_trade(self, agent_id, item_name, action_type, price, quantity, profit, tax=0, timestamp=None):
        """
        Record a trade in the shared repository
        
        Args:
            agent_id: ID of the agent making the trade
            item_name: Name of the item being traded
            action_type: Type of action ('buy', 'sell', 'hold')
            price: Price of the trade
            quantity: Quantity of the trade
            profit: Profit from the trade (0 for buys, actual profit for sells)
            tax: GE tax paid on the trade (0 for buys, 1% of price for sells)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Convert item name to ID if needed
        item_id = item_name
        if item_name in self.name_to_id_map:
            item_id = self.name_to_id_map[item_name]
        
        # Initialize trade history for this item if it doesn't exist
        if item_id not in self.trade_history:
            self.trade_history[item_id] = []
        
        # Record the trade
        # Check buy limit before recording buy trades
        if action_type == 'buy' and not self._check_buy_limit(item_id, quantity, timestamp):
            logger.warning(f"Buy limit reached for item {item_id}, trade not recorded")
            return False

        self.trade_history[item_id].append({
            'agent_id': agent_id,
            'action_type': action_type,
            'price': price,
            'quantity': quantity,
            'profit': profit,
            'tax': tax,
            'timestamp': timestamp
        })

        # Record purchase in history if it's a buy
        if action_type == 'buy':
            if item_id not in self.purchase_history:
                self.purchase_history[item_id] = []
            self.purchase_history[item_id].append((timestamp, quantity))
        
        # Trim history if needed
        if len(self.trade_history[item_id]) > self.max_history_length:
            self.trade_history[item_id] = self.trade_history[item_id][-self.max_history_length:]
        
        # Update success rates and profit metrics
        self._update_metrics(item_id)
        
        # Update agent specialization
        self._update_specialization(agent_id, item_id, profit, action_type)
        
        # Update consensus signals
        self._update_consensus_signals(item_id)
        
        logger.debug(f"Recorded {action_type} trade for item {item_id} by agent {agent_id}: {quantity} @ {price} GP, profit: {profit}")
    
    def _update_metrics(self, item_id):
        """
        Update success rates and profit metrics for an item
        
        Args:
            item_id: ID of the item to update metrics for
        """
        if item_id not in self.trade_history or not self.trade_history[item_id]:
            return
        
        # Initialize metrics if they don't exist
        if item_id not in self.profit_metrics:
            self.profit_metrics[item_id] = {
                'total_profit': 0,
                'total_volume': 0,
                'profit_per_unit': 0,
                'buy_price_avg': 0,
                'sell_price_avg': 0,
                'total_tax_paid': 0,
                'last_updated': 0
            }
        
        if item_id not in self.success_rates:
            self.success_rates[item_id] = {
                'total_trades': 0,
                'successful_trades': 0,
                'success_rate': 0,
                'last_updated': 0
            }
        
        # Calculate profit metrics
        buys = [t for t in self.trade_history[item_id] if t['action_type'] == 'buy']
        sells = [t for t in self.trade_history[item_id] if t['action_type'] == 'sell']
        
        total_profit = sum(t['profit'] for t in sells)
        total_volume = sum(t['quantity'] for t in sells)
        total_tax_paid = sum(t.get('tax', 0) for t in sells)  # Sum up all tax paid
        
        buy_price_avg = sum(t['price'] * t['quantity'] for t in buys) / sum(t['quantity'] for t in buys) if buys and sum(t['quantity'] for t in buys) > 0 else 0
        sell_price_avg = sum(t['price'] * t['quantity'] for t in sells) / sum(t['quantity'] for t in sells) if sells and sum(t['quantity'] for t in sells) > 0 else 0
        
        # Update profit metrics
        self.profit_metrics[item_id]['total_profit'] = total_profit
        self.profit_metrics[item_id]['total_volume'] = total_volume
        self.profit_metrics[item_id]['profit_per_unit'] = total_profit / total_volume if total_volume > 0 else 0
        self.profit_metrics[item_id]['buy_price_avg'] = buy_price_avg
        self.profit_metrics[item_id]['sell_price_avg'] = sell_price_avg
        self.profit_metrics[item_id]['total_tax_paid'] = total_tax_paid
        self.profit_metrics[item_id]['last_updated'] = int(time.time())
        
        # Calculate success rates
        total_trades = len(sells)
        successful_trades = len([t for t in sells if t['profit'] > 0])
        
        # Update success rates
        self.success_rates[item_id]['total_trades'] = total_trades
        self.success_rates[item_id]['successful_trades'] = successful_trades
        self.success_rates[item_id]['success_rate'] = successful_trades / total_trades if total_trades > 0 else 0
        self.success_rates[item_id]['last_updated'] = int(time.time())
    
    def _update_specialization(self, agent_id, item_id, profit, action_type):
        """
        Update agent specialization based on trading performance
        
        Args:
            agent_id: ID of the agent
            item_id: ID of the item
            profit: Profit from the trade
            action_type: Type of action ('buy', 'sell', 'hold')
        """
        # Only update specialization for sell actions (which have actual profit)
        if action_type != 'sell':
            return
        
        # Initialize agent specialization if it doesn't exist
        if agent_id not in self.agent_specializations:
            self.agent_specializations[agent_id] = {}
        
        # Initialize item specialization if it doesn't exist
        if item_id not in self.agent_specializations[agent_id]:
            self.agent_specializations[agent_id][item_id] = {
                'trades': 0,
                'profit': 0,
                'performance_score': 0,
                'last_updated': 0
            }
        
        # Update specialization
        spec = self.agent_specializations[agent_id][item_id]
        spec['trades'] += 1
        spec['profit'] += profit
        
        # Calculate performance score (profit per trade)
        if spec['trades'] > 0:
            spec['performance_score'] = spec['profit'] / spec['trades']
        
        spec['last_updated'] = int(time.time())
    
    def _update_consensus_signals(self, item_id):
        """
        Update consensus trading signals for an item
        
        Args:
            item_id: ID of the item to update signals for
        """
        if item_id not in self.trade_history or not self.trade_history[item_id]:
            return
        
        # Get recent trades (last 24 hours)
        current_time = int(time.time())
        recent_trades = [t for t in self.trade_history[item_id] if current_time - t['timestamp'] < 86400]
        
        if not recent_trades:
            return
        
        # Count votes for each action type
        buy_votes = len([t for t in recent_trades if t['action_type'] == 'buy'])
        sell_votes = len([t for t in recent_trades if t['action_type'] == 'sell'])
        hold_votes = len([t for t in recent_trades if t['action_type'] == 'hold'])
        
        # Determine consensus
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus = 'buy'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus = 'sell'
        else:
            consensus = 'hold'
        
        # Update consensus signal
        self.consensus_signals[item_id] = {
            'signal': consensus,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'hold_votes': hold_votes,
            'confidence': max(buy_votes, sell_votes, hold_votes) / len(recent_trades) if recent_trades else 0,
            'last_updated': current_time
        }
    
    def get_consensus_signal(self, item_id):
        """
        Get consensus trading signal for an item
        
        Args:
            item_id: ID of the item to get signal for
            
        Returns:
            Dictionary with consensus signal information, or None if no signal exists
        """
        return self.consensus_signals.get(item_id)
    
    def get_agent_specialization(self, agent_id, item_id):
        """
        Get agent specialization for an item
        
        Args:
            agent_id: ID of the agent
            item_id: ID of the item
            
        Returns:
            Dictionary with specialization information, or None if no specialization exists
        """
        if agent_id not in self.agent_specializations:
            return None
        
        return self.agent_specializations[agent_id].get(item_id)
    
    def is_agent_specialist(self, agent_id, item_id, threshold=0.0):
        """
        Check if an agent is a specialist for an item
        
        Args:
            agent_id: ID of the agent
            item_id: ID of the item
            threshold: Performance score threshold to be considered a specialist
            
        Returns:
            True if the agent is a specialist for the item, False otherwise
        """
        spec = self.get_agent_specialization(agent_id, item_id)
        if not spec:
            return False
        
        return spec['performance_score'] > threshold and spec['trades'] >= 5
    
    def get_best_specialist(self, item_id):
        """
        Get the best specialist agent for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            ID of the best specialist agent, or None if no specialist exists
        """
        specialists = []
        
        for agent_id in self.agent_specializations:
            if item_id in self.agent_specializations[agent_id]:
                spec = self.agent_specializations[agent_id][item_id]
                if spec['trades'] >= 5:  # Require at least 5 trades to be considered a specialist
                    specialists.append((agent_id, spec['performance_score']))
        
        if not specialists:
            return None
        
        # Return the agent with the highest performance score
        return max(specialists, key=lambda x: x[1])[0]
    
    def get_shared_experiences(self, item_id, limit=100):
        """
        Get shared experiences for an item across all agents
        
        Args:
            item_id: ID of the item
            limit: Maximum number of experiences to return
            
        Returns:
            List of trade records, sorted by recency
        """
        if item_id not in self.trade_history:
            return []
        
        # Sort by recency and return limited number
        return sorted(self.trade_history[item_id], 
                     key=lambda x: x['timestamp'], 
                     reverse=True)[:limit]
    
    def get_profit_metrics(self, item_id):
        """
        Get profit metrics for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary with profit metrics, or None if no metrics exist
        """
        return self.profit_metrics.get(item_id)
    
    def get_success_rate(self, item_id):
        """
        Get success rate for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary with success rate information, or None if no information exists
        """
        return self.success_rates.get(item_id)
    
    def update_volume_data(self, data_5m, data_1h):
        """
        Update volume data in the shared volume analyzer
        
        Args:
            data_5m: Data from the 5m endpoint
            data_1h: Data from the 1h endpoint
        """
        self.volume_analyzer.update_volume_data(data_5m, data_1h)
    
    def get_volume_metrics(self, item_id):
        """
        Get volume metrics for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Dictionary with volume metrics, or None if no metrics exist
        """
        return self.volume_analyzer.get_volume_metrics(item_id)
    
    def get_buy_sell_imbalance(self, item_id):
        """
        Get buy/sell imbalance for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Buy/sell imbalance score, or None if no data exists
        """
        return self.volume_analyzer.get_buy_sell_imbalance(item_id)
    
    def get_volume_momentum(self, item_id):
        """
        Get volume momentum for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Volume momentum score, or None if no data exists
        """
        return self.volume_analyzer.get_volume_momentum(item_id)
    
    def get_market_activity(self, item_id):
        """
        Get market activity for an item
        
        Args:
            item_id: ID of the item
            
        Returns:
            Market activity score, or None if no data exists
        """
        return self.volume_analyzer.get_market_activity(item_id)

    def _clean_old_purchases(self, item_id: int, current_time: int) -> None:
        """
        Remove purchases older than 4 hours from the purchase history

        Args:
            item_id: ID of the item
            current_time: Current timestamp
        """
        if item_id not in self.purchase_history:
            return

        four_hours_ago = current_time - (4 * 60 * 60)  # 4 hours in seconds
        self.purchase_history[item_id] = [
            (t, q) for t, q in self.purchase_history[item_id]
            if t > four_hours_ago
        ]

    def _check_buy_limit(self, item_id: int, quantity: int, timestamp: int) -> bool:
        """
        Check if a purchase would exceed the buy limit for an item

        Args:
            item_id: ID of the item
            quantity: Quantity attempting to purchase
            timestamp: Timestamp of the purchase attempt

        Returns:
            True if purchase is allowed, False if it would exceed limit
        """
        if item_id not in self.buy_limits:
            return True  # No limit defined for this item

        # Get buy limit for this item
        limit = self.buy_limits[item_id]

        # Clean up old purchases
        self._clean_old_purchases(item_id, timestamp)

        # Calculate total quantity purchased in last 4 hours
        total_quantity = sum(q for _, q in self.purchase_history.get(item_id, []))

        # Check if new purchase would exceed limit
        return (total_quantity + quantity) <= limit

    def load_trade_history_from_archives(self):
        """
        Method to load trade history from archive files.
        Currently skips loading historical data as we only care about current episode data.
        """
        logger.info("Skipping loading trade history from archives - focusing on current episode data only")
        return
        
    def save_trade_history(self, filename=None, episode_number=None):
        """
        Save trade history to either a specific file or archive files.
        
        Args:
            filename: Optional specific file to save to (used by API server)
            episode_number: Optional episode number to include in archive filenames
        """
        try:
            if filename:
                # Save all trade history to a single file (API server mode)
                with open(filename, 'w') as f:
                    json.dump(self.trade_history, f, indent=2)
                logger.info(f"Saved trade history to {filename}")
            else:
                # Save to archive files (training mode)
                if not os.path.exists('archives'):
                    os.makedirs('archives')
                    
                # Save trade history for each agent
                for agent_id in set([trade['agent_id'] for trades in self.trade_history.values() for trade in trades]):
                    # Filter trades for this agent
                    agent_trades = {}
                    for item_id, trades in self.trade_history.items():
                        agent_item_trades = [t for t in trades if t['agent_id'] == agent_id]
                        if agent_item_trades:
                            agent_trades[item_id] = agent_item_trades
                    
                    if not agent_trades:
                        continue
                        
                    # Determine filename
                    episode_suffix = f"_episode_{episode_number}" if episode_number is not None else ""
                    archive_file = f"archives/agent_{agent_id}{episode_suffix}_trade_archive.json"
                    
                    # Save to file
                    with open(archive_file, 'w') as f:
                        json.dump(agent_trades, f, indent=2)
                        
                    logger.info(f"Saved trade history for agent {agent_id} to {archive_file}")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
            raise

    def load_trade_history(self, filename=None):
        """
        Load trade history from a specific file.
        Used by API server to load consolidated trade history.
        
        Args:
            filename: Path to the trade history file
        """
        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded trade history from {filename}")
            except Exception as e:
                logger.error(f"Error loading trade history from {filename}: {e}")
                raise
        else:
            # If file doesn't exist, try loading from archives
            self.load_trade_history_from_archives()
        
    def get_remaining_limit(self, item_id: int) -> Optional[int]:
        """
        Get remaining buy limit for an item

        Args:
            item_id: ID of the item

        Returns:
            Remaining quantity that can be purchased, or None if no limit exists
        """
        if item_id not in self.buy_limits:
            return None

        current_time = int(time.time())
        self._clean_old_purchases(item_id, current_time)

        limit = self.buy_limits[item_id]
        total_quantity = sum(q for _, q in self.purchase_history.get(item_id, []))
        
        return max(0, limit - total_quantity)