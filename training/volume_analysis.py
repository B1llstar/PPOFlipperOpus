import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import time

# Set up module-level logger
logger = logging.getLogger("volume_analysis")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("volume_analysis.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(fh)
logger.propagate = False
logger.info("volume_analysis logger initialized (imported)")

class VolumeAnalyzer:
    """
    Analyzes volume data from the Grand Exchange to identify trading opportunities
    and market trends based on volume patterns and imbalances.
    """
    
    def __init__(self, id_to_name_map: Dict[str, str], name_to_id_map: Dict[str, str]):
        """
        Initialize the VolumeAnalyzer with mappings between item IDs and names.
        
        Args:
            id_to_name_map: Dictionary mapping item IDs to item names
            name_to_id_map: Dictionary mapping item names to item IDs
        """
        self.id_to_name_map = id_to_name_map
        self.name_to_id_map = name_to_id_map
        
        # Historical data storage
        self.volume_history_5m = {}  # item_id -> list of (timestamp, highVol, lowVol) tuples
        self.volume_history_1h = {}  # item_id -> list of (timestamp, highVol, lowVol) tuples
        self.price_history_5m = {}   # item_id -> list of (timestamp, highPrice, lowPrice) tuples
        self.price_history_1h = {}   # item_id -> list of (timestamp, highPrice, lowPrice) tuples
        
        # Market metrics
        self.volume_momentum = {}    # item_id -> momentum score
        self.market_activity = {}    # item_id -> activity score
        self.buy_sell_imbalance = {} # item_id -> imbalance score
        
        # Maximum history length to prevent memory issues
        self.max_history_length = 24  # 24 hours worth of data
        self.data_loaded = False # Flag to indicate initial data load

        logger.info(f"VolumeAnalyzer initialized with {len(id_to_name_map)} items")
    
    def update_volume_data(self, data_5m: Dict[str, Dict], data_1h: Dict[str, Dict]) -> None:
        """
        Update volume history with new data from 5m and 1h endpoints.
        
        Args:
            data_5m: Data from the 5m endpoint
            data_1h: Data from the 1h endpoint
        """
        timestamp = int(time.time())
        
        # Invalidate the volume threshold cache when new data arrives
        if hasattr(self, '_volume_threshold_cache'):
            self._volume_threshold_cache = {}
            logger.debug("Cleared volume threshold cache.") # Log cache clear

        processed_5m_ids = set() # Track processed IDs
        # Update 5m data
        for item_id, item_data in data_5m.items():
            processed_5m_ids.add(item_id) # Add ID to tracker
            high_vol = item_data.get("highPriceVolume", 0) or 0
            low_vol = item_data.get("lowPriceVolume", 0) or 0
            high_price = item_data.get("avgHighPrice")
            low_price = item_data.get("avgLowPrice")
            
            if item_id not in self.volume_history_5m:
                self.volume_history_5m[item_id] = []
            if item_id not in self.price_history_5m:
                self.price_history_5m[item_id] = []
            
            self.volume_history_5m[item_id].append((timestamp, high_vol, low_vol))
            self.price_history_5m[item_id].append((timestamp, high_price, low_price))
            
            # Trim history if needed
            if len(self.volume_history_5m[item_id]) > self.max_history_length * 12:  # 12 5-min intervals per hour
                self.volume_history_5m[item_id] = self.volume_history_5m[item_id][-self.max_history_length * 12:]
            if len(self.price_history_5m[item_id]) > self.max_history_length * 12:
                self.price_history_5m[item_id] = self.price_history_5m[item_id][-self.max_history_length * 12:]

        processed_1h_ids = set() # Track processed IDs
        # Update 1h data
        for item_id, item_data in data_1h.items():
            processed_1h_ids.add(item_id) # Add ID to tracker
            high_vol = item_data.get("highPriceVolume", 0) or 0
            low_vol = item_data.get("lowPriceVolume", 0) or 0
            high_price = item_data.get("avgHighPrice")
            low_price = item_data.get("avgLowPrice")
            
            if item_id not in self.volume_history_1h:
                self.volume_history_1h[item_id] = []
            if item_id not in self.price_history_1h:
                self.price_history_1h[item_id] = []
            
            self.volume_history_1h[item_id].append((timestamp, high_vol, low_vol))
            self.price_history_1h[item_id].append((timestamp, high_price, low_price))
            
            # Trim history if needed
            if len(self.volume_history_1h[item_id]) > self.max_history_length:
                self.volume_history_1h[item_id] = self.volume_history_1h[item_id][-self.max_history_length:]
            if len(self.price_history_1h[item_id]) > self.max_history_length:
                self.price_history_1h[item_id] = self.price_history_1h[item_id][-self.max_history_length:]

        # Log summary after processing both
        logger.info(f"Updated volume data. Processed {len(processed_5m_ids)} unique 5m IDs and {len(processed_1h_ids)} unique 1h IDs.")
        # Optional: Log a few example IDs that were processed (for debugging)
        if processed_5m_ids: logger.debug(f"Example 5m IDs processed: {list(processed_5m_ids)[:5]}")
        if processed_1h_ids: logger.debug(f"Example 1h IDs processed: {list(processed_1h_ids)[:5]}")

        # Set flag after first successful update that processes at least one item
        if not self.data_loaded and (processed_5m_ids or processed_1h_ids):
            self.data_loaded = True
            logger.info("VolumeAnalyzer: Initial data load complete.")

    def calculate_volume_weighted_price(self, item_id: str, interval: str = "1h") -> Optional[float]:
        """
        Calculate volume-weighted average price (VWAP) for an item.
        
        Args:
            item_id: The item ID
            interval: "5m" or "1h"
            
        Returns:
            Volume-weighted average price or None if insufficient data
        """
        if interval == "5m":
            price_history = self.price_history_5m.get(item_id, [])
            volume_history = self.volume_history_5m.get(item_id, [])
        else:
            price_history = self.price_history_1h.get(item_id, [])
            volume_history = self.volume_history_1h.get(item_id, [])
        
        if not price_history or not volume_history:
            return None
        
        # Match timestamps between price and volume data
        price_dict = {ts: (hp, lp) for ts, hp, lp in price_history}
        volume_dict = {ts: (hv, lv) for ts, hv, lv in volume_history}
        
        # Calculate VWAP
        total_volume = 0
        weighted_sum = 0
        
        for ts in set(price_dict.keys()).intersection(volume_dict.keys()):
            high_price, low_price = price_dict[ts]
            high_vol, low_vol = volume_dict[ts]
            
            # Skip if prices are None
            if high_price is None and low_price is None:
                continue
            
            # Use available price, prioritize the one with volume
            if high_price is not None and high_vol > 0:
                weighted_sum += high_price * high_vol
                total_volume += high_vol
            
            if low_price is not None and low_vol > 0:
                weighted_sum += low_price * low_vol
                total_volume += low_vol
        
        if total_volume == 0:
            return None
        
        return weighted_sum / total_volume
    
    def calculate_buy_sell_imbalance(self, item_id: str, interval: str = "1h") -> Optional[float]:
        """
        Calculate the imbalance between buy and sell volumes.
        Positive values indicate more buying than selling (bullish).
        Negative values indicate more selling than buying (bearish).
        
        Args:
            item_id: The item ID
            interval: "5m" or "1h"
            
        Returns:
            Imbalance score between -1 and 1, or None if insufficient data
        """
        if interval == "5m":
            volume_history = self.volume_history_5m.get(item_id, [])
        else:
            volume_history = self.volume_history_1h.get(item_id, [])
        
        if not volume_history:
            return None
        
        # Get the most recent volume data
        _, high_vol, low_vol = volume_history[-1]
        
        total_vol = high_vol + low_vol
        if total_vol == 0:
            return 0
        
        # Calculate imbalance: (buy_vol - sell_vol) / total_vol
        # Note: highPriceVolume = sell volume, lowPriceVolume = buy volume
        imbalance = (low_vol - high_vol) / total_vol
        
        # Store for later use
        self.buy_sell_imbalance[item_id] = imbalance
        
        return imbalance
    
    def calculate_volume_momentum(self, item_id: str, interval: str = "1h", periods: int = 3) -> Optional[float]:
        """
        Calculate volume momentum by comparing recent volume to previous periods.
        
        Args:
            item_id: The item ID
            interval: "5m" or "1h"
            periods: Number of periods to compare
            
        Returns:
            Momentum score or None if insufficient data
        """
        if interval == "5m":
            volume_history = self.volume_history_5m.get(item_id, [])
        else:
            volume_history = self.volume_history_1h.get(item_id, [])
        
        if len(volume_history) < periods + 1:
            return None
        
        # Calculate total volume (high + low) for each period
        total_volumes = [high_vol + low_vol for _, high_vol, low_vol in volume_history]
        
        # Calculate recent average volume vs previous average volume
        recent_avg = sum(total_volumes[-periods:]) / periods
        
        # If we have enough history, compare to previous periods
        if len(total_volumes) >= periods * 2:
            previous_avg = sum(total_volumes[-(periods*2):-periods]) / periods
            if previous_avg == 0:
                momentum = 0 if recent_avg == 0 else 1
            else:
                momentum = (recent_avg - previous_avg) / previous_avg
        else:
            # Not enough history for comparison, use absolute volume
            momentum = recent_avg / 1000  # Normalize
        
        # Store for later use
        self.volume_momentum[item_id] = momentum
        
        return momentum
    
    def calculate_market_activity_score(self, item_id: str) -> float:
        """
        Calculate a market activity score based on volume changes and imbalances.
        
        Args:
            item_id: The item ID
            
        Returns:
            Market activity score (higher = more active)
        """
        # Calculate components if not already calculated
        if item_id not in self.volume_momentum:
            self.calculate_volume_momentum(item_id)
        if item_id not in self.buy_sell_imbalance:
            self.calculate_buy_sell_imbalance(item_id)
        
        momentum = self.volume_momentum.get(item_id, 0)
        imbalance = abs(self.buy_sell_imbalance.get(item_id, 0))  # Use absolute imbalance
        
        # Calculate recent volume
        recent_volume = 0
        if item_id in self.volume_history_1h and self.volume_history_1h[item_id]:
            _, high_vol, low_vol = self.volume_history_1h[item_id][-1]
            recent_volume = high_vol + low_vol
        
        # Combine factors into a single score
        # Weight momentum more heavily for trending markets
        activity_score = (0.5 * momentum) + (0.3 * imbalance) + (0.2 * min(1.0, recent_volume / 10000))
        
        # Store for later use
        self.market_activity[item_id] = activity_score
        
        return activity_score
    
    def find_volume_price_divergence(self, item_id: str, interval: str = "1h", periods: int = 3) -> Optional[float]:
        """
        Find divergence between price and volume trends, which can indicate potential reversals.
        
        Args:
            item_id: The item ID
            interval: "5m" or "1h"
            periods: Number of periods to analyze
            
        Returns:
            Divergence score (positive = bullish divergence, negative = bearish divergence)
            or None if insufficient data
        """
        if interval == "5m":
            price_history = self.price_history_5m.get(item_id, [])
            volume_history = self.volume_history_5m.get(item_id, [])
        else:
            price_history = self.price_history_1h.get(item_id, [])
            volume_history = self.volume_history_1h.get(item_id, [])
        
        if len(price_history) < periods or len(volume_history) < periods:
            return None
        
        # Calculate price trend
        recent_prices = []
        for _, high_price, low_price in price_history[-periods:]:
            # Use average of high and low if both available
            if high_price is not None and low_price is not None:
                recent_prices.append((high_price + low_price) / 2)
            elif high_price is not None:
                recent_prices.append(high_price)
            elif low_price is not None:
                recent_prices.append(low_price)
            else:
                recent_prices.append(None)
        
        # Filter out None values
        recent_prices = [p for p in recent_prices if p is not None]
        if len(recent_prices) < 2:
            return None
        
        # Simple price trend: positive = up, negative = down
        price_trend = recent_prices[-1] - recent_prices[0]
        
        # Calculate volume trend
        recent_volumes = []
        for _, high_vol, low_vol in volume_history[-periods:]:
            recent_volumes.append(high_vol + low_vol)
        
        volume_trend = recent_volumes[-1] - recent_volumes[0]
        
        # Normalize trends
        if abs(recent_prices[0]) > 0:
            price_trend = price_trend / recent_prices[0]
        else:
            price_trend = 0
            
        if recent_volumes[0] > 0:
            volume_trend = volume_trend / recent_volumes[0]
        else:
            volume_trend = 0
        
        # Calculate divergence
        # Positive divergence: Price down, volume up (bullish)
        # Negative divergence: Price up, volume down (bearish)
        divergence = volume_trend - price_trend
        
        return divergence
    
    def identify_profit_opportunities(self, min_volume: int = 10000) -> List[Dict[str, Any]]:
        """
        Identify potential profit opportunities based on volume analysis.
        
        Args:
            min_volume: Minimum volume threshold to consider
            
        Returns:
            List of opportunities with scores and metrics
        """
        opportunities = []
        
        for item_id in self.volume_history_1h.keys():
            # Skip items with insufficient volume
            if item_id not in self.volume_history_1h or not self.volume_history_1h[item_id]:
                continue
                
            _, high_vol, low_vol = self.volume_history_1h[item_id][-1]
            if high_vol + low_vol < min_volume:
                continue
            
            # Calculate metrics
            imbalance = self.calculate_buy_sell_imbalance(item_id)
            momentum = self.calculate_volume_momentum(item_id)
            vwap = self.calculate_volume_weighted_price(item_id)
            divergence = self.find_volume_price_divergence(item_id)
            activity = self.calculate_market_activity_score(item_id)
            
            # Skip items with insufficient data
            if imbalance is None or momentum is None or vwap is None:
                continue
            
            # Get current prices
            current_high = None
            current_low = None
            if item_id in self.price_history_1h and self.price_history_1h[item_id]:
                _, current_high, current_low = self.price_history_1h[item_id][-1]
            
            # Skip items with no price data
            if current_high is None and current_low is None:
                continue
            
            # Use available price
            current_price = current_high if current_high is not None else current_low
            
            # Calculate opportunity score
            # Higher score = better opportunity
            opportunity_score = 0
            
            # Factor 1: Price vs VWAP (undervalued or overvalued)
            if current_price is not None and vwap is not None and vwap > 0:
                price_vwap_ratio = current_price / vwap - 1
                opportunity_score += -price_vwap_ratio * 2  # Negative ratio = undervalued = good opportunity
            
            # Factor 2: Buy/sell imbalance (positive = more buying pressure)
            if imbalance is not None:
                opportunity_score += imbalance * 3
            
            # Factor 3: Volume momentum (positive = increasing volume)
            if momentum is not None:
                opportunity_score += min(momentum, 2) * 2  # Cap extreme values
            
            # Factor 4: Price-volume divergence
            if divergence is not None:
                opportunity_score += divergence * 1.5
            
            # Factor 5: Market activity
            if activity is not None:
                opportunity_score += activity * 1
            
            # Create opportunity object
            item_name = self.id_to_name_map.get(item_id, f"Unknown ({item_id})")
            opportunity = {
                "item_id": item_id,
                "item_name": item_name,
                "score": opportunity_score,
                "current_price": current_price,
                "vwap": vwap,
                "imbalance": imbalance,
                "momentum": momentum,
                "divergence": divergence,
                "activity": activity,
                "volume": high_vol + low_vol
            }
            
            opportunities.append(opportunity)
        
        # Sort by opportunity score (descending)
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        return opportunities
    
    def detect_real_time_market_changes(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect significant real-time changes in market activity.
        
        Args:
            threshold: Threshold for significant changes
            
        Returns:
            List of items with significant market changes
        """
        market_changes = []
        
        for item_id in self.volume_history_5m.keys():
            # Need at least 2 data points to detect changes
            if item_id not in self.volume_history_5m or len(self.volume_history_5m[item_id]) < 2:
                continue
            
            # Get current and previous volume data
            current = self.volume_history_5m[item_id][-1]
            previous = self.volume_history_5m[item_id][-2]
            
            _, current_high_vol, current_low_vol = current
            _, previous_high_vol, previous_low_vol = previous
            
            current_total = current_high_vol + current_low_vol
            previous_total = previous_high_vol + previous_low_vol
            
            # Skip items with very low volume
            if current_total < 1000 and previous_total < 1000:
                continue
            
            # Calculate volume change
            if previous_total > 0:
                volume_change = (current_total - previous_total) / previous_total
            else:
                volume_change = 1 if current_total > 0 else 0
            
            # Calculate buy/sell ratio change
            current_ratio = current_low_vol / max(1, current_high_vol)  # buy/sell ratio
            previous_ratio = previous_low_vol / max(1, previous_high_vol)
            
            ratio_change = current_ratio - previous_ratio
            
            # Detect significant changes
            if abs(volume_change) > threshold or abs(ratio_change) > threshold:
                item_name = self.id_to_name_map.get(item_id, f"Unknown ({item_id})")
                
                # Get current prices
                current_high = None
                current_low = None
                if item_id in self.price_history_5m and self.price_history_5m[item_id]:
                    _, current_high, current_low = self.price_history_5m[item_id][-1]
                
                change = {
                    "item_id": item_id,
                    "item_name": item_name,
                    "volume_change": volume_change,
                    "ratio_change": ratio_change,
                    "current_volume": current_total,
                    "high_price": current_high,
                    "low_price": current_low,
                    "buy_volume": current_low_vol,
                    "sell_volume": current_high_vol
                }
                
                market_changes.append(change)
        
        # Sort by absolute volume change (descending)
        market_changes.sort(key=lambda x: abs(x["volume_change"]), reverse=True)
        
        return market_changes
    
    def check_volume_threshold(self, item_id: str, min_volume: int = 1000) -> bool:
        """
        Check if an item meets the minimum volume threshold for trading.
        
        Args:
            item_id: The item ID to check
            min_volume: Minimum required volume threshold (default: 1000)
            
        Returns:
            True if the item meets the volume threshold, False otherwise. Returns True if data hasn't loaded yet.
        """
        # If initial data hasn't loaded, assume threshold is met to avoid blocking early trades
        if not self.data_loaded:
            # logger.debug(f"Volume data not yet loaded, assuming item {item_id} meets threshold.") # Optional debug log
            return True

        # Use a cache key that includes the item_id and min_volume
        cache_key = f"{item_id}_{min_volume}"
        
        # Check if we have a cached result
        if hasattr(self, '_volume_threshold_cache') and cache_key in self._volume_threshold_cache:
            return self._volume_threshold_cache[cache_key]
        
        # If not, initialize the cache
        if not hasattr(self, '_volume_threshold_cache'):
            self._volume_threshold_cache = {}
            
        item_name = self.id_to_name_map.get(item_id, f"Unknown ({item_id})")
        
        # First check if item exists in recent data
        exists_in_1h = (item_id in self.volume_history_1h and len(self.volume_history_1h[item_id]) > 0)
        exists_in_5m = (item_id in self.volume_history_5m and len(self.volume_history_5m[item_id]) > 0)
        
        # If item doesn't exist in either dataset (and data *has* been loaded), it has no volume
        if not exists_in_1h and not exists_in_5m:
            # Only log warning if data has actually been loaded
            # logger.warning(f"Item {item_name} not found in recent volume data (data loaded: {self.data_loaded}) - treating as zero volume") # Keep warning concise
            logger.debug(f"Item {item_name} not found in recent volume data - treating as zero volume") # Use debug to reduce noise
            self._volume_threshold_cache[cache_key] = False
            return False
            
        # Check 1h data first (more stable)
        if exists_in_1h:
            # Get average volume from recent history (up to last 3 data points)
            recent_data = self.volume_history_1h[item_id][-3:]
            total_volume = 0
            for _, high_vol, low_vol in recent_data:
                total_volume += high_vol + low_vol
            avg_volume = total_volume / len(recent_data)
            
            # Check if average volume meets threshold
            if avg_volume >= min_volume:
                self._volume_threshold_cache[cache_key] = True
                return True
        
        # If no 1h data or threshold not met, check 5m data
        if item_id in self.volume_history_5m and self.volume_history_5m[item_id]:
            # Get average volume from recent history (up to last 6 data points)
            recent_data = self.volume_history_5m[item_id][-6:]
            total_volume = 0
            for _, high_vol, low_vol in recent_data:
                total_volume += high_vol + low_vol
            avg_volume = total_volume / len(recent_data)
            
            # Check if average volume meets threshold
            if avg_volume >= min_volume:
                self._volume_threshold_cache[cache_key] = True
                return True
        
        # If we get here, the item doesn't meet the volume threshold
        self._volume_threshold_cache[cache_key] = False
        return False

    def batch_check_volume_threshold(self, item_ids: List[str], min_volume: int = 1000) -> List[str]:
        """
        Efficiently check multiple items against volume threshold in one pass.
        
        Args:
            item_ids: List of item IDs to check
            min_volume: Minimum required volume threshold
            
        Returns:
            List of item IDs that don't meet the threshold
        """
        # Initialize cache if needed
        if not hasattr(self, '_volume_threshold_cache'):
            self._volume_threshold_cache = {}
            
        # Items that don't meet the threshold
        low_volume_items = []
        
        # First check which items we already have cached results for
        items_to_check = []
        for item_id in item_ids:
            cache_key = f"{item_id}_{min_volume}"
            if cache_key in self._volume_threshold_cache:
                if not self._volume_threshold_cache[cache_key]:
                    low_volume_items.append(item_id)
            else:
                items_to_check.append(item_id)
                
        if not items_to_check:
            return low_volume_items
            
        # Process 1h data for all remaining items
        for item_id in items_to_check:
            cache_key = f"{item_id}_{min_volume}"
            
            # Check if item exists in recent data
            exists_in_1h = (item_id in self.volume_history_1h and len(self.volume_history_1h[item_id]) > 0)
            exists_in_5m = (item_id in self.volume_history_5m and len(self.volume_history_5m[item_id]) > 0)
            
            # If item doesn't exist in either dataset, it has no volume
            if not exists_in_1h and not exists_in_5m:
                item_name = self.id_to_name_map.get(item_id, f"Unknown ({item_id})")
                logger.warning(f"Item {item_name} not found in recent volume data - treating as zero volume")
                self._volume_threshold_cache[cache_key] = False
                low_volume_items.append(item_id)
                continue
                
            # Check 1h data first (more stable)
            meets_threshold = False
            if exists_in_1h:
                # Get average volume from recent history (up to last 3 data points)
                recent_data = self.volume_history_1h[item_id][-3:]
                total_volume = 0
                for _, high_vol, low_vol in recent_data:
                    total_volume += high_vol + low_vol
                avg_volume = total_volume / len(recent_data)
                
                # Check if average volume meets threshold
                if avg_volume >= min_volume:
                    meets_threshold = True
            
            # If no 1h data or threshold not met, check 5m data
            if not meets_threshold and item_id in self.volume_history_5m and self.volume_history_5m[item_id]:
                # Get average volume from recent history (up to last 6 data points)
                recent_data = self.volume_history_5m[item_id][-6:]
                total_volume = 0
                for _, high_vol, low_vol in recent_data:
                    total_volume += high_vol + low_vol
                avg_volume = total_volume / len(recent_data)
                
                # Check if average volume meets threshold
                if avg_volume >= min_volume:
                    meets_threshold = True
            
            # Cache the result
            self._volume_threshold_cache[cache_key] = meets_threshold
            
            # If threshold not met, add to low volume items
            if not meets_threshold:
                low_volume_items.append(item_id)
        
        return low_volume_items

# Helper functions for integration with the existing codebase

def create_volume_analyzer(id_name_map: Dict[str, str]) -> VolumeAnalyzer:
    """
    Create a VolumeAnalyzer instance with the given ID-to-name mapping.
    
    Args:
        id_name_map: Dictionary mapping item IDs to item names
        
    Returns:
        VolumeAnalyzer instance
    """
    # Create reverse mapping (name to ID)
    name_to_id_map = {name: id for id, name in id_name_map.items()}
    
    return VolumeAnalyzer(id_name_map, name_to_id_map)

def update_volume_analyzer(analyzer: VolumeAnalyzer, client) -> None:
    """
    Update the VolumeAnalyzer with fresh data from the GrandExchangeClient.
    
    Args:
        analyzer: VolumeAnalyzer instance
        client: GrandExchangeClient instance
    """
    try:
        # Clear metrics cache when updating data
        if hasattr(analyzer, '_metrics_cache'):
            analyzer._metrics_cache = {}
            logger.info("Cleared volume metrics cache")
            
        data_5m = client.get_5m()
        data_1h = client.get_1h()
        analyzer.update_volume_data(data_5m, data_1h)
        logger.info(f"Updated VolumeAnalyzer with fresh data: {len(data_5m)} 5m items, {len(data_1h)} 1h items")
    except Exception as e:
        logger.error(f"Error updating VolumeAnalyzer: {e}")

def get_volume_metrics_for_item(analyzer: VolumeAnalyzer, item_name: str) -> Dict[str, Any]:
    """
    Get volume-based metrics for a specific item.
    
    Args:
        analyzer: VolumeAnalyzer instance
        item_name: Name of the item
        
    Returns:
        Dictionary of volume metrics
    """
    # Check if we have a cached result
    if hasattr(analyzer, '_metrics_cache') and item_name in analyzer._metrics_cache:
        # Check if cache is still valid (less than 60 seconds old)
        cache_time, cached_metrics = analyzer._metrics_cache[item_name]
        if time.time() - cache_time < 60:  # 60 second cache validity
            return cached_metrics
    
    # If not, initialize the cache
    if not hasattr(analyzer, '_metrics_cache'):
        analyzer._metrics_cache = {}
    
    item_id = analyzer.name_to_id_map.get(item_name)
    if not item_id:
        return {}
    
    metrics = {
        "vwap_1h": analyzer.calculate_volume_weighted_price(item_id, "1h"),
        "vwap_5m": analyzer.calculate_volume_weighted_price(item_id, "5m"),
        "imbalance_1h": analyzer.calculate_buy_sell_imbalance(item_id, "1h"),
        "imbalance_5m": analyzer.calculate_buy_sell_imbalance(item_id, "5m"),
        "momentum_1h": analyzer.calculate_volume_momentum(item_id, "1h"),
        "momentum_5m": analyzer.calculate_volume_momentum(item_id, "5m"),
        "divergence_1h": analyzer.find_volume_price_divergence(item_id, "1h"),
        "activity_score": analyzer.calculate_market_activity_score(item_id),
        "meets_volume_threshold": analyzer.check_volume_threshold(item_id, 100)
    }
    
    # Add recent volume data
    if item_id in analyzer.volume_history_1h and analyzer.volume_history_1h[item_id]:
        recent_data = analyzer.volume_history_1h[item_id][-3:]
        total_volume = 0
        for _, high_vol, low_vol in recent_data:
            total_volume += high_vol + low_vol
        metrics["recent_volume"] = total_volume / len(recent_data)
    
    # Cache the result with current timestamp
    analyzer._metrics_cache[item_name] = (time.time(), metrics)
    
    return metrics