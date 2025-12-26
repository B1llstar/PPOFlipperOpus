# Volume Analysis Enhancement for PPOFlipper

This document explains the volume analysis enhancements added to the PPOFlipper trading bot to enable more informed trading decisions based on real-time market data.

## Overview

The original PPOFlipper trading bot primarily used simulated price fluctuations and had limited use of volume data (mainly for blacklisting low-volume items). The new enhancements leverage real-time market volume data to:

1. Compare volume distribution curves with price data to determine maximum profit opportunities
2. Analyze volume differences at frequent intervals (5-minute and hourly) to detect what's actually selling
3. Create a real-time market activity detection system based on volume changes

## Key Concepts

### Volume Data Structure

The Grand Exchange API provides volume data in two key endpoints:

- **5m endpoint**: 5-minute average price/volume data
- **1h endpoint**: 1-hour average price/volume data

Each item in these endpoints has:
- `avgHighPrice`: The average high price (sell price) for the item
- `highPriceVolume`: The volume of items sold at the high price (sell volume)
- `avgLowPrice`: The average low price (buy price) for the item
- `lowPriceVolume`: The volume of items bought at the low price (buy volume)

### Market Dynamics

In the Grand Exchange market:
- `highPriceVolume` represents sell volume (items being sold by players)
- `lowPriceVolume` represents buy volume (items being bought by players)
- When there are more buy offers than sell offers, price tends to go up
- When there are more sell offers than buy offers, price tends to go down

## New Features

### 1. Volume-Weighted Price Metrics

The `calculate_volume_weighted_price` method calculates the Volume-Weighted Average Price (VWAP) for an item. This provides a more accurate representation of an item's true market value by weighting the price by the volume traded at that price.

```python
vwap = volume_analyzer.calculate_volume_weighted_price(item_id, interval="1h")
```

### 2. Buy/Sell Imbalance Analysis

The `calculate_buy_sell_imbalance` method analyzes the imbalance between buy and sell volumes. A positive value indicates more buying than selling (bullish), while a negative value indicates more selling than buying (bearish).

```python
imbalance = volume_analyzer.calculate_buy_sell_imbalance(item_id, interval="1h")
```

### 3. Volume Momentum Calculation

The `calculate_volume_momentum` method compares recent volume to previous periods to identify items with increasing or decreasing trading activity.

```python
momentum = volume_analyzer.calculate_volume_momentum(item_id, interval="1h", periods=3)
```

### 4. Price-Volume Divergence Detection

The `find_volume_price_divergence` method identifies divergences between price and volume trends, which can indicate potential market reversals.

```python
divergence = volume_analyzer.find_volume_price_divergence(item_id, interval="1h")
```

### 5. Market Activity Scoring

The `calculate_market_activity_score` method combines various volume metrics to produce a single score representing the level of market activity for an item.

```python
activity = volume_analyzer.calculate_market_activity_score(item_id)
```

### 6. Profit Opportunity Identification

The `identify_profit_opportunities` method analyzes volume and price data to identify potential profit opportunities, ranking items by their opportunity score.

```python
opportunities = volume_analyzer.identify_profit_opportunities(min_volume=1000)
```

### 7. Real-Time Market Change Detection

The `detect_real_time_market_changes` method identifies significant changes in market activity by comparing current volume data with previous data.

```python
market_changes = volume_analyzer.detect_real_time_market_changes(threshold=0.5)
```

## Integration with PPOAgent

The volume analysis functionality is integrated with the PPOAgent class to enhance trading decisions:

1. **Item Selection**: The agent now considers volume metrics when selecting which items to trade, favoring items with favorable volume patterns.

2. **Price Determination**: The agent uses volume-weighted price metrics to determine more optimal buy and sell prices.

3. **Market Awareness**: The agent is now aware of real-time market changes and can adjust its strategy accordingly.

## Usage Examples

### Basic Usage

```python
from ge_rest_client import GrandExchangeClient
from volume_analysis import create_volume_analyzer, update_volume_analyzer

# Initialize client and read mapping file
client = GrandExchangeClient()
id_name_map = read_mapping_file()

# Create volume analyzer
volume_analyzer = create_volume_analyzer(id_name_map)

# Update with fresh data
data_5m = client.get_5m()
data_1h = client.get_1h()
volume_analyzer.update_volume_data(data_5m, data_1h)

# Identify profit opportunities
opportunities = volume_analyzer.identify_profit_opportunities()
for opp in opportunities[:5]:  # Top 5 opportunities
    print(f"{opp['item_name']}: Score={opp['score']:.2f}, Price={opp['current_price']}, VWAP={opp['vwap']}")
```

### Getting Volume Metrics for a Specific Item

```python
from volume_analysis import get_volume_metrics_for_item

# Get metrics for a specific item
metrics = get_volume_metrics_for_item(volume_analyzer, "Dragon bones")
print(f"VWAP (1h): {metrics['vwap_1h']}")
print(f"Buy/Sell Imbalance (1h): {metrics['imbalance_1h']}")
print(f"Volume Momentum (1h): {metrics['momentum_1h']}")
```

## Testing

A test script `test_volume_analysis.py` is provided to demonstrate the functionality of the volume analysis enhancements. Run it to see the volume analysis in action:

```
python test_volume_analysis.py
```

## Benefits

These enhancements transform the PPOFlipper bot from using primarily simulated data to making decisions based on actual market conditions:

1. **More Accurate Pricing**: Using volume-weighted prices helps the bot determine more accurate buy and sell prices.

2. **Better Item Selection**: The bot can now identify items with favorable volume patterns, leading to more profitable trades.

3. **Real-Time Adaptability**: The bot can detect and respond to real-time market changes, allowing it to capitalize on sudden market movements.

4. **Reduced Risk**: By analyzing buy/sell imbalances, the bot can avoid items with unfavorable market conditions.

5. **Increased Profit Potential**: The combination of these enhancements leads to more informed trading decisions and higher profit potential.