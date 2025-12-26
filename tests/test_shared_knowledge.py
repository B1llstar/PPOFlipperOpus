import json
import time
import logging
from shared_knowledge import SharedKnowledgeRepository
from ge_rest_client import GrandExchangeClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("test_shared_knowledge")

def main():
    print("Testing Shared Knowledge Repository")
    print("===================================")
    
    # Load mapping data
    try:
        with open("endpoints/mapping.txt", 'r') as f:
            mapping_data = json.load(f)
            
        # Create ID to name mapping
        id_to_name_map = {}
        name_to_id_map = {}
        for item in mapping_data:
            item_id = str(item.get('id'))
            item_name = item.get('name')
            if item_id and item_name:
                id_to_name_map[item_id] = item_name
                name_to_id_map[item_name] = item_id
        
        print(f"Loaded mapping data for {len(id_to_name_map)} items")
    except Exception as e:
        print(f"Error loading mapping data: {e}")
        # Provide minimal default data
        id_to_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        name_to_id_map = {"Fire rune": "554", "Water rune": "555", "Air rune": "556"}
    
    # Create GE client
    client = GrandExchangeClient()
    
    # Create shared knowledge repository
    shared_knowledge = SharedKnowledgeRepository(id_to_name_map, name_to_id_map)
    
    # Fetch and update volume data
    print("Fetching volume data...")
    data_5m = client.get_5m()
    data_1h = client.get_1h()
    shared_knowledge.update_volume_data(data_5m, data_1h)
    print(f"Updated volume data for {len(data_1h)} items")
    
    # Simulate multiple agents trading
    print("\nSimulating agent trading...")
    
    # Sample items to trade
    sample_items = []
    for item_id, item_name in list(id_to_name_map.items())[:10]:  # First 10 items
        sample_items.append((item_id, item_name))
    
    # Simulate trades from 5 agents
    for agent_id in range(5):
        print(f"\nAgent {agent_id} trading:")
        for item_id, item_name in sample_items:
            # Simulate buy
            buy_price = 100 + agent_id * 10  # Different prices for different agents
            buy_quantity = 10
            shared_knowledge.record_trade(
                agent_id=agent_id,
                item_name=item_name,
                action_type='buy',
                price=buy_price,
                quantity=buy_quantity,
                profit=0,
                timestamp=int(time.time())
            )
            print(f"  BUY: {item_name} x{buy_quantity} @ {buy_price}gp")
            
            # Simulate sell (with profit)
            sell_price = buy_price + 20  # 20gp profit per item
            sell_quantity = 5
            
            # Calculate GE tax (1% of price for items >= 100gp)
            tax = 0
            if sell_price >= 100:
                tax_per_item = min(int(sell_price * 0.01), 5000000)  # 1% capped at 5M per item
                tax = tax_per_item * sell_quantity
            
            # Calculate profit after tax
            gross_profit = (sell_price - buy_price) * sell_quantity
            net_profit = gross_profit - tax
            
            shared_knowledge.record_trade(
                agent_id=agent_id,
                item_name=item_name,
                action_type='sell',
                price=sell_price,
                quantity=sell_quantity,
                profit=net_profit,
                tax=tax,
                timestamp=int(time.time()) + 60  # 1 minute later
            )
            
            # Print with tax information
            if tax > 0:
                print(f"  SELL: {item_name} x{sell_quantity} @ {sell_price}gp (profit: {net_profit}gp, tax: {tax}gp)")
            else:
                print(f"  SELL: {item_name} x{sell_quantity} @ {sell_price}gp (profit: {net_profit}gp)")
    
    # Wait for data to be processed
    time.sleep(1)
    
    # Check consensus signals
    print("\nConsensus Signals:")
    for item_id, item_name in sample_items:
        consensus = shared_knowledge.get_consensus_signal(item_name)
        if consensus:
            print(f"  {item_name}: {consensus['signal']} (confidence: {consensus['confidence']:.2f})")
    
    # Check agent specializations
    print("\nAgent Specializations:")
    for agent_id in range(5):
        print(f"  Agent {agent_id}:")
        for item_id, item_name in sample_items:
            spec = shared_knowledge.get_agent_specialization(agent_id, item_name)
            if spec and spec['trades'] > 0:
                print(f"    {item_name}: {spec['performance_score']:.2f}gp/trade ({spec['trades']} trades)")
    
    # Check profit metrics
    print("\nProfit Metrics:")
    for item_id, item_name in sample_items:
        metrics = shared_knowledge.get_profit_metrics(item_name)
        if metrics:
            print(f"  {item_name}: {metrics['profit_per_unit']:.2f}gp/unit (total: {metrics['total_profit']}gp, tax paid: {metrics['total_tax_paid']}gp)")
    
    # Check best specialists
    print("\nBest Specialists:")
    for item_id, item_name in sample_items:
        best = shared_knowledge.get_best_specialist(item_name)
        if best is not None:
            spec = shared_knowledge.get_agent_specialization(best, item_name)
            print(f"  {item_name}: Agent {best} ({spec['performance_score']:.2f}gp/trade)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()