"""
Test script to verify the dashboard backend is working correctly.
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/api/health")
    health = response.json()
    
    print(f"  Status: {health['status']}")
    print(f"  Database exists: {health['database']['exists']}")
    print(f"  Database size: {health['database']['size_mb']} MB")
    print(f"  Training state: {health['training']['state']}")
    print(f"  WebSocket connections: {health['connections']['websockets']}")
    
    if not health['database']['exists']:
        print("  ⚠️  WARNING: Database not found!")
        return False
    
    print("  ✓ Health check passed\n")
    return True

def test_config():
    """Test config endpoint."""
    print("Testing config endpoint...")
    response = requests.get(f"{API_URL}/api/config")
    config = response.json()
    
    print(f"  Starting GP: {config['env']['starting_gp']:,}")
    print(f"  Episode length: {config['env']['max_steps']}")
    print(f"  Learning rate: {config['ppo']['lr']}")
    print(f"  Number of agents: {config['train']['num_agents']}")
    print("  ✓ Config endpoint working\n")

def test_state():
    """Test state endpoint."""
    print("Testing state endpoint...")
    response = requests.get(f"{API_URL}/api/state")
    state = response.json()
    
    print(f"  Training state: {state['training']['state']}")
    print(f"  Total episodes: {state['training']['total_episodes']}")
    print(f"  Total steps: {state['training']['total_steps']}")
    print(f"  Number of agents: {len(state['agents'])}")
    print("  ✓ State endpoint working\n")

def test_training_control():
    """Test training control endpoints."""
    print("Testing training control...")
    
    # Test start
    print("  Starting training...")
    response = requests.post(f"{API_URL}/api/training/start")
    if response.status_code == 200:
        print("  ✓ Training started")
    else:
        print(f"  ✗ Failed to start: {response.text}")
        return
    
    # Wait a bit
    time.sleep(3)
    
    # Check state
    response = requests.get(f"{API_URL}/api/state")
    state = response.json()
    print(f"  Training state: {state['training']['state']}")
    
    # Test pause
    print("  Pausing training...")
    response = requests.post(f"{API_URL}/api/training/pause")
    if response.status_code == 200:
        print("  ✓ Training paused")
    
    time.sleep(1)
    
    # Test resume
    print("  Resuming training...")
    response = requests.post(f"{API_URL}/api/training/resume")
    if response.status_code == 200:
        print("  ✓ Training resumed")
    
    time.sleep(2)
    
    # Test stop
    print("  Stopping training...")
    response = requests.post(f"{API_URL}/api/training/stop")
    if response.status_code == 200:
        print("  ✓ Training stopped")
    
    print()

def main():
    print("=" * 60)
    print("PPO Flipper Dashboard Backend Test")
    print("=" * 60)
    print()
    
    try:
        # Test basic endpoints
        if not test_health():
            print("\n⚠️  Health check failed. Please ensure:")
            print("  1. Backend server is running (python server.py)")
            print("  2. ge_prices.db exists in project root")
            return
        
        test_config()
        test_state()
        
        # Ask before testing training control
        print("=" * 60)
        response = input("Test training control? (y/n): ")
        if response.lower() == 'y':
            test_training_control()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to backend")
        print("  Please ensure the backend server is running:")
        print("  cd dashboard/backend")
        print("  python server.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
