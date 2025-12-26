import asyncio
import logging
import time

from ppo_inferencing.integration import PPOWebSocketIntegration
from ppo_inferencing.constants import OrderState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_buy_cancel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_buy_cancel_client")

async def run_buy_cancel_test():
    logger.info("Starting PPO Inferencing Client Buy/Cancel Test...")
    integration = PPOWebSocketIntegration(id_to_name_map={}, name_to_id_map={})

    logger.info("Attempting to connect to WebSocket server...")
    if not await integration.connect():
        logger.error("Failed to connect to WebSocket server. Exiting.")
        return

    logger.info("Connection successful.")
    order_id_to_track = None

    try:
        # 1. Place a buy order
        order_data = {
            "item": "Blue feather",
            "type": "buy",
            "price": 1,
            "quantity": 1,
            "is_experiment": False # Or True, depending on default server handling desired
        }
        logger.info(f"Attempting to place buy order: {order_data}")
        place_success = await integration.place_order(order_data)

        if not place_success:
            logger.error("Failed to place order initially.")
            # Check if it was queued
            if integration.order_manager.order_queue:
                logger.info("Order might be queued. Will not proceed with cancellation test for queued order in this script.")
            return

        order_id_to_track = order_data.get('order_id')
        if not order_id_to_track:
            logger.error("Order ID not found in order_data after successful placement call.")
            return

        logger.info(f"Order placement request submitted. Order ID: {order_id_to_track}")

        # 2. Wait for confirmation (ACTIVE or ACKNOWLEDGED)
        max_wait_time = 30  # seconds
        start_time = time.time()
        confirmed = False
        while time.time() - start_time < max_wait_time:
            order_info = integration.order_manager.get_order_info(order_id_to_track)
            if order_info:
                current_state = OrderState(order_info['state']) # Ensure it's an Enum for comparison
                logger.info(f"Order {order_id_to_track} current state: {current_state.name}")
                if current_state == OrderState.ACTIVE: # Wait specifically for ACTIVE state
                    logger.info(f"Order {order_id_to_track} confirmed with state: {current_state.name}")
                    confirmed = True
                    break
                elif current_state == OrderState.ACKNOWLEDGED:
                    logger.info(f"Order {order_id_to_track} is ACKNOWLEDGED. Waiting for ACTIVE...")
                elif current_state in [OrderState.FILLED, OrderState.CANCELED, OrderState.ERROR, OrderState.TIMEOUT]:
                    logger.warning(f"Order {order_id_to_track} reached terminal state {current_state.name} before intended cancellation point.")
                    confirmed = False # Mark as not confirmed for cancellation
                    break
            else:
                logger.info(f"Waiting for order {order_id_to_track} to appear in order_manager...")
            await asyncio.sleep(1)

        if not confirmed:
            logger.error(f"Order {order_id_to_track} was not confirmed (ACTIVE/ACKNOWLEDGED) within {max_wait_time} seconds or reached other terminal state.")
            # Attempt to log final state if order_info was ever fetched
            final_order_info = integration.order_manager.get_order_info(order_id_to_track)
            if final_order_info:
                logger.error(f"Final state of order {order_id_to_track}: {OrderState(final_order_info['state']).name}")
            else:
                logger.error(f"Order {order_id_to_track} not found in order_manager at the end of confirmation wait.")
            return

        # 3. Cancel the order
        logger.info(f"Attempting to cancel order {order_id_to_track}...")
        cancel_success = await integration.cancel_order(order_id_to_track)

        if cancel_success:
            logger.info(f"Cancel request for order {order_id_to_track} submitted successfully. Waiting for cancellation confirmation...")

            # Wait for CANCELED state
            max_cancel_wait_time = 30  # seconds
            cancel_start_time = time.time()
            cancellation_confirmed = False
            while time.time() - cancel_start_time < max_cancel_wait_time:
                order_info = integration.order_manager.get_order_info(order_id_to_track)
                if order_info:
                    current_state = OrderState(order_info['state'])
                    logger.info(f"Order {order_id_to_track} current state (waiting for cancel): {current_state.name}")
                    if current_state == OrderState.CANCELED:
                        logger.info(f"Order {order_id_to_track} successfully CANCELED.")
                        cancellation_confirmed = True
                        break
                    elif current_state not in [OrderState.ACTIVE, OrderState.ACKNOWLEDGED, OrderState.SUBMITTING]: # Still pending cancellation
                        logger.warning(f"Order {order_id_to_track} entered unexpected state {current_state.name} while waiting for cancellation.")
                        break
                else:
                    logger.warning(f"Order {order_id_to_track} not found in order_manager while waiting for cancellation. This might indicate it was already removed.")
                    # This could happen if the order was cancelled and removed very quickly.
                    # We might need a more robust way to confirm cancellation if the order object disappears.
                    # For now, if it's gone, we might assume cancellation was processed.
                    # However, ideally, we'd see the CANCELED state before it's removed.
                    break # Exit loop if order info is gone
                await asyncio.sleep(1)

            if not cancellation_confirmed:
                logger.error(f"Cancellation of order {order_id_to_track} not confirmed within {max_cancel_wait_time} seconds.")
                final_order_info = integration.order_manager.get_order_info(order_id_to_track)
                if final_order_info:
                    logger.error(f"Final state of order {order_id_to_track} after cancel attempt: {OrderState(final_order_info['state']).name}")
                else:
                    logger.error(f"Order {order_id_to_track} not found in order_manager at the end of cancellation wait.")

        else:
            logger.error(f"Failed to submit cancel request for order {order_id_to_track}.")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
    finally:
        # Disconnect
        logger.info("Disconnecting...")
        await integration.disconnect()
        logger.info("PPO Inferencing Client Buy/Cancel Test Finished.")

if __name__ == "__main__":
    asyncio.run(run_buy_cancel_test())