import logging
import sys
import threading
import time
from typing import Any

try:
    from proxy_inference_engine import pie_core
except ImportError:
    logging.error(
        "Failed to import pie_core. Ensure the C++ extension is built and installed correctly."
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

consumer: pie_core.ResponseReader | None = None  # type: ignore[assignment]
stop_consuming = False
request_completed_event = threading.Event()  # Use threading Event


def consume_responses_sync():
    """Synchronous function to consume responses."""
    global consumer, stop_consuming, request_completed_event
    if not consumer:
        logging.error("Consumer not initialized.")
        return

    logging.info("Response consumer thread started.")
    while not stop_consuming:
        try:
            # Use a timeout so the loop doesn't block forever if no data arrives
            delta: dict[str, Any] | None = consumer.consume_next_delta(timeout_ms=100)

            if delta:
                logging.info(f"Received Delta for Request ID: {delta['request_id']}")
                logging.info(f"  Tokens: {delta['tokens']}")
                logging.info(f"  Is Final: {delta['is_final_delta']}")
                logging.info(f"  Finish Reason: {delta['finish_reason']}")
                if delta["is_final_delta"]:
                    logging.info(f"--- Request {delta['request_id']} Completed ---")
                    request_completed_event.set()  # Signal completion
                    # Optionally break if only waiting for one request
                    # break

            # Check stop flag more frequently if polling
            if stop_consuming:
                break

        except Exception as e:
            logging.error(f"Error in consumer thread: {e}", exc_info=True)
            if stop_consuming:  # Avoid tight loop on error during shutdown
                break
            time.sleep(0.1)

    logging.info("Response consumer thread stopped.")


def main():
    global consumer, stop_consuming, request_completed_event
    request_id_counter = 1
    consumer_thread = None  # Initialize thread variable

    try:
        logging.info("Initializing Request Writer...")
        pie_core.init_request_writer()
        logging.info("Request Writer Initialized.")

        logging.info("Initializing Response Consumer...")
        consumer = pie_core.ResponseReader("/pie_response_slots")
        logging.info("Response Consumer Initialized.")

        # Start the consumer thread
        stop_consuming = False
        request_completed_event.clear()
        consumer_thread = threading.Thread(target=consume_responses_sync, daemon=True)
        consumer_thread.start()

        logging.info("Submitting test request...")
        req_id = request_id_counter
        request_id_counter += 1

        submitted_req_id = pie_core.submit_request(
            request_id=req_id,
            prompt_string="This is a test prompt.",
            # ... other params ...
            max_generated_tokens=5,
        )
        logging.info(f"Submitted request with ID: {submitted_req_id}")

        # Wait for the consumer thread to signal completion or timeout
        logging.info("Waiting for response completion...")
        completed = request_completed_event.wait(timeout=10.0)  # Wait up to 10 seconds

        if completed:
            logging.info("Request completion signaled by consumer.")
        else:
            logging.warning("Timeout waiting for request completion signal.")

    except Exception as e:
        logging.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        logging.info("Shutting down...")
        stop_consuming = True  # Signal consumer thread to stop
        if consumer_thread and consumer_thread.is_alive():
            logging.info("Joining consumer thread...")
            consumer_thread.join(timeout=2.0)
            if consumer_thread.is_alive():
                logging.warning("Consumer thread did not finish gracefully.")

        logging.info("Shutting down Request Writer...")
        pie_core.shutdown_request_writer()
        # No need to explicitly shutdown consumer if not globally managed
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    logging.info("--- Starting Python IPC Test Client ---")
    logging.info("--- Ensure the C++ pie_engine is running separately! ---")
    main()  # Run synchronously
    logging.info("--- Python IPC Test Client Finished ---")
