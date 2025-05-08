import asyncio
import logging
import sys
from typing import Any

from proxy_inference_engine.pie_core import ResponseReader

logger = logging.getLogger(__name__)

# --- Constants ---
DISPATCHER_POLL_INTERVAL_MS = 10  # How often to poll C++ queue (milliseconds)
DISPATCHER_IDLE_SLEEP_S = 0.005  # Sleep duration when queue is empty (seconds)
QUEUE_PUT_TIMEOUT_S = 2.0  # Max time to wait putting item onto handler queue
SHUTDOWN_TASK_TIMEOUT_S = 5.0  # Max time to wait for dispatcher task shutdown

# Type alias for the delta dictionary received from C++
ResponseDeltaDict = dict[str, Any]


# --- Shared State ---
class IPCState:
    """Holds shared state related to IPC components."""

    def __init__(self):
        self.response_reader: ResponseReader | None = None
        self.active_request_queues: dict[int, asyncio.Queue[ResponseDeltaDict]] = {}
        self.request_id_counter: int = 0
        self.dispatcher_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def get_next_request_id(self) -> int:
        """Atomically increments and returns the next request ID."""
        async with self._lock:
            self.request_id_counter += 1
            # Basic overflow check (unlikely in practice but good hygiene)
            if self.request_id_counter > sys.maxsize - 100:
                logger.warning("Request ID counter nearing overflow, resetting.")
                self.request_id_counter = 1
            return self.request_id_counter


# --- Core Dispatcher Logic ---
async def run_response_dispatcher(ipc_state: IPCState) -> None:
    """
    Dedicated asyncio task that continuously polls the C++ ResponseReader SHM queue
    and dispatches incoming response deltas to the appropriate request-specific
    asyncio.Queue.
    """
    logger.info("Response dispatcher task starting...")
    if not ipc_state.response_reader:
        # This check should ideally be redundant due to lifespan handling, but belt-and-suspenders.
        logger.critical("ResponseReader not initialized. Dispatcher cannot run.")
        return

    reader = ipc_state.response_reader
    active_queues = ipc_state.active_request_queues
    poll_timeout_ms = DISPATCHER_POLL_INTERVAL_MS

    while True:
        delta: ResponseDeltaDict | None = None
        try:
            # Poll C++ queue using asyncio.to_thread for the potentially blocking call
            delta = await asyncio.to_thread(reader.consume_next_delta, poll_timeout_ms)

            if delta:
                req_id = delta.get("request_id")
                if req_id is None:
                    logger.warning("Received delta with no request_id: %s", delta)
                    continue  # Skip malformed delta

                if req_id in active_queues:
                    dest_queue = active_queues[req_id]
                    try:
                        # Put delta onto the specific handler's queue with a timeout
                        await asyncio.wait_for(
                            dest_queue.put(delta), timeout=QUEUE_PUT_TIMEOUT_S
                        )
                        logger.debug("Dispatched delta for request_id %d", req_id)
                    except TimeoutError:
                        # This indicates the handler's queue might be full or the handler
                        # is blocked/stuck and not consuming. This is an app-level issue.
                        logger.error(
                            "Timeout putting delta onto asyncio queue for request_id %d. "
                            "Associated handler might be unresponsive or queue full. Delta lost for this request.",
                            req_id,
                        )
                    except Exception as e:
                        # Catch unexpected errors during the put operation
                        logger.error(
                            "Unexpected error putting delta onto queue for request_id %d: %s",
                            req_id,
                            e,
                            exc_info=True,
                        )
                else:
                    # Handler likely finished/cancelled and cleaned up its queue before this delta arrived.
                    logger.warning(
                        "Received delta for unknown/cleaned-up request_id %d. Discarding delta.",
                        req_id,
                    )
            else:
                # No delta received (timeout from C++ poll), yield control briefly
                await asyncio.sleep(DISPATCHER_IDLE_SLEEP_S)

        # Handle potential errors during the C++ call itself
        except RuntimeError as e:
            if "Response SHM not initialized" in str(e):
                logger.critical(
                    "Response SHM error in dispatcher: %s. Stopping dispatcher.", e
                )
                break
            else:
                logger.error(
                    "Runtime error polling response queue: %s", e, exc_info=True
                )
                await asyncio.sleep(0.1)
        except Exception as e:
            # Catch any other unexpected errors in the loop
            logger.error("Unexpected error in dispatcher loop: %s", e, exc_info=True)
            await asyncio.sleep(0.1)

        try:
            await asyncio.sleep(
                0
            )  # Yield briefly to allow cancellation to be processed
        except asyncio.CancelledError:
            logger.info("Dispatcher task cancellation requested.")
            break

    logger.info("Response dispatcher task finished.")
