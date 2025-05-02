from typing import Any

from proxy_inference_engine.cache.kv_cache import BaseCache


class PagedKVCache(BaseCache):
    """
    A key-value cache that uses blocks of memory to store the cache,
    and a page manager to manage the memory.
    """

    def __init__(self, page_manager: Any):
        """Initialize a PagedKVCache with a page manager."""
        self.page_manager = page_manager
