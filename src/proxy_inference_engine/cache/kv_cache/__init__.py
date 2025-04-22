from __future__ import annotations

from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class BaseCache(ABC):
    """
    Abstract base class for transformer caching mechanisms.

    This class defines the interface that all cache implementations must adhere to.
    Subclasses should override these methods to provide specific caching behaviors
    for different use cases (e.g., standard KV caching, quantized caching, rotating caches).
    """

    offset: int
    step: int

    @staticmethod
    def make_kv_cache(
        model: nn.Module,
        max_kv_size: int | None = None,
        reusable: bool = False,
    ) -> list[BaseCache]:
        """
        Construct the model's key-value cache for use during generation.

        This function will defer the cache construction to the model if it has a
        ``make_cache`` method, otherwise it will make a default KV cache.

        Args:
            model (nn.Module): The language model.
            max_kv_size (Optional[int]): If provided and the model does not have a
                ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
                size of ``max_kv_size``
        """
        if hasattr(model, "make_cache") and model.make_cache is not None:
            return model.make_cache()

        num_layers = len(model.layers) if model.layers else 0
        if max_kv_size is not None:
            return [
                RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
            ]
        elif reusable:
            return [ReusableKVCache() for _ in range(num_layers)]
        else:
            return [KVCache() for _ in range(num_layers)]

    @property
    def state(self):
        """
        Get the current state of the cache.

        Returns:
            An empty list by default, indicating no state is maintained.
            Subclasses should override to return their specific state representation.
        """
        return []

    @state.setter
    def state(self, v):
        """
        Set the state of the cache.

        Args:
            v: The state to set.

        Raises:
            ValueError: If attempting to set a state on a cache that doesn't support it.
        """
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        """
        Get metadata about the cache state.

        Returns:
            An empty string by default. Subclasses should override to return
            relevant metadata about their cache implementation.
        """
        return ""

    @meta_state.setter
    def meta_state(self, v):
        """
        Set metadata about the cache state.

        Args:
            v: The metadata to set.

        Raises:
            ValueError: If attempting to set metadata on a cache that doesn't support it.
        """
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        """
        Check if this cache supports trimming operations.

        Returns:
            False by default. Subclasses should override to return True if they
            support trimming operations to manage cache size.
        """
        return False

    @abstractmethod
    def trim(self, n: int) -> int:
        """
        Trim the cache by a specified number of tokens.

        Args:
            n: The number of tokens to trim.

        Returns:
            The number of tokens actually trimmed.  Subclasses should implement
            the specific trimming logic.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement trim")

    @abstractmethod
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update the cache and fetch the current state.

        Args:
            keys: The new keys to add to the cache.
            values: The new values to add to the cache.

        Returns:
            The updated keys and values from the cache. Subclasses should
            implement the specific update and retrieval logic.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement update_and_fetch")

    @abstractmethod
    def to_quantized(self, group_size: int = 64, bits: int = 4) -> BaseCache:
        """Convert to a quantized representation.

        Args:
            group_size: elements per group
            bits: number of bits per weight

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement to_quantized")

    @staticmethod
    def save_cache(
        file_name: str,
        cache: list[BaseCache],
        metadata: dict[str, str] | None = None,
    ):
        """
        Save a pre-computed key-value cache to a file.

        Args:
            file_name (str): The ``.safetensors`` file name.
            cache (List[BaseCache]): The model's KV cache.
            metadata (Dict[str, str]): Optional metadata to save along with model
                state.
        """
        metadata = metadata or {}
        cache_data = [c.state for c in cache]
        cache_info = [c.meta_state for c in cache]
        cache_data = dict(tree_flatten(cache_data))
        cache_classes = [type(c).__name__ for c in cache]
        cache_metadata = [cache_info, metadata, cache_classes]
        cache_metadata = dict(tree_flatten(cache_metadata))
        mx.save_safetensors(file_name, cache_data, cache_metadata)

    @staticmethod
    def load_cache(file_name: str) -> tuple[list[BaseCache], dict[str, str]]:
        """
        Load a key-value cache from a file.

        Args:
            file_name (str): The ``.safetensors`` file name.
            return_metadata (bool): Whether to return metadata. Default: ``False``.

        Returns:
            List[BaseCache] or Tuple[List[BaseCache], Dict[str, str]]: The key-value cache and
                the metadata if requested.
        """
        arrays, cache_metadata = mx.load(file_name, return_metadata=True)
        arrays = tree_unflatten(list(arrays.items()))
        cache_metadata = tree_unflatten(list(cache_metadata.items()))
        info, metadata, classes = cache_metadata
        cache = [globals()[c]() for c in classes]
        for c, state, meta_state in zip(cache, arrays, info):
            assert isinstance(c, BaseCache)
            c.state = state
            c.meta_state = meta_state
  
        return cache, metadata

    @staticmethod
    def can_trim(cache: list[BaseCache]) -> bool:
        """
        Check if model's key-value cache can be trimmed.

        Args:
            cache (List[BaseCache]): The model's key-value cache.

        Returns:
            bool: True if all caches in the list are trimmable.
        """
        return all(c.is_trimmable() for c in cache)

    @staticmethod
    def trim_cache(cache: list[BaseCache], num_tokens: int) -> int:
        """
        Trim the model's key-value cache by the given number of tokens.

        Args:
            cache (List[BaseCache]): The model's key-value cache.
            num_tokens (int): The number of tokens to trim.

        Returns:
            int: The number of tokens that were trimmed.
        """
        if not BaseCache.can_trim(cache) or len(cache) == 0:
            return 0
        return next(c.trim(num_tokens) for c in cache)

    @staticmethod
    def maybe_quantize(
        cache: list[BaseCache],
        quantized_start: int,
        group_size: int,
        bits: int | None,
    ):
        """
        Quantize the key-value cache if conditions are met.

        Args:
            cache (list[BaseCache]): The model's key-value cache.
            quantized_start (int): Token position after which to start quantization.
            group_size (int): Number of elements per quantization group.
            bits (int | None): Number of bits for quantization (4 or 8).
                If None, no quantization is performed.
        """
        if (
            bits is not None
            and not isinstance(cache[0], QuantizedKVCache)
            and cache[0].offset > quantized_start
        ):
            for i in range(len(cache)):
                if isinstance(cache[i], BaseCache):
                    cache[i] = cache[i].to_quantized(group_size=group_size, bits=bits)


from proxy_inference_engine.cache.kv_cache.cache import KVCache  # noqa: E402
from proxy_inference_engine.cache.kv_cache.quantized import QuantizedKVCache  # noqa: E402
from proxy_inference_engine.cache.kv_cache.reusable import ReusableKVCache  # noqa: E402
from proxy_inference_engine.cache.kv_cache.rotating import RotatingKVCache  # noqa: E402
