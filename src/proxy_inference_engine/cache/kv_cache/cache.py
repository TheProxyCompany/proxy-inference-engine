import mlx.core as mx

from proxy_inference_engine.cache.kv_cache import BaseCache
from proxy_inference_engine.cache.kv_cache.quantized import QuantizedKVCache


class KVCache(BaseCache):
    """
    A key-value cache for transformer models.

    This cache stores the key and value tensors from previous forward passes,
    allowing for efficient autoregressive generation by avoiding recomputation
    of previously processed tokens.
    """

    keys: mx.array | None
    values: mx.array | None
    offset: int
    step: int

    def __init__(self):
        """Initialize an empty KV cache with default parameters."""
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update the cache with new key-value pairs and return the full cache.

        This method dynamically resizes the cache as needed to accommodate new tokens.

        Args:
            keys: New key tensors to add to the cache
            values: New value tensors to add to the cache

        Returns:
            A tuple containing the full cached keys and values up to the current offset
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            needed = (
                prev + keys.shape[2] - (0 if self.keys is None else self.keys.shape[2])
            )
            n_steps = (needed + self.step - 1) // self.step
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None and self.values is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        assert self.keys is not None and self.values is not None
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """
        Get the current state of the cache.

        Returns:
            A tuple containing the cached keys and values, trimmed to the current offset.
            Returns (None, None) if the cache is empty.
        """
        if self.keys is None or self.values is None:
            return None, None

        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]):
        """
        Set the state of the cache.

        Args:
            v: A tuple containing the keys and values to set
        """
        self.keys, self.values = v
        self.offset = self.keys.shape[2] if self.keys is not None else 0

    def is_trimmable(self) -> bool:
        """
        Check if this cache can be trimmed.

        Returns:
            True, as KVCache supports trimming
        """
        return True

    def trim(self, n: int) -> int:
        """
        Trim the cache by reducing the offset.

        This effectively discards the oldest n tokens from the cache without
        actually modifying the underlying tensors.

        Args:
            n: Number of tokens to trim from the beginning of the cache

        Returns:
            The actual number of tokens trimmed
        """
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        """
        Convert this cache to a quantized version for memory efficiency.

        Quantization reduces memory usage by representing values with fewer bits,
        trading some precision for significant memory savings.

        Args:
            group_size: Number of elements per quantization group
            bits: Number of bits to use for quantization (4 or 8)

        Returns:
            A new QuantizedKVCache containing the quantized version of this cache
        """
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None and self.values is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache
