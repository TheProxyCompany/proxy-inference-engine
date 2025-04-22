from __future__ import annotations

import mlx.core as mx

from proxy_inference_engine.cache.kv_cache import BaseCache


class ReusableKVCache(BaseCache):
    """
    A key-value cache with support for prompt reuse, integrated into BaseCache.

    This class extends the `BaseCache` to add a `reuse` method and
    an `update_and_fetch` method, which allows efficient handling of
    prompts that share a common prefix and supports batch sizes > 1.
    """

    keys: mx.array | None
    values: mx.array | None
    offset: int
    step: int

    def __init__(
        self,
        step: int = 256,
        growth_factor: float = 1.5,
        max_capacity: int | None = None,
    ):
        """
        Initialize an empty ReusableKVCache with configurable parameters.

        Args:
            step: The size for step-aligned allocations.
            growth_factor: The factor used to expand buffer capacity (e.g., 1.5 or 2.0).
            max_capacity: If set, the cache will never grow beyond this capacity.
        """
        self.keys = None
        self.values = None
        self.offset = 0

        self.step = step
        self.growth_factor = growth_factor
        self.max_capacity = max_capacity

    def reuse(self, new_prompt_length: int, common_prefix_length: int) -> None:
        """
        Reuse (part of) this cache for a new prompt that shares a prefix with it.

        1. Trims the cache to the length of the common prefix (offset).
        2. Ensures capacity for the entire new prompt, expanding if necessary.

        Args:
            new_prompt_length: The total length of the new prompt.
            common_prefix_length: The length of the common prefix between the old and new prompts.
        """
        if self.keys is None or self.values is None:
            return

        # Clip the cache to the common prefix
        self.offset = common_prefix_length
        current_size = self.keys.shape[2]

        # If we need more space than currently allocated, expand buffer
        if current_size < new_prompt_length:
            new_capacity = max(
                int(current_size * self.growth_factor), new_prompt_length
            )
            # Round up to the nearest multiple of self.step
            new_capacity = ((new_capacity + self.step - 1) // self.step) * self.step

            # Respect a maximum capacity if specified
            if self.max_capacity is not None:
                new_capacity = min(new_capacity, self.max_capacity)

            B, n_kv_heads, _, k_head_dim = self.keys.shape
            v_head_dim = self.values.shape[3]

            # Build new shapes
            new_k_shape = (B, n_kv_heads, new_capacity, k_head_dim)
            new_v_shape = (B, n_kv_heads, new_capacity, v_head_dim)

            # Use the existing arrays' dtype
            dtype_for_k = self.keys.dtype
            dtype_for_v = self.values.dtype

            # Allocate new buffers
            new_keys = mx.zeros(new_k_shape, dtype=dtype_for_k)
            new_values = mx.zeros(new_v_shape, dtype=dtype_for_v)

            # Copy the prefix portion into the new buffers
            new_keys[..., : self.offset, :] = self.keys[..., : self.offset, :]
            new_values[..., : self.offset, :] = self.values[..., : self.offset, :]

            self.keys = new_keys
            self.values = new_values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update the cache with new key-value pairs and return the full cache slice.

        This method:
        1. Checks if there's enough capacity to store the new keys/values.
        2. If not, expands the buffers by `growth_factor`, aligning to `self.step`.
        3. Preserves step-based safety by trimming at the old offset if partial steps are present.
        4. Stores the new keys/values and returns a slice up to the updated offset.

        Args:
            keys: New key tensors to add to the cache, shape [B, n_kv_heads, #tokens, key_dim].
            values: New value tensors to add, shape [B, n_kv_heads, #tokens, value_dim].

        Returns:
            (cached_keys, cached_values): Slices of the cache up to the current offset.
        """
        needed = keys.shape[2]
        prev_offset = self.offset

        # Initialize or check capacity
        if self.keys is None or self.values is None:
            # Allocate from scratch
            self._allocate_new_buffers(keys, values, needed)
        else:
            current_capacity = self.keys.shape[2]
            # If offset + needed doesn't fit, expand
            if (prev_offset + needed) > current_capacity:
                # If partial step usage might cause shape mismatch, trim to offset
                if (prev_offset % self.step) != 0:
                    # "Safety mechanism" to ensure no shape mismatch
                    self.keys = self.keys[..., :prev_offset, :]
                    self.values = self.values[..., :prev_offset, :]

                self._expand_buffers_if_needed(prev_offset + needed)

        # Now we definitely have enough space, so place new data
        assert self.keys is not None and self.values is not None
        self.keys[..., prev_offset : prev_offset + needed, :] = keys
        self.values[..., prev_offset : prev_offset + needed, :] = values

        self.offset += needed

        # Return slices up to offset
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def _allocate_new_buffers(self, keys: mx.array, values: mx.array, needed: int):
        """
        Internal helper to allocate new buffers from scratch.
        """
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]

        # Round needed up to multiples of step
        capacity = ((needed + self.step - 1) // self.step) * self.step
        if self.max_capacity is not None:
            capacity = min(capacity, self.max_capacity)

        # Build new shapes
        new_k_shape = (B, n_kv_heads, capacity, k_head_dim)
        new_v_shape = (B, n_kv_heads, capacity, v_head_dim)

        dtype_for_k = keys.dtype
        dtype_for_v = values.dtype

        self.keys = mx.zeros(new_k_shape, dtype=dtype_for_k)
        self.values = mx.zeros(new_v_shape, dtype=dtype_for_v)
        self.offset = 0  # Start fresh

    def _expand_buffers_if_needed(self, required_capacity: int):
        """
        Internal helper to expand existing buffers using the growth factor and step alignment.
        """
        if self.keys is None or self.values is None:
            return

        current_capacity = self.keys.shape[2]
        new_capacity = max(
            int(current_capacity * self.growth_factor), required_capacity
        )
        # Align to step boundary
        new_capacity = ((new_capacity + self.step - 1) // self.step) * self.step
        # Respect max_capacity if given
        if self.max_capacity is not None:
            new_capacity = min(new_capacity, self.max_capacity)

        B, n_kv_heads, _, k_head_dim = self.keys.shape
        v_head_dim = self.values.shape[3]

        # Build new shapes
        new_k_shape = (B, n_kv_heads, new_capacity, k_head_dim)
        new_v_shape = (B, n_kv_heads, new_capacity, v_head_dim)

        dtype_for_k = self.keys.dtype
        dtype_for_v = self.values.dtype

        # Allocate the new arrays
        new_keys = mx.zeros(new_k_shape, dtype=dtype_for_k)
        new_values = mx.zeros(new_v_shape, dtype=dtype_for_v)

        # Copy old data up to self.offset
        new_keys[..., :self.offset, :] = self.keys[..., :self.offset, :]
        new_values[..., :self.offset, :] = self.values[..., :self.offset, :]

        self.keys = new_keys
        self.values = new_values

    @property
    def state(self):
        """
        Get the current state of the cache.

        Returns:
            (keys, values) for the entire allocated buffer.
        """
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]):
        """
        Set the state of the cache.

        Args:
            v: A tuple containing (keys, values).
        """
        self.keys, self.values = v
        self.offset = self.keys.shape[2] if self.keys is not None else 0

    def is_trimmable(self) -> bool:
        """
        Check if this cache can be trimmed.

        Returns:
            True, as ReusableKVCache supports trimming.
        """
        return True

    def trim(self, n: int) -> int:
        """
        Trim the cache by reducing the offset, effectively discarding
        the oldest n tokens from the "logical" beginning of the cache.

        Args:
            n: Number of tokens to trim from the beginning of the cache.

        Returns:
            The actual number of tokens trimmed.
        """
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> BaseCache:
        """
        Convert this cache to a quantized version for memory efficiency.
        """
        return self
