import mlx.core as mx

from proxy_inference_engine.cache.kv_cache import BaseCache


class RotatingKVCache(BaseCache):
    """
    A memory-efficient key-value cache with fixed maximum size for transformer models.

    This cache maintains a fixed-size buffer by rotating out older tokens when the
    maximum size is reached, while preserving a configurable number of initial tokens
    to maintain important context. It supports both single-token and multi-token updates
    with different strategies for each case.
    """
    keys: mx.array | None
    values: mx.array | None
    offset: int
    max_size: int
    step: int
    keep: int
    _idx: int | None

    def __init__(self, max_size: int, keep: int = 0, step: int = 256):
        """
        Initialize a rotating KV cache with a maximum size.

        Args:
            max_size: Maximum number of tokens to store in the cache
            keep: Number of initial tokens to always preserve (e.g., prompt tokens)
            step: Size increment when expanding the cache
        """
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size: int, v: mx.array, append: mx.array | None = None) -> mx.array:
        """
        Trim the cache by removing tokens while preserving the initial 'keep' tokens.

        Args:
            trim_size: Number of tokens to trim
            v: The array to trim
            append: Optional array to append after trimming

        Returns:
            The trimmed array, possibly with appended content
        """
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v: mx.array) -> mx.array:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.

        Args:
            v: The array to rearrange

        Returns:
            The rearranged array in proper temporal order
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx is not None and self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Update the cache by concatenating new keys and values (for multi-token updates).

        Args:
            keys: New key tensors to add to the cache
            values: New value tensors to add to the cache

        Returns:
            A tuple containing the updated cached keys and values
        """
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)

            # The largest size is self.max_size + S to ensure
            # every token gets at least self.max_size context
            if self._idx is not None:
                trim_size = self._idx - (self.max_size or 0)
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2] if self.keys is not None else None
        return self.keys, self.values

    def _update_in_place(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Update the cache in-place (for single-token updates), rotating as needed.

        Args:
            keys: New key tensors to add to the cache
            values: New value tensors to add to the cache

        Returns:
            A tuple containing the updated cached keys and values
        """
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, (self.max_size or 0) - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None and self.values is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = (self.keys.shape[2] - self.max_size) if self.keys is not None else 0
        if trim_size > 0 and self.keys is not None and self.values is not None:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        assert self.keys is not None and self.values is not None
        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update the cache with new key-value pairs and return the full cache.

        This method chooses between in-place updates (for single tokens) and
        concatenation (for multiple tokens), and handles rotation when the cache
        reaches its maximum size.

        Args:
            keys: New key tensors to add to the cache
            values: New value tensors to add to the cache

        Returns:
            A tuple containing the full cached keys and values
        """
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

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
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        """
        Set the state of the cache.

        Args:
            v: A tuple containing the keys and values to set
        """
        self.keys, self.values = v

    @property
    def meta_state(self) -> tuple[str, ...]:
        """
        Get metadata about the cache state.

        Returns:
            A tuple of strings containing the cache configuration and state information
        """
        return tuple(
            map(str, (self.keep, self.max_size, self.step, self.offset, self._idx))
        )

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        """
        Set metadata about the cache state.

        Args:
            v: A tuple of strings containing the cache configuration and state information
        """
        self.keep, self.max_size, self.step, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self) -> bool:
        """
        Check if this cache can be trimmed.

        Returns:
            True if the cache can be trimmed, False otherwise
        """
        if self.keys is None or self.values is None:
            return False
        return self.offset < (self.max_size or 0)

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
        if self._idx is not None:
            self._idx -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        """
        Convert this cache to a quantized version for memory efficiency.

        Not yet implemented for RotatingKVCache.

        Args:
            group_size: Number of elements per quantization group
            bits: Number of bits to use for quantization (4 or 8)

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        return self
