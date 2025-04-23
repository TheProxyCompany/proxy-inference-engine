import mlx.core as mx
from mlx.utils import tree_map

from proxy_inference_engine.cache.kv_cache import BaseCache


class QuantizedKVCache(BaseCache):
    """
    A memory-efficient key-value cache using quantization for transformer models.

    This cache stores quantized key and value tensors from previous forward passes,
    reducing memory usage by representing values with fewer bits while trading
    some precision for significant memory savings.
    """
    keys: tuple[mx.array, mx.array, mx.array] | None
    values: tuple[mx.array, mx.array, mx.array] | None
    offset: int
    step: int
    group_size: int
    bits: int

    def __init__(self, group_size: int = 64, bits: int = 8):
        """
        Initialize an empty quantized KV cache.

        Args:
            group_size: Number of elements per quantization group
            bits: Number of bits to use for quantization (4 or 8)
        """
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[
        tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]
    ]:
        """
        Update the cache with new key-value pairs and return the full quantized cache.

        This method dynamically resizes the cache as needed to accommodate new tokens,
        and quantizes the new keys and values before storing them.

        Args:
            keys: New key tensors to add to the cache
            values: New value tensors to add to the cache

        Returns:
            A tuple containing the full quantized cached keys and values up to the current offset
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim: int) -> tuple[mx.array, mx.array, mx.array]:
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(
                x: mx.array,
            ) -> mx.array:
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys_quant, keys_scale, keys_bias = mx.quantize(
            keys, group_size=self.group_size, bits=self.bits
        )
        values_quant, values_scale, values_bias = mx.quantize(
            values, group_size=self.group_size, bits=self.bits
        )

        # Ensure cache exists before assignment (type checker placation)
        assert self.keys is not None and self.values is not None

        # Tuple wrangling without index acrobatics
        for key_arr, quant_val in zip(self.keys, (keys_quant, keys_scale, keys_bias)):
            key_arr[..., prev : self.offset, :] = quant_val
        for val_arr, quant_val in zip(self.values, (values_quant, values_scale, values_bias)):
            val_arr[..., prev : self.offset, :] = quant_val

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(
        self,
    ) -> tuple[
        tuple[mx.array, mx.array, mx.array] | None,
        tuple[mx.array, mx.array, mx.array] | None,
    ]:
        """
        Get the current state of the quantized cache.

        Returns:
            A tuple containing the quantized cached keys and values, trimmed to the current offset.
            Returns (None, None) if the cache is empty.
        """
        if self.keys is None:
            return None, None
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(
        self,
        v: tuple[
            tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]
        ],
    ) -> None:
        """
        Set the state of the quantized cache.

        Args:
            v: A tuple containing the quantized keys and values to set
        """
        self.keys, self.values = v
        if self.keys is not None:
            self.offset = self.keys[0].shape[-2]

    @property
    def meta_state(self) -> tuple[str, ...]:
        """
        Get metadata about the quantized cache state.

        Returns:
            A tuple of strings containing the step size, offset, group size, and bits
        """
        return tuple(map(str, (self.step, self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        """
        Set metadata about the quantized cache state.

        Args:
            v: A tuple of strings containing the step size, offset, group size, and bits
        """
        self.step, self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self) -> bool:
        """
        Check if this cache can be trimmed.

        Returns:
            True, as QuantizedKVCache supports trimming
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

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> "QuantizedKVCache":
        """
        Convert this cache to a quantized version for memory efficiency.

        Args:
            group_size: Number of elements per quantization group
            bits: Number of bits to use for quantization (4 or 8)

        Returns:
            A new QuantizedKVCache containing the quantized version of this cache
        """
        return self
