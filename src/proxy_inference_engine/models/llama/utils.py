import mlx.core as mx
import mlx.nn as nn


class Llama3RoPE(nn.Module):
    """
    RoPE implementation for Llama 3
    """

    def __init__(
        self,
        max_embedding_length: int,
        global_embedding_length: int,
        dimensions: int,
        base: float,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        traditional: bool = False,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.max_position_embeddings = max_embedding_length
        self.traditional = traditional

        low_freq_wavelen = global_embedding_length / low_freq_factor
        high_freq_wavelen = global_embedding_length / high_freq_factor

        freqs = base ** (mx.arange(0, dimensions, 2) / dimensions)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)

        smooth_factors = (max_embedding_length / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dimensions,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )
