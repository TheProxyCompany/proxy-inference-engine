import hashlib
import json
import logging
import pathlib

import mlx.core as mx
import mlx.nn as nn

from proxy_inference_engine.cache.kv_cache import BaseCache, ReusableKVCache

logger = logging.getLogger(__name__)

class PromptCache:
    """
    A class for caching system prompts and KV caches.

    This class provides methods for caching system prompts and KV caches to a file,
    as well as loading cached system prompts from a file.
    """

    def __init__(
        self,
        directory: str | pathlib.Path | None = None,
        cache: list[BaseCache] | None = None,
        computed_ids: mx.array | None = None,
    ):
        self.cache_directory = self._get_cache_directory(directory)
        self.cache: list[BaseCache] = cache or []
        self.computed_ids: mx.array = computed_ids or mx.array([])

    def __call__(self, prompt_ids: mx.array) -> mx.array:
        return self.reuse_cache(prompt_ids)

    def create_kv_cache(self, model: nn.Module) -> None:
        if hasattr(model, "make_cache") and callable(model.make_cache):
            self.cache = model.make_cache()
            return

        assert hasattr(model, "layers") and isinstance(model.layers, list), "Model must have a layers attribute"
        layer_count = len(model.layers)
        self.cache = [ReusableKVCache() for _ in range(layer_count)]

    def update(self, prompt_ids: mx.array) -> None:
        """
        Update the cache with the given prompt IDs.
        """
        if self.computed_ids.size == 0:
            self.computed_ids = prompt_ids
        else:
            self.computed_ids = mx.concat([self.computed_ids, prompt_ids])

    def reuse_cache(
        self,
        prompt_ids: mx.array,
    ) -> mx.array:
        """
        Reuse the cache for the given prompt and precomputed ids.
        """

        if not self.cache or self.computed_ids.size == 0:
            return prompt_ids

        common_prefix = 0
        for i, id in enumerate(self.computed_ids):
            if i >= len(prompt_ids) - 1 or prompt_ids[i] != id:
                break
            common_prefix += 1

        if common_prefix == 0:
            return prompt_ids

        for layer_cache in self.cache:
            assert isinstance(layer_cache, ReusableKVCache)
            layer_cache.reuse(len(prompt_ids), common_prefix)

        return prompt_ids[common_prefix:]

    def cache_prompt(self) -> None:
        """
        Cache the prompt token IDs and KV cache to a file.

        Args:
            token_ids (mx.array): The token IDs to cache.
        """
        try:
            # Get the cache directory and compute the hash
            prompt_hash = self._compute_prompt_hash(self.computed_ids)
            cache_path = self.cache_directory / f"{prompt_hash}.safetensors"
            # Save the cache
            BaseCache.save_cache(
                str(cache_path),
                self.cache,
                {"computed_ids": json.dumps(self.computed_ids.tolist())}
            )
            logger.debug(f"Cached system prompt to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache system prompt: {e}")

    def load_cached_prompt(self, token_ids: mx.array) -> None:
        """
        Load a cached prompt if available.

        Args:
            token_ids (mx.array): The token IDs to look up in the cache.

        Returns:
            Optional[tuple[list[BaseCache], mx.array]]: The cached token IDs and KV cache if available, None otherwise.
        """

        try:
            # Get the cache directory and compute the hash
            prompt_hash = self._compute_prompt_hash(token_ids)
            cache_path = self.cache_directory / f"{prompt_hash}.safetensors"
            # Check if the cache file exists
            if not cache_path.exists():
                logger.debug(f"No cache found for prompt hash {prompt_hash}")
                return
            cache, metadata = BaseCache.load_cache(str(cache_path))
            computed_ids = json.loads(metadata["computed_ids"])
            assert isinstance(computed_ids, list)
            self.computed_ids = mx.array(computed_ids)
            self.cache = cache

        except Exception as e:
            logger.error(f"Failed to load cached system prompt: {e}")

    def _get_cache_directory(self, directory: str | pathlib.Path | None = None) -> pathlib.Path:
        """
        Get the cache directory path, creating it if it doesn't exist.

        Args:
            directory (str | pathlib.Path | None): The directory to use for the cache.

        Returns:
            pathlib.Path: The path to the cache directory.
        """
        if isinstance(directory, str):
            directory = pathlib.Path(directory)

        if isinstance(directory, pathlib.Path):
            return directory

        module_dir = pathlib.Path(__file__).parent.absolute()
        cache_dir = module_dir / ".cache"

        # Create the cache directory if it doesn't exist
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")

        return cache_dir

    def _compute_prompt_hash(self, token_ids: mx.array) -> str:
        """
        Compute a hash of the token IDs to use as the cache key.

        This function creates a deterministic hash by taking the first half of the token sequence
        and converting it to a JSON string before hashing.

        Args:
            token_ids (mx.array): The token IDs to hash.

        Returns:
            str: A hexadecimal hash digest.
        """
        hash_obj = hashlib.sha256(str(token_ids.tolist()).encode())
        return hash_obj.hexdigest()
