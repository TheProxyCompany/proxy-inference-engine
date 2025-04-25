# Proxy Inference Engine (PIE)

<h3 align="center">
  <strong>Optimized MLX inference engine for Apple Silicon.</strong>
</h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/TheProxyCompany/proxy-inference-engine/main/logo.png" alt="Proxy Inference Engine" style="object-fit: contain; max-width: 50%; padding-top: 20px;"/>
</p>

**Proxy Inference Engine (PIE)** is the specialized inference layer developed by The Proxy Company, built upon the foundation of Apple's [MLX framework](https://github.com/ml-explore/mlx). It is designed for high-performance execution of language models (including VLMs) on Apple Silicon hardware.

This engine is the successor to the experimental `mlx-proxy` repository (now archived).

## Installation

```bash
pip install proxy-inference-engine
```

*Note: Requires Python 3.12+ and macOS with Apple Silicon hardware.*

## Features

- Efficient key-value caching strategies for transformer models
- Advanced logits processing for high-quality text generation
- Multiple sampling methods (categorical, top-k, top-p, min-p)
- Vision model support for multimodal inference
- Optimized for MLX and Apple Silicon
