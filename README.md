# Zephyr-7B Fine-Tuning Example

A lightweight implementation for fine-tuning the Zephyr-7B language model on consumer hardware.

## Features

- 🚀 CPU-optimized fine-tuning with 4-bit quantization
- 📊 Comprehensive logging for monitoring training progress
- 🛠️ LoRA (Low-Rank Adaptation) for efficient parameter updates
- 📝 Easy-to-modify template for custom datasets

## Prerequisites

- Python 3.8+
- Pip package manager
- 16GB+ RAM recommended

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install torch transformers datasets peft accelerate bitsandbytes sentencepiece