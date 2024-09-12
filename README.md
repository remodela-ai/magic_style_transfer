# Cog SDXL Depth ControlNet with LoRA support and IP-Adapter

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/batouresearch/sdxl-controlnet-lora)

This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model with ControlNet and Replicate's LoRA support.


This project combines Stable Diffusion XL (SDXL) with IP-Adapter to enhance image generation by allowing control over various visual styles using depth maps and canny edges. The models used include LoRA weights, IP-Adapter, and additional ControlNet functionality for depth and canny conditioning.

## Features

- **IP-Adapter Integration**: Generate detailed, controllable outputs using IP-Adapter combined with SDXL.
- **ControlNet Models**: Support for depth and canny edge conditioning for more structured image generation.
- **LoRA Fine-tuning**: Ability to load and unload LoRA weights for fine-tuning with different prompts and outputs.
- **Depth Map & Canny Edge Detection**: Automatically extracts and processes depth maps and canny edges from input images to enhance the conditioning of ControlNet.
- **Custom Schedulers**: Supports various schedulers including DDIM, DPMSolverMultistep, HeunDiscrete, and more.
- **Multiple Outputs**: Generate up to four images at a time, with control over inference steps, guidance scale, and other parameters.

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.9 or later
- Docker (optional, for easier environment setup)

## Installation

### Clone the repository
```bash
git clone https://github.com/<your-repo>/recognize-anything.git
cd recognize-anything
