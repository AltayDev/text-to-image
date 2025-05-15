# Text-to-Image Fine-Tuning with Stable Diffusion

Created by Altay

This repository contains a framework for fine-tuning Stable Diffusion models on custom datasets, optimized for RTX GPUs. The project uses LoRA (Low-Rank Adaptation) for memory-efficient training.

## 📋 Project Structure

```
├── config/                  # Configuration files
│   ├── default_config.json  # Default training configuration
│   └── test_config.json     # Test configuration (created by test script)
├── dataset/                 # Your custom dataset (images + captions)
├── output/                  # Training outputs
│   ├── samples/             # Generated samples during training
│   ├── logs/                # Training logs
│   └── lora/                # LoRA weights
├── scripts/                 # Batch scripts for common operations
│   ├── train.bat            # Full training script
│   ├── generate.bat         # Image generation script
│   ├── test_train.bat       # Quick test training
│   └── test_cuda.bat        # CUDA/GPU testing script
└── src/                     # Source code
    ├── train.py             # Main training script
    ├── generate.py          # Image generation script
    ├── models/              # Model-related code
    │   └── stable_diffusion.py # Stable Diffusion model setup
    └── utils/               # Utility functions
        ├── checkpoint.py    # Model checkpointing
        ├── cuda_test.py     # CUDA testing
        ├── dataset.py       # Custom dataset implementation
        ├── generate.py      # Sample generation during training
        ├── logging.py       # Logging utilities
        └── vae.py           # VAE encoder utilities
```

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+ with pip

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Test your CUDA installation:

```bash
scripts/test_cuda.bat
```

3. Prepare your dataset in the `dataset` directory:
   - PNG images with corresponding TXT files having the same name
   - Each TXT file contains a single line caption describing the image

## 🔧 Training

### Quick Test

Run a short training test to verify everything is set up correctly:

```bash
scripts/test_train.bat
```

### Full Training

Start the full training process:

```bash
scripts/train.bat
```

### Training Parameters

All training parameters can be modified in `config/default_config.json`:

- `model`: Base model configuration
- `training`: Training parameters (learning rate, batch size, etc.)
- `generation`: Parameters for image generation
- `sample_prompts`: Prompts to use for generating samples during training

## 🖼️ Generating Images

After training, generate images with your fine-tuned model:

```bash
scripts/generate.bat
```

## 📊 How It Works

### Stable Diffusion Fine-Tuning Process

The `train_stable_diffusion.py` script implements a fine-tuning pipeline optimized for efficiency:

1. **Data Loading**: Uses a custom dataset to load images and captions
2. **LoRA Adaptation**: Applies Low-Rank Adaptation to the UNet part of Stable Diffusion
3. **Optimization**: Uses 8-bit Adam and mixed precision to save memory
4. **Monitoring**: Generates sample images throughout training to visualize progress
5. **Checkpointing**: Saves the best model based on loss and visual quality

### Memory Optimizations

- **LoRA fine-tuning**: Only updates a small set of parameters
- **8-bit Adam optimizer**: Reduces optimizer memory usage
- **Mixed precision (fp16)**: Uses half-precision for calculations
- **Gradient accumulation**: Simulates larger batch sizes
- **Attention slicing**: Reduces peak memory usage during attention computation

## 📝 Customization

### Training Your Own Concept

1. Replace the sample dataset with your own images
2. Update sample prompts in the config file
3. Adjust training parameters based on your GPU capabilities
4. Run the training script

## 🔍 Troubleshooting

### Common Issues

- **Out of memory errors**: Reduce batch size or increase gradient accumulation steps
- **Poor quality results**: Increase training steps or adjust learning rate
- **CUDA errors**: Update your GPU drivers or check CUDA compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- This project uses [Diffusers](https://github.com/huggingface/diffusers) library by Hugging Face
- LoRA implementation from [PEFT](https://github.com/huggingface/peft)
