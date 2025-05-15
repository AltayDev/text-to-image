import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model


def setup_pipeline(pretrained_model_path):
    """Set up the UNet model for diffusion"""
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
    )

    # Ensure UNet is in float32 before applying LoRA
    unet.to(torch.float32)

    return unet


def prepare_unet_lora(unet, lora_rank, lora_alpha):
    """Apply LoRA to the UNet model"""
    # Character-focused target modules for LoRA - expanded for better character learning
    target_modules = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
    ]

    # Set up LoRA using PEFT library
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,  # Added small dropout for better generalization
        bias="none",
    )

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)

    # Only train the LoRA parameters and ensure they're in float32
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
        else:
            param.requires_grad = False

    return unet
