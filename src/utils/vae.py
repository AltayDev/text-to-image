import torch
from diffusers import AutoencoderKL


def encode_vae(pixel_values, model_path, mixed_precision, device):
    """Encode images to latent space using the VAE"""
    # Determine compute dtype
    dtype = torch.float16 if mixed_precision in ["fp16", "bf16"] else torch.float32

    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    # Explicitly convert VAE parameters to the correct dtype first
    for param in vae.parameters():
        param.data = param.data.to(dtype)

    # Then move the model to device
    vae = vae.to(device)

    # Set to eval mode
    vae.eval()

    # Make sure input and model are in the same precision
    with torch.no_grad():
        pixel_values = pixel_values.to(device, dtype=dtype)
        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        # Ensure consistent dtype for latents
        latents = latents.to(dtype=dtype)

    return latents
