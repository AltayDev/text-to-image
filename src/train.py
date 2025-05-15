import os
import argparse
import torch
import random
import json
from datetime import datetime
from utils.dataset import CustomDataset
from utils.logging import log_message
from utils.vae import encode_vae
from models.stable_diffusion import setup_pipeline, prepare_unet_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the training data (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the fine-tuned model (overrides config)",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps (overrides config)",
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Override config with command line arguments
    if args.data_dir:
        config["training"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.max_train_steps:
        config["training"]["max_train_steps"] = args.max_train_steps

    # Import necessary modules
    from accelerate import Accelerator
    from diffusers import DDPMScheduler
    from transformers import CLIPTokenizer, CLIPTextModel
    from tqdm.auto import tqdm

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
    )

    # Create output directories
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    samples_dir = os.path.join(config["training"]["output_dir"], "samples")
    os.makedirs(samples_dir, exist_ok=True)
    logs_dir = os.path.join(config["training"]["output_dir"], "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")

    # Set random seeds for reproducibility
    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])

    # Determine compute dtype
    compute_dtype = (
        torch.float16
        if config["training"]["mixed_precision"] == "fp16"
        else (
            torch.bfloat16
            if config["training"]["mixed_precision"] == "bf16"
            else torch.float32
        )
    )

    # Print training info
    log_message(log_file, "Training Charlie from Purrfect Universe")
    log_message(log_file, f"Training steps: {config['training']['max_train_steps']}")
    log_message(log_file, f"Learning rate: {config['training']['learning_rate']}")
    log_message(log_file, f"LoRA rank: {config['training']['lora_rank']}")

    # Count images in dataset
    image_count = len(
        [f for f in os.listdir(config["training"]["data_dir"]) if f.endswith(".png")]
    )
    log_message(log_file, f"Found {image_count} images in dataset")

    # Display effective batch size
    effective_batch_size = (
        config["training"]["train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    )
    log_message(
        log_file,
        f"Effective batch size: {effective_batch_size} (batch_size={config['training']['train_batch_size']} × grad_accum={config['training']['gradient_accumulation_steps']})",
    )

    # Load the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="text_encoder",
    )

    # Freeze the text encoder
    text_encoder.requires_grad_(False)

    # Setup UNet model with LoRA
    unet = setup_pipeline(config["model"]["pretrained_model_name_or_path"])
    unet = prepare_unet_lora(
        unet, config["training"]["lora_rank"], config["training"]["lora_alpha"]
    )

    # Print trainable parameters
    log_message(
        log_file,
        f"Number of trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}",
    )

    # Setup the optimizer
    if config["training"]["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
            log_message(log_file, "Using 8-bit Adam optimizer")
        except ImportError:
            log_message(log_file, "bitsandbytes not found, using regular AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    # Configure optimizer with weight decay
    optimizer = optimizer_cls(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=0.01,
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["max_train_steps"],
        eta_min=config["training"]["learning_rate"] / 10,
    )

    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model"]["pretrained_model_name_or_path"], subfolder="scheduler"
    )

    # Setup dataset and dataloader
    train_dataset = CustomDataset(
        data_dir=config["training"]["data_dir"],
        tokenizer=tokenizer,
        size=config["model"]["resolution"],
    )

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encoder to GPU and set to compute dtype
    text_encoder = text_encoder.to(accelerator.device, dtype=compute_dtype)

    # Calculate training steps
    num_update_steps_per_epoch = (
        len(train_dataloader) // config["training"]["gradient_accumulation_steps"]
    )
    max_train_steps = config["training"]["max_train_steps"]

    # Training loop
    total_batch_size = (
        config["training"]["train_batch_size"]
        * accelerator.num_processes
        * config["training"]["gradient_accumulation_steps"]
    )

    # Setup progress bar
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Training Charlie")

    # Training variables
    global_step = 0
    best_loss = float("inf")
    best_visual_loss = float("inf")
    moving_avg_loss = 0
    last_sample_step = 0
    sample_frequency = 100

    # Training loop by epoch
    for epoch in range(0, (max_train_steps // num_update_steps_per_epoch) + 1):
        unet.train()
        epoch_loss = 0

        log_message(log_file, f"\nStarting epoch {epoch+1}")
        log_message(
            log_file, f"Current learning rate: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # Training loop by batch
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = encode_vae(
                        batch["pixel_values"],
                        config["model"]["pretrained_model_name_or_path"],
                        config["training"]["mixed_precision"],
                        accelerator.device,
                    )

                # Text encoder forward pass
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    encoder_hidden_states = encoder_hidden_states.to(
                        dtype=compute_dtype
                    )

                # Add noise to latents
                noise = torch.randn_like(latents).to(dtype=compute_dtype)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss calculation
                target = noise

                # UNet forward pass
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate loss
                model_pred = model_pred.to(dtype=compute_dtype)
                target = target.to(dtype=compute_dtype)
                loss = torch.nn.functional.mse_loss(
                    model_pred, target, reduction="mean"
                )

                epoch_loss += loss.detach().item()

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

                # Update learning rate
                lr_scheduler.step()

            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Update moving average loss
                if moving_avg_loss == 0:
                    moving_avg_loss = loss.detach().item()
                else:
                    moving_avg_loss = 0.9 * moving_avg_loss + 0.1 * loss.detach().item()

                # Print loss at regular intervals
                if global_step % 10 == 0:
                    log_message(
                        log_file,
                        f"Step {global_step}: Loss = {loss.detach().item():.6f}, Moving Avg = {moving_avg_loss:.6f}",
                    )

                # Generate sample images at regular intervals
                if (
                    global_step % sample_frequency == 0
                    and global_step > last_sample_step
                ):
                    from utils.generate import generate_samples

                    generate_samples(
                        accelerator,
                        unet,
                        config,
                        global_step,
                        samples_dir,
                        log_file,
                        best_visual_loss,
                    )

                # Save checkpoint every 500 steps
                if global_step % 500 == 0 and accelerator.is_main_process:
                    from utils.checkpoint import save_checkpoint

                    save_checkpoint(
                        accelerator,
                        unet,
                        global_step,
                        config["training"]["output_dir"],
                        log_file,
                    )

            if global_step >= max_train_steps:
                break

        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        log_message(
            log_file, f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}"
        )

        # Save best model based on loss
        if avg_epoch_loss < best_loss and accelerator.is_main_process:
            from utils.checkpoint import save_best_model

            best_loss = avg_epoch_loss
            save_best_model(
                accelerator, unet, best_loss, config["training"]["output_dir"], log_file
            )

    # Save the final model
    from utils.checkpoint import save_final_model

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_final_model(
            accelerator,
            unet,
            config["model"]["pretrained_model_name_or_path"],
            config["training"]["output_dir"],
            config["training"]["mixed_precision"],
            config,
        )

        log_message(log_file, f"Model saved to {config['training']['output_dir']}")
        log_message(
            log_file,
            f"LoRA adapter saved to {os.path.join(config['training']['output_dir'], 'lora')}",
        )
        log_message(
            log_file, "Training complete! You can now generate images using generate.py"
        )


if __name__ == "__main__":
    main()
