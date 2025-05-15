import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from utils.logger import log_message


def save_checkpoint(accelerator, unet, global_step, output_dir, log_file):
    """Save a checkpoint during training"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(os.path.join(checkpoint_dir, "lora"))
    log_message(log_file, f"Saved checkpoint at step {global_step} to {checkpoint_dir}")


def save_best_model(accelerator, unet, best_loss, output_dir, log_file):
    """Save the best model based on loss"""
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(os.path.join(best_model_dir, "lora"))
    log_message(
        log_file, f"Saved best model with loss {best_loss:.6f} to {best_model_dir}"
    )


def save_final_model(
    accelerator, unet, pretrained_model_path, output_dir, mixed_precision, config
):
    """Save the final fine-tuned model"""
    # Determine compute dtype
    compute_dtype = (
        torch.float16
        if mixed_precision == "fp16"
        else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
    )

    # Unwrap the unet
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the LoRA weights separately
    unwrapped_unet.save_pretrained(os.path.join(output_dir, "lora"))

    # Load original unet
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
    )

    # Save the full pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        unet=original_unet,  # Use original weights for saving
        torch_dtype=compute_dtype,
    )
    pipeline.save_pretrained(output_dir)

    # Create a README with prompt examples and samples
    create_model_readme(output_dir, config)

    # Copy best visual model to final output if available
    copy_best_visual_model(output_dir)


def create_model_readme(output_dir, config):
    """Create a README file for the model"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Charlie from Purrfect Universe Model\n\n")
        f.write(
            "This model is trained to generate images of Charlie from Purrfect Universe.\n\n"
        )
        f.write("## Example Prompts\n\n")
        for prompt in config["sample_prompts"]:
            f.write(f'- "{prompt}"\n')
        f.write("\n## Training Details\n\n")
        f.write(f"- Training steps: {config['training']['max_train_steps']}\n")
        f.write(f"- Learning rate: {config['training']['learning_rate']}\n")
        f.write(f"- LoRA rank: {config['training']['lora_rank']}\n")

        # Try to count dataset size
        try:
            image_count = len(
                [
                    f
                    for f in os.listdir(config["training"]["data_dir"])
                    if f.endswith(".png")
                ]
            )
            f.write(f"- Dataset size: {image_count} images\n")
        except:
            pass

        f.write(f"- Batch size: {config['training']['train_batch_size']}\n")
        f.write(
            f"- Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}\n\n"
        )
        f.write("## Training Progress\n\n")
        f.write("Check the samples directory for training progress visualizations.\n")


def copy_best_visual_model(output_dir):
    """Copy the best visual model to final_tuned_lora directory if it exists"""
    best_visual_model_path = os.path.join(output_dir, "best_visual_model", "lora")
    if os.path.exists(best_visual_model_path):
        import shutil

        final_model_path = os.path.join(output_dir, "final_tuned_lora")
        shutil.copytree(best_visual_model_path, final_model_path, dirs_exist_ok=True)
        print(f"Copied best visual model to {final_model_path} for easier access")
