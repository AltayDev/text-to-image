import os
import torch
from utils.logger import log_message


def generate_samples(
    accelerator, unet, config, global_step, samples_dir, log_file, best_visual_loss
):
    """
    Generate sample images during training to visualize progress
    """
    # Try to import the generate_image function from generate.py
    try:
        import sys
        import importlib.util

        generate_image_spec = importlib.util.spec_from_file_location(
            "generate_image", "./src/generate.py"
        )
        generate_image_module = importlib.util.module_from_spec(generate_image_spec)
        sys.modules["generate_image"] = generate_image_module
        generate_image_spec.loader.exec_module(generate_image_module)
        generate_image_fn = getattr(generate_image_module, "generate_image", None)

        if not generate_image_fn:
            log_message(
                log_file,
                "Could not import generate_image function, skipping sample generation",
            )
            return best_visual_loss
    except Exception as e:
        log_message(log_file, f"Error importing generate_image function: {str(e)}")
        return best_visual_loss

    # Only proceed with sample generation if accelerator is main process
    if not accelerator.is_main_process:
        return best_visual_loss

    # Unwrap the unet for inference
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Save temporary checkpoint for inference
    temp_lora_dir = os.path.join(config["training"]["output_dir"], "temp_lora")
    os.makedirs(temp_lora_dir, exist_ok=True)
    unwrapped_unet.save_pretrained(temp_lora_dir)

    # Generate sample images with different prompts
    log_message(log_file, f"Generating sample images at step {global_step}...")

    # Create step samples directory
    step_samples_dir = os.path.join(samples_dir, f"step_{global_step:05d}")
    os.makedirs(step_samples_dir, exist_ok=True)

    visual_loss_sum = 0.0  # To track visual quality

    for i, prompt in enumerate(config["sample_prompts"]):
        try:
            sample_path = os.path.join(step_samples_dir, f"sample_{i+1}.png")
            generate_image_fn(
                model_path=config["model"]["pretrained_model_name_or_path"],
                lora_path=temp_lora_dir,
                prompt=prompt,
                output_file=sample_path,
                num_inference_steps=30,  # Faster inference for samples
                guidance_scale=7.5,
                seed=(
                    config["training"]["seed"] + i
                    if config["training"]["seed"] is not None
                    else None
                ),
            )
            log_message(log_file, f"  - Created sample {i+1} with prompt: '{prompt}'")

            # Simple estimate of visual quality - could be improved with CLIP score
            visual_loss_sum += loss.detach().item() if "loss" in locals() else 0
        except Exception as e:
            log_message(log_file, f"  - Failed to generate sample {i+1}: {str(e)}")

    # Save best model based on visual quality if it's better than previous best
    new_best_visual_loss = best_visual_loss
    if visual_loss_sum < best_visual_loss:
        new_best_visual_loss = visual_loss_sum
        best_visual_model_dir = os.path.join(
            config["training"]["output_dir"], "best_visual_model"
        )
        os.makedirs(best_visual_model_dir, exist_ok=True)
        unwrapped_unet.save_pretrained(os.path.join(best_visual_model_dir, "lora"))
        log_message(
            log_file,
            f"Saved best visual model at step {global_step} to {best_visual_model_dir}",
        )

    return new_best_visual_loss
