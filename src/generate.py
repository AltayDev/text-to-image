import argparse
import torch
import os
import json
from diffusers import StableDiffusionPipeline
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with fine-tuned Stable Diffusion"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the fine-tuned model (overrides config)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="Path to the LoRA weights (overrides config)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to generate an image",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to avoid undesired features",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated.png",
        help="Output file path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    return parser.parse_args()


def generate_image(
    model_path,
    lora_path=None,
    base_model_path="sd-legacy/stable-diffusion-v1-5",
    prompt="Charlie the cat parachuting down from the sky, wearing a little purple Purrfect Universe parachute shaped like a PUR token. Grey and white fur, heterochromia eyes (blue and green), cartoon style, soft pastel clouds in the background. Dynamic pose, sense of motion, light breeze, whimsical and playful mood. --ar 16:9 --v 5 --q 2",
    negative_prompt="deformed, ugly, bad anatomy, poorly drawn, blurry",
    output_file="generated.png",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None,
    device="cuda",
):
    """Function that can be called from training loops or standalone script"""

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Check if fine-tuned model exists
    if model_path is None:
        model_path = base_model_path

    model_index_path = os.path.join(model_path, "model_index.json")
    use_fine_tuned = os.path.exists(model_index_path)

    # Load the model
    if use_fine_tuned:
        print(f"Loading fine-tuned model from {model_path}...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,  # Remove safety checker for faster generation
        )
    else:
        # Fall back to base model if fine-tuned model isn't available
        print(
            f"Fine-tuned model not found. Using base model {base_model_path} instead..."
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

    # Load LoRA weights if available
    if lora_path and os.path.exists(lora_path):
        try:
            print(f"Loading LoRA weights from {lora_path}...")
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
            print("LoRA weights loaded successfully!")
        except Exception as e:
            print(f"Failed to load LoRA weights: {str(e)}")
            print("Continuing with base model...")
    elif lora_path:
        print(f"LoRA path {lora_path} not found. Using model without LoRA weights.")

    # Move to GPU
    pipeline = pipeline.to(device)

    # Enable memory optimizations
    pipeline.enable_attention_slicing()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Could not enable xformers: {e}")

    # Generate an image
    print(f"Generating image with prompt: '{prompt}'")
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Save the image
    image.save(output_file)
    print(f"Image saved to {output_file}")

    return image


def main():
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Override config with command line arguments
    model_path = (
        args.model_path
        if args.model_path
        else config["model"]["pretrained_model_name_or_path"]
    )
    lora_path = (
        args.lora_path
        if args.lora_path
        else os.path.join(config["training"]["output_dir"], "lora")
    )
    negative_prompt = (
        args.negative_prompt
        if args.negative_prompt
        else config["generation"]["negative_prompt"]
    )
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps
        else config["generation"]["num_inference_steps"]
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale
        else config["generation"]["guidance_scale"]
    )

    # Call the generation function
    generate_image(
        model_path=model_path,
        lora_path=lora_path,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        output_file=args.output_file,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
