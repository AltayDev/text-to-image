@echo off
echo ===== TESTING MODE =====
echo This script will run a very short training (10 steps) to verify everything is working
echo.

echo Setting up training environment...
python -m pip install -r ..\requirements.txt

echo Warming up GPU for better performance...
python -c "import torch; torch.ones(1, device='cuda').to('cpu')"

echo Creating test configuration...
(
echo {
echo   "model": {
echo     "pretrained_model_name_or_path": "sd-legacy/stable-diffusion-v1-5",
echo     "resolution": 512
echo   },
echo   "training": {
echo     "data_dir": "../dataset",
echo     "output_dir": "../test_output",
echo     "train_batch_size": 1,
echo     "gradient_accumulation_steps": 1,
echo     "lora_rank": 4,
echo     "lora_alpha": 32,
echo     "mixed_precision": "fp16",
echo     "max_train_steps": 10,
echo     "learning_rate": 1e-4,
echo     "use_8bit_adam": true,
echo     "seed": 42
echo   },
echo   "generation": {
echo     "num_inference_steps": 30,
echo     "guidance_scale": 7.5,
echo     "negative_prompt": "deformed, ugly, bad anatomy, poorly drawn, blurry"
echo   },
echo   "sample_prompts": [
echo     "Charlie the cat from Purrfect Universe",
echo     "Charlie with big cute eyes"
echo   ]
echo }
) > "..\config\test_config.json"

echo Starting TEST training (10 steps only)...
python ..\src\train.py ^
  --config="..\config\test_config.json"

echo.
echo Test training complete!
echo.
echo Now let's generate a test image with our 10-step model...
python ..\src\generate.py ^
  --config="..\config\test_config.json" ^
  --prompt="Charlie the cat from Purrfect Universe" ^
  --output_file="test_generated.png"

echo.
echo If everything ran without errors, your setup is working correctly!
echo You can now run the full training with: scripts/train.bat
echo.
echo ===== TEST COMPLETE =====
pause