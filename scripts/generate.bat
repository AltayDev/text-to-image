@echo off
echo Generate images with your fine-tuned model

echo Example prompts:
echo - "Charlie the cat from Purrfect Universe"
echo - "Charlie in a space suit"
echo - "Charlie playing with a ball of yarn"
echo - "Charlie sleeping on a window sill"
echo - "Charlie with big cute eyes"

set /p PROMPT="Enter your prompt: "
set /p OUTPUT="Output filename (default: generated.png): "

if "%OUTPUT%"=="" (
  set OUTPUT=generated.png
)

echo.
echo Checking for fine-tuned model...

python src/generate.py ^
  --config="config/default_config.json" ^
  --prompt="%PROMPT%" ^
  --output_file="%OUTPUT%"

echo.
echo Done! Image saved as %OUTPUT%
echo.
pause