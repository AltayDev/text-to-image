@echo off
echo Setting up training environment...
python -m pip install -r ..\requirements.txt

echo Warming up GPU for better performance...
python -c "import torch; torch.ones(1, device='cuda').to('cpu')"

echo Starting optimized training for Charlie character from Purrfect Universe...
python ..\src\train.py ^
  --config="..\config\default_config.json" ^
  --data_dir="..\dataset" ^
  --output_dir="..\output"

echo Training complete! Your model is saved in ../output
echo You can find sample images showing training progress in ../output/samples
echo To generate new images, run scripts/generate.bat
pause