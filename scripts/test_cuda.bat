@echo off
echo === GPU Compatibility and CUDA Test ===
echo This will test your setup for AI training compatibility

python src/utils/cuda_test.py > cuda_test_results.txt

echo.
echo Test results have been saved to cuda_test_results.txt
echo.
echo If all tests passed, you can proceed with the quick training test:
echo   scripts/test_train.bat
echo.
echo Or start the full training:
echo   scripts/train.bat
echo.
pause