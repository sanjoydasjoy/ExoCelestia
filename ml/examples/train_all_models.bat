@echo off
REM Train all three model types and compare results (Windows version)

set DATA_PATH=%1
if "%DATA_PATH%"=="" set DATA_PATH=../data/exoplanets.csv
set OUTPUT_DIR=../models

echo ==================================
echo Training All Models
echo ==================================
echo Data: %DATA_PATH%
echo Output: %OUTPUT_DIR%
echo.

REM Train Random Forest
echo 1. Training Random Forest...
python ../src/train.py "%DATA_PATH%" --model random_forest --output "%OUTPUT_DIR%/rf_model.pkl" --test-size 0.2 --random-state 42
echo.

REM Train XGBoost
echo 2. Training XGBoost...
python ../src/train.py "%DATA_PATH%" --model xgboost --output "%OUTPUT_DIR%/xgb_model.pkl" --test-size 0.2 --random-state 42
echo.

REM Train Neural Network (if PyTorch available)
echo 3. Training Neural Network...
python ../src/train.py "%DATA_PATH%" --model nn --output "%OUTPUT_DIR%/nn_model.pt" --test-size 0.2 --random-state 42 --epochs 50 --batch-size 64
echo.

echo ==================================
echo Training Complete!
echo ==================================
echo Models saved to %OUTPUT_DIR%/
echo.
echo Compare metrics:
echo   type %OUTPUT_DIR%\metrics.json
pause

