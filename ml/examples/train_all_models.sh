#!/bin/bash
# Train all three model types and compare results

DATA_PATH="${1:-../data/exoplanets.csv}"
OUTPUT_DIR="../models"

echo "=================================="
echo "Training All Models"
echo "=================================="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Train Random Forest
echo "1. Training Random Forest..."
python ../src/train.py "$DATA_PATH" \
    --model random_forest \
    --output "$OUTPUT_DIR/rf_model.pkl" \
    --test-size 0.2 \
    --random-state 42

echo ""

# Train XGBoost
echo "2. Training XGBoost..."
python ../src/train.py "$DATA_PATH" \
    --model xgboost \
    --output "$OUTPUT_DIR/xgb_model.pkl" \
    --test-size 0.2 \
    --random-state 42

echo ""

# Train Neural Network (if PyTorch available)
echo "3. Training Neural Network..."
python ../src/train.py "$DATA_PATH" \
    --model nn \
    --output "$OUTPUT_DIR/nn_model.pt" \
    --test-size 0.2 \
    --random-state 42 \
    --epochs 50 \
    --batch-size 64

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo "Models saved to $OUTPUT_DIR/"
echo ""
echo "Compare metrics:"
echo "  cat $OUTPUT_DIR/metrics.json"

