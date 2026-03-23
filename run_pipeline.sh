#!/usr/bin/env bash
set -e

# Activate virtualenv
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -d "$HOME/fairseq2_env" ]; then
    source "$HOME/fairseq2_env/bin/activate"
  else
    echo "ERROR: virtualenv fairseq2_env not found at $HOME/fairseq2_env"
    exit 1
  fi
fi

# Allow passing DEMO_MODE=1 to run quick demo instead of full training
DEMO_MODE="${DEMO_MODE:-0}"

echo "[1/7] Running preprocess.py"
python preprocess.py

echo "[2/7] Running train_tokenizer.py"
python train_tokenizer.py

echo "[3/7] Running apply_tokenizer.py"
python apply_tokenizer.py

echo "[4/7] Running prepare_data.py"
python prepare_data.py

echo "[5/7] Running model_config.py"
python model_config.py

if [ "$DEMO_MODE" = "1" ]; then
  echo "[6/7] Running demo_train.py (3 epochs, 4k samples)"
  python demo_train.py
else
  echo "[6/7] Running train.py (30 epochs, full dataset)"
  python train.py
fi

echo "[7/7] Running evaluate.py"
python evaluate.py

echo "✓ Pipeline completed successfully"
