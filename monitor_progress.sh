#!/usr/bin/env bash
# Monitoring script for NMT training pipeline

echo "=== NMT Pipeline Training Monitor ==="
echo ""

# Check if training is running
DEMO_LOG="/tmp/demo_train.log"
TRAIN_LOG="$HOME/eng_mni_nmt/results/training_log.csv"

if [ -f "$DEMO_LOG" ]; then
  echo "📊 Demo Training Status:"
  tail -5 "$DEMO_LOG" | grep -E "Starting|Step|Epoch|✓|Saved"
  echo ""
fi

if [ -f "$TRAIN_LOG" ]; then
  echo "📈 Training Metrics (Latest):"
  tail -3 "$TRAIN_LOG"
  echo ""
fi

# Check checkpoints
CKPT_DIR="$HOME/eng_mni_nmt/fairseq2_experiments/checkpoints"
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR)" ]; then
  echo "💾 Saved Checkpoints:"
  ls -lto "$CKPT_DIR" | head -5
  echo ""
fi

# Check evaluation results
RESULT_FILE="$HOME/eng_mni_nmt/results/scores/test_scores.json"
if [ -f "$RESULT_FILE" ]; then
  echo "🎯 Test Scores:"
  cat "$RESULT_FILE" | python3 -m json.tool 2>/dev/null || cat "$RESULT_FILE"
  echo ""
fi

echo "📁 Output Structure:"
echo "  Data: $HOME/eng_mni_nmt/data/"
echo "  Results: $HOME/eng_mni_nmt/results/"
echo "  Checks: $HOME/eng_mni_nmt/fairseq2_experiments/"
echo ""
echo "🔍 Watch training:"
echo "  tail -f /tmp/demo_train.log"
echo ""
echo "✅ When trained model is ready, run:"
echo "  source ~/fairseq2_env/bin/activate"
echo "  cd /workspaces/Fairseq2-Implementation"
echo "  python evaluate.py"
