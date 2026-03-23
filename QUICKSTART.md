# NMT Pipeline: Quick Start Guide

## 🚀 Running the Pipeline

The complete NMT pipeline consists of 7 steps. Choose your mode based on time/resources:

### Mode 1: ⚡ Demo (Fastest - 30 minutes)
Quick demonstration on reduced dataset (4k samples, 3 epochs):
```bash
cd /workspaces/Fairseq2-Implementation
source ~/fairseq2_env/bin/activate
DEMO_MODE=1 ./run_pipeline.sh
```

**Produces**: 
- Trained checkpoint with ~2-3 BLEU score
- Full evaluation on test set
- Proof of concept in < 1 hour on CPU

### Mode 2: 📊 Full Training (Production - 24-48 hours)
Complete dataset with proper hyperparameters (16k samples, 30 epochs):
```bash
cd /workspaces/Fairseq2-Implementation
source ~/fairseq2_env/bin/activate
./run_pipeline.sh
# OR
# DEMO_MODE=0 ./run_pipeline.sh
```

**Produces**:
- Production-grade checkpoint
- Optimal BLEU/chrF++ metrics (5-8 BLEU estimated)
- Full 16k example training

---

## 📋 Pipeline Steps Explained

| Step | Script | Input | Output | Time |
|------|--------|-------|--------|------|
| 1 | `preprocess.py` | Raw parallel corpus | Cleaned 80/10/10 split | 1sec |
| 2 | `train_tokenizer.py` | Processed splits | SentencePiece BPE model | 10sec |
| 3 | `apply_tokenizer.py` | Raw text | Tokenized splits | 5sec |
| 4 | `prepare_data.py` | Tokenized files | fairseq2 manifest + dirs | 2sec |
| 5 | `model_config.py` | - | Load NLLB + config | 30sec (HF download) |
| 6a | `demo_train.py` | Training data | Best checkpoint + logs | **20 min** (CPU) |
| 6b | `train.py` | Training data | Best checkpoint + logs | **12-24 hrs** (CPU) / **2-4 hrs** (GPU) |
| 7 | `evaluate.py` | Test data + checkpoint | Translation + metrics | 5 min |

---

## 🗂️ Output Directories

After running, check results in `~/eng_mni_nmt/`:

```
eng_mni_nmt/
├── data/
│   ├── processed/          # Cleaned splits (80/10/10)
│   ├── tokenized/          # BPE-tokenized text
│   └── prepared/           # fairseq2-compatible format
├── tokenizer/
│   └── sentencepiece/      # SPM model & vocab
├── fairseq2_experiments/
│   └── checkpoints/        # Best model checkpoint
└── results/
    ├── training_log.csv    # Epoch metrics (loss, BLEU, chrF++)
    ├── translations/
    │   └── test_output.mni_bng   # Generated translations
    └── scores/
        └── test_scores.json      # BLEU/chrF++/TER metrics
```

---

## 🖥️ Running Individual Steps

You can also run steps independently:

```bash
source ~/fairseq2_env/bin/activate
cd /workspaces/Fairseq2-Implementation

# Step 1-4: Preprocess & tokenize
python preprocess.py
python train_tokenizer.py
python apply_tokenizer.py
python prepare_data.py

# Step 5: Load model
python model_config.py

# Step 6 (choose one):
python demo_train.py      # Quick: 3 epochs on 4k samples
python train.py           # Full: 30 epochs on 16k samples

# Step 7: Evaluate
python evaluate.py
```

---

## ⚙️ Customization

### Change Demo Parameters
Edit `demo_train.py` line ~16:
```python
@dataclass
class DemoConfig:
    max_epochs: int = 3          # Change to 5 or 10
    sample_size: int = 4000      # Change to 8000 for more data
    batch_size: int = 16         # Reduce to 8 if OOM
    learning_rate: float = 5e-5  # Decrease for stability
```

### Change Full Training Parameters
Edit `train.py` line ~25 or use config JSON:
```python
cfg.max_epochs = 20
cfg.batch_size = 8
cfg.learning_rate = 3e-5
```

### Use GPU
If CUDA available, it's auto-detected. Force CPU or GPU:
```python
# In train.py or demo_train.py, line ~60:
device = torch.device("cuda:0")  # Force GPU 0
device = torch.device("cpu")     # Force CPU
```

---

## 📊 Monitoring Training

During training, watch the logs:

```bash
# In another terminal
tail -f ~/eng_mni_nmt/results/training_log.csv

# Or view live
watch cat ~/eng_mni_nmt/results/training_log.csv
```

Expected progress:
- **Epoch 1-5**: Loss decreases from ~8 → ~3
- **Epoch 5-15**: Loss ~2-3, BLEU slowly increases
- **Epoch 15+**: BLEU plateaus, checkpoint updates rare

**Early stopping**: If BLEU hasn't improved for 5 epochs, you can safely stop.

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: transformers`
```bash
source ~/fairseq2_env/bin/activate
pip install transformers
```

### `Out of Memory (OOM)` during training
Reduce batch size in train.py:
```python
cfg.batch_size = 4  # Instead of 16
```

### Model download hangs
Models are cached in `~/.cache/huggingface/`. You can pre-download:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
```

### Segmentation fault / CUDA errors
Fallback to CPU-only training:
```python
device = torch.device("cpu")  # In train.py / demo_train.py
```

---

## 📈 Expected Results

| Mode | Epochs | Samples | Time (CPU) | Time (GPU) | Est. BLEU |
|------|--------|---------|-----------|-----------|-----------|
| Demo | 3 | 4k | 20 min | 2 min | 2-3 |
| Full | 30 | 16k | 12-24 hrs | 2-4 hrs | 5-8 |

*BLEU = Bilingual Evaluation Understudy score (0-100); higher is better*
*chrF++ = Character F-score (0-100); language-agnostic metric*

---

## 📚 Input Data Format

The parallel corpus at `/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng/` must have:

```
train.eng_Latn    # English sentences (one per line)
train.mni_Beng    # Meitei sentences (one per line)
```

Each file should have matching line counts. Example:
```
The quick brown fox jumps over the lazy dog.
কুইক ব্রাউন ফক্স অলস কুকুরের উপর জাম্প করে।
```

---

## 📝 Output Formats

### translations/test_output.mni_bng
Plain text file with one generated translation per line:
```
দ্রুত বাদামী শিয়াল অলস কুকুরের উপরে লাফায়।
...
```

### scores/test_scores.json
JSON with evaluation metrics:
```json
{
  "BLEU": 5.43,
  "chrF++": 45.23,
  "TER": 78.90
}
```

### training_log.csv
CSV with epoch-by-epoch metrics:
```
epoch,train_loss,dev_bleu,dev_chrf
1,4.2341,0.12,25.34
2,2.1543,1.45,31.21
3,1.8765,2.12,35.67
```

---

## 🎯 Next Steps After Training

1. **Fine-tune on more data**: Add more parallel sentences and re-run
2. **Adjust vocabulary**: Change `vocab_size=8000` in `train_tokenizer.py`
3. **Language-specific tuning**: Modify Bengali normalization in `preprocess.py`
4. **Ensemble models**: Train multiple seeds and ensemble outputs
5. **Deploy to production**: Export checkpoint to ONNX or TensorFlow format

---

**Questions?** Check individual script docstrings:
```bash
python -c "import train; help(train.run_evaluation)"
```
