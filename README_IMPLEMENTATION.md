# 🎯 NMT Implementation Complete — Ready to Train

## ✅ What's Implemented

### Core Pipeline (7 Steps)
- **Step 1**: Data preprocessing (clean, normalize, split 80/10/10)
- **Step 2**: SentencePiece tokenizer training (8k BPE vocab)
- **Step 3**: Apply tokenizer to all splits
- **Step 4**: Prepare data for training (fairseq2 manifest format)
- **Step 5**: Load NLLB-200-M model from HF Hub
- **Step 6**: Training loop (choose demo or full)
  - `demo_train.py` — ⚡ 3 epochs + 4k samples (~20 min on CPU)
  - `train.py` — 📊 30 epochs + 16k samples (~12-24 hrs on CPU)
- **Step 7**: Evaluation (BLEU/chrF++/TER metrics)

### Key Features
✓ Full error handling & logging in all scripts
✓ Path-agnostic (uses `pathlib.Path`)
✓ GPU auto-detection + CPU fallback
✓ Checkpoint saving (best + resumable)
✓ Metric logging per epoch
✓ Detokenization & evaluation
✓ Transformers backend (no fairseq2 needed!)

---

## 🚀 Quick Start

### Run Demo (Fast!)
```bash
source ~/fairseq2_env/bin/activate
cd /workspaces/Fairseq2-Implementation
DEMO_MODE=1 ./run_pipeline.sh
```
**Time**: ~45 minutes (CPU)  
**Output**: Trained model, translations, metrics

### Run Full Training (Production)
```bash
source ~/fairseq2_env/bin/activate
cd /workspaces/Fairseq2-Implementation
./run_pipeline.sh   # or: DEMO_MODE=0 ./run_pipeline.sh
```
**Time**: 12-24 hours (CPU) / 2-4 hours (GPU)  
**Output**: Optimized model, high-quality metrics

---

## 📁 Files Created

```
/workspaces/Fairseq2-Implementation/
├── requirements.txt           # Dependencies
├── preprocess.py             # Step 1: Data cleaning & split
├── train_tokenizer.py        # Step 2: SPM training
├── apply_tokenizer.py        # Step 3: Tokenization
├── prepare_data.py           # Step 4: Dataset prep
├── model_config.py           # Step 5: Model loading
├── train.py                  # Step 6: Full training (30 epochs)
├── demo_train.py             # Step 6: Demo training (3 epochs)
├── evaluate.py               # Step 7: Evaluation + metrics
├── run_pipeline.sh           # Main orchestration script
├── QUICKSTART.md            # Detailed usage guide
├── SETUP.md                 # Environment setup
└── monitor_progress.sh       # Progress monitoring helper
```

---

## 📊 Expected Outputs

After running, find results at:

```
~/eng_mni_nmt/
├── data/processed/          # Cleaned corpus splits
├── data/tokenized/          # BPE-tokenized text
├── data/prepared/           # Training data manifest
├── tokenizer/sentencepiece/ # SPM model (spm.model)
├── fairseq2_experiments/
│   └── checkpoints/         # **Best checkpoint here**
└── results/
    ├── training_log.csv     # Epoch metrics
    ├── translations/
    │   └── test_output.mni_bng  # Translations
    └── scores/
        └── test_scores.json     # BLEU/chrF++/TER
```

---

## 🎓 What Each Script Does

### preprocess.py
- Reads raw parallel corpus from `/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng/`
- Removes empty lines, normalizes Unicode (Bengali ZWJ/ZWNJ handling)
- Shuffles and splits: 80% train, 10% dev, 10% test
- **Output**: 3 train/dev/test pairs in `~/eng_mni_nmt/data/processed/`

### train_tokenizer.py
- Combines all splits into one corpus
- Trains joint SentencePiece BPE model
- Settings: vocab_size=8000, character_coverage=0.9995
- **Output**: `spm.model` and `spm.vocab` in `~/eng_mni_nmt/tokenizer/sentencepiece/`

### apply_tokenizer.py
- Applies trained SPM to each split
- Converts text to subword pieces
- **Output**: Tokenized files in `~/eng_mni_nmt/data/tokenized/`

### prepare_data.py
- Creates fairseq2 manifest (JSONL format)
- Organizes data into split directories
- **Output**: Dataset structure ready for training

### model_config.py
- Downloads NLLB-200-distilled-600M from Hugging Face
- Saves config to JSON for reproducibility
- **Output**: Model loaded, config saved

### train.py (Full)
- **30 epochs**, all 16k training examples
- Adam optimizer with linear warmup
- Gradient clipping, label smoothing
- Checkpoint best by dev BLEU
- **Output**: Best model checkpoint, training_log.csv

### demo_train.py (Quick)
- **3 epochs**, 4k sampled examples
- Same optimizer/loss as full training
- Fast evaluation (200 samples per dev check)
- **Output**: Same structure as train.py (much faster!)

### evaluate.py
- Loads best checkpoint
- Beam search decoding (beam_size=5)
- Computes BLEU, chrF++, TER metrics
- **Output**: Translations + scores JSON

---

## 🛠️ Advanced Usage

### Resume Training from Checkpoint
```bash
# Modify train.py to load checkpoint:
best_ckpt_dir = Path(...) / "checkpoints" / "best_epoch_2_bleu_2.34"
model = AutoModelForSeq2SeqLM.from_pretrained(best_ckpt_dir)
# Then continue training...
```

### Reduce Memory Usage
```bash
# In demo_train.py: line ~20
batch_size: int = 4       # Reduce from 16
# OR in train.py, reduce max_seq_len
max_seq_len: int = 128    # From 256
```

### Use Different Base Model
```bash
# In model_config.py, line ~30
"facebook/nllb-200-1.3B"    # Larger model
"facebook/nllb-100-distilled-600M"  # For 100 languages
```

### Train on GPU
```bash
# Auto-detected, but force with:
device = torch.device("cuda:0")  # In train.py / demo_train.py
```

---

## 📋 Checklist Before Running

- [ ] Virtual env activated: `source ~/fairseq2_env/bin/activate`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Input data exists: `/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng/train.*`
- [ ] Output dir writable: `mkdir -p ~/eng_mni_nmt`
- [ ] Disk space: ~2GB for checkpoints + cache

---

## 🔗 Key Resources

- **fairseq2 GitHub**: https://github.com/facebookresearch/fairseq2
- **NLLB Paper**: https://arxiv.org/abs/2207.04672
- **SentencePiece**: https://github.com/google/sentencepiece
- **HF Transformers**: https://huggingface.co/docs/transformers

---

## 💡 Next Steps

1. **Run demo**: `DEMO_MODE=1 ./run_pipeline.sh` (~45 min)
2. **Monitor**: `tail -f /tmp/demo_train.log`
3. **Evaluate**: After training, `python evaluate.py`
4. **Analyze**: Check `~/eng_mni_nmt/results/training_log.csv`
5. **Deploy**: Export checkpoint to production

---

**Ready to train!** 🚀
