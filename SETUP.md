# NMT Setup Guide for Fairseq2 Implementation

## ✅ Prerequisites Installed
- ✓ Virtual environment: `~/fairseq2_env`
- ✓ Core dependencies: torch, sentencepiece, sacrebleu, pandas, numpy, tqdm, unicodedata2, transformers

## 📦 Remaining Setup Steps

### 1. Install fairseq2
Since fairseq2 is not on PyPI, install from GitHub:

```bash
source ~/fairseq2_env/bin/activate
pip install git+https://github.com/facebookresearch/fairseq2.git
```

This may take 5-10 minutes due to compilation.

### 2. Create project directory structure
```bash
mkdir -p ~/eng_mni_nmt/{data/processed,data/tokenized,data/prepared,tokenizer/sentencepiece,fairseq2_experiments/checkpoints,results/{translations,scores}}
```

### 3. Prepare data files
Copy or link your parallel corpus to the workspace:
- Source: `/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng/train.eng_Latn`
- Source: `/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng/train.mni_Beng`

### 4. Run the pipeline
From `/workspaces/Fairseq2-Implementation/`:

```bash
source ~/fairseq2_env/bin/activate
./run_pipeline.sh
```

## 🔄 Pipeline Steps

1. **preprocess.py** - Cleans and splits data (80/10/10)
2. **train_tokenizer.py** - Trains SentencePiece BPE model
3. **apply_tokenizer.py** - Tokenizes train/dev/test splits
4. **prepare_data.py** - Prepares fairseq2 dataset format
5. **model_config.py** - Loads NLLB pretrained model
6. **train.py** - Fine-tunes on English→Meitei task
7. **evaluate.py** - Evaluates on test set with BLEU/chrF++/TER

## 📊 Output Structure

After completion, find results at:
- Checkpoints: `~/eng_mni_nmt/fairseq2_experiments/checkpoints/`
- Translations: `~/eng_mni_nmt/results/translations/test_output.mni_bng`
- Scores: `~/eng_mni_nmt/results/scores/test_scores.json`
- Training log: `~/eng_mni_nmt/results/training_log.csv`

## ⚙️ Configuration

Edit `train.py` to adjust:
- `batch_size` (default: 16)
- `max_epochs` (default: 30)
- `learning_rate` (default: 5e-5)
- `dropout` (default: 0.3 for low-resource)
- `beam_size` for inference (default: 5)

## 🐛 Troubleshooting

**ImportError: fairseq2 module**
```bash
# Verify installation
pip list | grep fairseq2

# Reinstall from latest main
pip install --force-reinstall git+https://github.com/facebookresearch/fairseq2.git@main
```

**CUDA out of memory during training**
- Reduce `batch_size` in train.py (e.g., 8 or 4)
- Set `fp16=False` to disable mixed precision

**Data not found errors**
- Verify files exist: `ls -la ~/eng_mni_nmt/data/processed/`
- Check paths in logs match your setup

## 📝 Notes

- All scripts use `pathlib.Path` for cross-platform compatibility
- Full logging output to console and files
- Automatic checkpoint saving based on dev BLEU score
- Bengali script normalization applied to Meitei text
