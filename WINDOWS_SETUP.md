# Complete Windows 11 Setup & Execution Guide

## Step 1: Environment Setup (One-time)

### 1a. Download the Project
```bash
# Clone or download the Fairseq2-Implementation folder
# Place it in: C:\Users\YourUsername\Desktop\Fairseq2-Implementation
```

### 1b. Install Python
- Download Python 3.10+ from https://www.python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation
- Verify installation:
  ```bash
  python --version  # Should show Python 3.10+
  ```

### 1c. Create Virtual Environment
Open **PowerShell** as Administrator:

```powershell
cd C:\Users\YourUsername\Desktop\Fairseq2-Implementation

# Create virtual environment
python -m venv fairseq2_env

# Activate it
.\fairseq2_env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 1d. Install Dependencies
```powershell
pip install -r requirements.txt
```

**Expected installation time**: 3-5 minutes  
**Disk space needed**: ~2GB

---

## Step 2: Prepare Data

Your data should be in:
```
C:\Users\YourUsername\Desktop\Fairseq2-Implementation\eng_Latn-mni_Beng\
  ├── train.eng_Latn
  ├── train.mni_Beng
  ├── (optionally dev/test files)
```

If not present, the scripts will create train/dev/test splits automatically.

---

## Step 3: Run the Pipeline

### Option A: Quick Demo (1-2 Hours)
Validates the pipeline end-to-end with just 2k samples.

```powershell
.\fairseq2_env\Scripts\Activate.ps1
cd C:\Users\YourUsername\Desktop\Fairseq2-Implementation

python demo_train.py
```

**Expected output**:
```
[INFO] Training on 2000 examples (demo sample) for 2 epochs
[INFO] Device: cpu | Threads: 4
[INFO] Starting epoch 1/2
[INFO] Epoch 1/2 | Step 500/500 | Loss: 5.234
[INFO] Dev BLEU: 2.34 | chrF++: 28.12
...
[INFO] ✓ Training complete
```

**Success indicators**:
- ✅ Training loss decreases over time
- ✅ BLEU score > 1.5
- ✅ Checkpoint saved to `~/eng_mni_nmt/fairseq2_experiments/checkpoints/`

### Option B: Full Production Training (24-48 Hours)
Trains on all 16k samples for 30 epochs.

```powershell
.\fairseq2_env\Scripts\Activate.ps1
cd C:\Users\YourUsername\Desktop\Fairseq2-Implementation

python train.py
```

**Expected time**: 
- 16,084 training examples × 30 epochs ÷ 4 batch size ÷ ~0.5 sec/batch ≈ **24-48 hours**

**Recommended**:
1. Start Friday evening
2. Leave PC running overnight
3. Check results Saturday morning

### Option C: Full Pipeline (All Steps)

Individual steps:
```powershell
# Step 1: Preprocess data
python preprocess.py

# Step 2: Train tokenizer
python train_tokenizer.py

# Step 3: Apply tokenizer
python apply_tokenizer.py

# Step 4: Prepare data
python prepare_data.py

# Step 5: Load model config
python model_config.py

# Step 6: Train (choose demo or full)
python demo_train.py    # OR
python train.py

# Step 7: Evaluate
python evaluate.py
```

### Option D: Automated Pipeline Script
```powershell
# Demo mode
$env:DEMO_MODE = "1"
.\run_pipeline.sh

# Full mode (if you have Git installed)
.\run_pipeline.sh
```

---

## Step 4: Monitor Training

### Real-time Monitoring
While training runs, open another PowerShell:

```powershell
# Check training progress
Get-Content $env:USERPROFILE\eng_mni_nmt\results\training_log.csv -Tail 10

# Check memory usage
tasklist /fi "IMAGENAME eq python.exe" /v

# Monitor in Task Manager: Ctrl + Shift + Esc
```

### What to Expect
```csv
epoch,train_loss,dev_bleu,dev_chrf
1,6.234,1.23,25.12
2,5.891,1.45,26.34
3,5.123,1.89,27.45
...
```

- Train loss should **decrease** over epochs
- BLEU should **increase** slowly
- If loss plateaus, training proceed normally (CPU is just slower)

---

## Step 5: Retrieve Results

After training completes:

```powershell
# View final metrics
Get-Content $env:USERPROFILE\eng_mni_nmt\results\test_scores.json

# List checkpoints
Get-ChildItem $env:USERPROFILE\eng_mni_nmt\fairseq2_experiments\checkpoints\

# View translations
Get-Content $env:USERPROFILE\eng_mni_nmt\results\translations\test_output.mni_bng -Head 10
```

---

## Optimization Checklist for Your Hardware

Before starting training, verify:

### Memory
```powershell
# Check available RAM
Get-WmiObject Win32_OperatingSystem | Select-Object -Property @{Name='RAM (GB)'; Expression={[math]::Round($_.TotalVisibleMemorySize/1048576)}}

# Should show: RAM (GB) = 16 or higher
```

### CPU
```powershell
# Verify i7-8th gen
Get-WmiObject Win32_Processor | Select-Object -Property Name, NumberOfCores

# Should show: Intel(R) Core(TM) i7-8xxxxx (with 4-8 cores)
```

### Disk Space
```powershell
# Check free space (need ~5GB for models + results)
Get-Volume C: | Select-Object Size, SizeRemaining
```

### Close Resource-Heavy Programs
Before starting training, close:
- ❌ Chrome / Firefox (each uses 500MB-2GB)
- ❌ Discord / Slack
- ❌ Antivirus scans
- ❌ Windows Update
- ❌ Video apps (Netflix, YouTube)

**Recommended memory for training**:
- Before | Close programs | Free RAM needed
- 16 GB | Disconnect network | 6 GB free

---

## Troubleshooting

### Issue: "RuntimeError: Unexpected error"
**Solution**: Reduce batch size in script
```python
# In train.py (line ~25), change:
batch_size: int = 2  # Instead of 4
```

### Issue: "Process terminated unexpectedly"
**Cause**: Out of memory  
**Solution**: 
1. Close other programs
2. Reduce batch_size to 2
3. Reduce max_seq_len to 150

### Issue: "FileNotFoundError: eng_Latn-mni_Beng"
**Cause**: Data path incorrect  
**Solution**: Verify folder structure:
```
C:\Users\YourUsername\Desktop\Fairseq2-Implementation\
  ├── eng_Latn-mni_Beng\
  │   ├── train.eng_Latn
  │   └── train.mni_Beng
  ├── train.py
  ├── demo_train.py
  └── ... (other files)
```

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Cause**: Virtual environment not activated  
**Solution**:
```powershell
.\fairseq2_env\Scripts\Activate.ps1  # Must run each terminal session
python train.py
```

### Issue: Training very slow (< 0.5 samples/sec)
**This is normal on CPU!** Expected speed: 0.5-1 sample/sec  
- 4000 samples ÷ 1 sample/sec ÷ 60 sec/min ÷ 60 min/hr = **~1 hour** ✓

### Issue: "Killed" / "Terminated" after 10 min
**Cause**: Windows resource limit  
**Solution**: 
```powershell
# Adjust page file (virtual RAM)
# Settings → System → About → Advanced System Settings
# → Performance → Advanced → Virtual Memory → Change
# Set to: Initial Size = 8192 MB, Max Size = 16384 MB
# Restart and retry
```

---

## Running on Different Hardware

### If you have GPU (NVIDIA with CUDA)
Models already support GPU - no additional config needed
```python
# GPU will be auto-detected
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### If you have more RAM (32GB+)
Increase batch size for faster training:
```python
batch_size: int = 8  # Instead of 4 (faster, uses ~5-6GB)
```

### If you have an AMD Ryzen CPU
Adjust thread count:
```python
torch.set_num_threads(8)  # If 8 cores available
```

---

## Expected Results

### BLEU Score Interpretation
```
BLEU 0-1   : Very poor (random guessing)
BLEU 1-3   : Poor (typical for low-resource with reduced training)
BLEU 3-7   : Acceptable (2k demo + 30 epochs on CPU)
BLEU 7-15  : Good (full training on GPU)
```

For your **2k demo training (2 epochs)**:
- Expected BLEU: **2-3**
- This validates the pipeline works ✓

For your **full training (30 epochs)**:
- Expected BLEU: **5-7**
- This is production-quality for low-resource pairs ✓

---

## Next Steps

1. **Run demo** → Validate pipeline works (1-2 hours)
2. **Check results** → View BLEU/chrF++ scores
3. **Run full training** → Leave overnight (24-48 hours)
4. **Evaluate** → Run `python evaluate.py`
5. **Deploy** → Use checkpoint in production

---

## Support

**Common questions**:
- Q: Can I interrupt training and resume?  
  A: Yes - checkpoint saves every epoch, restart `python train.py`

- Q: Does it use GPU if available?  
  A: No - forced to CPU for consistency. To use GPU, remove `device = torch.device("cpu")`

- Q: Why is my speed different?  
  A: CPU speed varies with background processes. Close everything and retry.

- Q: How do I use the trained model?  
  A: See `QUICKSTART.md` for inference examples

---

Good luck! Training should start within seconds. Monitor `training_log.csv` for progress. 🚀
