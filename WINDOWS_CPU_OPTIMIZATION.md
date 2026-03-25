# Windows 11 CPU Optimization Guide (i7-8th Gen + 16GB RAM)

## Hardware Profile
- **CPU**: Intel i7-8th Generation (4-8 cores)
- **RAM**: 16 GB
- **GPU**: None
- **OS**: Windows 11

## Optimizations Applied

### 1. **Batch Size Reduction** (4 instead of 16)
- **Why**: Reduces memory footprint from ~6GB to ~2-3GB
- **Trade-off**: More frequent parameter updates, slower wall-time per epoch
- **Benefit**: Prevents OOM errors and system slowdown

### 2. **Sequence Length Reduction** (200 tokens instead of 256)
- **Why**: Memory = O(seq_length²) for attention
- **Impact**: ~27% reduction in memory per sample
- **Quality**: Minimal  impact on low-resource languages

### 3. **CPU Thread Optimization**
```python
torch.set_num_threads(4)  # Match i7-8th physical cores
```
- Prevents thread contention with Windows OS
- Improves CPU efficiency

### 4. **Disabled fp16 (Mixed Precision)**
- Not beneficial on CPU
- Adds unnecessary complexity
- Keep `fp16: False`

### 5. **Gradient Accumulation** (steps=2)
- Simulates batch_size=8 with memory of batch_size=4
- Effective gradient computation without extra memory

## Realistic Time Estimates

### Demo Training (2k samples, 2 epochs)
- **Hardware**: i7-8th gen + 16GB RAM
- **Batch size**: 4
- **Expected time**: **1-2 hours**
- **When to use**: Quick validation, testing the pipeline

### Full Training (16k samples, 30 epochs)
- **Hardware**: i7-8th gen + 16GB RAM
- **Batch size**: 4
- **Expected time**: **24-48 hours** (leave PC running overnight)
- **Memory usage**: ~3-4GB (safe for 16GB system)
- **When to use**: Final production model

## Windows-Specific Setup

### 1. Install Python & Dependencies
```bash
# Download Python 3.10+ from python.org
# During installation: ✓ "Add Python to PATH"

# Open PowerShell as Administrator
python -m venv fairseq2_env
.\fairseq2_env\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Data
```bash
# Copy eng_Latn-mni_Beng folder from your source
# Expected path: C:\Users\YourName\Fairseq2-Implementation\eng_Latn-mni_Beng
```

### 3. Run Pipeline

**Demo (Fast validation - 1-2 hours):**
```bash
.\fairseq2_env\Scripts\Activate.ps1
cd Fairseq2-Implementation
python demo_train.py
```

**Full (Production - 24-48 hours):**
```bash
python train.py
```

Or use the orchestration script:
```bash
# For demo: set environment variable before running
$env:DEMO_MODE = "1"
.\run_pipeline.sh

# For full pipeline
.\run_pipeline.sh
```

## Memory Management Tips

### 1. **Close Unnecessary Programs**
- Close Chrome, Discord, Slack, VS Code during training
- Frees ~2-4GB of RAM for training
- Every GB matters on CPU!

### 2. **Monitor Memory**
- Open Task Manager (Ctrl+Shift+Esc)
- Watch Python.exe memory usage
- Should not exceed 6GB for safety

### 3. **If OOM Occurs**
```python
# Further reduce batch_size in demo_train.py:
batch_size: int = 2  # Instead of 4
max_seq_len: int = 150  # Instead of 200
```

### 4. **Hibernation Workaround**
- Windows may sleep during long training
- Settings > Power > Screen off: Never
- Settings > Power > Put device to sleep: Never (or 60 min)

## Performance Profiling

Check actual performance:
```python
import torch
import time

# Inside training loop:
start = time.time()
# ... forward/backward ...
elapsed = time.time() - start
samples_per_sec = batch_size / elapsed
hours_total = (num_samples * num_epochs) / (samples_per_sec * 3600)
print(f"Throughput: {samples_per_sec:.1f} samples/sec → ~{hours_total:.1f}h total")
```

## Expected BLEU Scores

| Dataset Size | Epochs | Batch Size | Platform | BLEU | Time |
|---|---|---|---|---|---|
| 500 (demo) | 1 | 4 | i7-CPU | 1-2 | 15 min |
| 2k (demo) | 2 | 4 | i7-CPU | 2-3 | 1-2 hr |
| 16k (full) | 30 | 4 | i7-CPU | 5-7 | 24-48 hr |
| 16k (full) | 30 | 16 | V100-GPU | 8-12 | 2-4 hr |

## Troubleshooting

### "RuntimeError: CUDA out of memory"
- It shouldn't happen (no CUDA), but if importing CUDA libraries:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA detection
```

### "Process killed" / "Terminated unexpectedly"
- Windows resource limits reached
- **Solution**: Reduce batch_size, seq_length, or close other programs
- Enable page file if RAM < 4GB free

### "Training very slow" (< 0.5 samples/sec)
- Normal on i7 CPU
- Don't interrupt; let it run
- If < 0.1 samples/sec, something is wrong (check Task Manager)

### "FileNotFoundError" on data paths
- Windows uses backslashes: `C:\path\to\file`
- Python PathLib handles both auto
- If manual string: use raw string `r"C:\path"` or `C:\\path`

## Recommended Schedule

```
Monday evening: Start ./run_pipeline.sh (full training)
Tuesday morning: Check results, BLEU scores ready
→ Model ready for inference / evaluation
```

Or:

```
Anytime: python demo_train.py (quick validation)
→ Verify pipeline works, ~1 hour
Then: ./run_pipeline.sh (full training overnight/next day)
```

## Next Steps After Training

1. **Training complete** → Check `eng_mni_nmt/results/training_log.csv`
2. **Evaluate** → `python evaluate.py` (5 min)
3. **View metrics** → `eng_mni_nmt/results/test_scores.json`
4. **Expected BLEU**: 5-7 for low-resource language pair
5. **Deploy model** → Copy `eng_mni_nmt/fairseq2_experiments/checkpoints/best_epoch_*`

## References

- PyTorch CPU optimization: https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
- Mixed precision on CPU: Not recommended
- Windows Python venv: https://docs.python.org/3/library/venv.html

---

**Questions?** Check logs in `eng_mni_nmt/results/training_log.csv` and model outputs in `fairseq2_experiments/checkpoints/`.
