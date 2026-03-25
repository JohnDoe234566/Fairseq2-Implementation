import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class FineTuneConfig:
    """Production training config - optimized for Windows CPU (i7-8th gen + 16GB)."""
    max_seq_len: int = 200  # Reduced from 256 for CPU memory efficiency
    batch_size: int = 4  # Reduced from 16 for i7-8th + 16GB RAM
    max_epochs: int = 30  # Full production training
    learning_rate: float = 3e-5  # Slightly lower for CPU stability
    warmup_steps: int = 100
    label_smoothing: float = 0.1
    dropout: float = 0.3
    optimizer: str = "adamw"
    fp16: bool = False  # Disabled on CPU (not beneficial)
    beam_size: int = 4  # Reduce for CPU speed (was 5)
    gradient_accumulation_steps: int = 2  # Simulate batch_size=8 with mem of 4


def run_evaluation(model, tokenizer, src_path, tgt_path, device, cfg):
    """Evaluate model on dev set using BLEU and chrF++."""
    references = [l.strip() for l in tgt_path.open("r", encoding="utf-8") if l.strip()]
    sources = [l.strip() for l in src_path.open("r", encoding="utf-8") if l.strip()]
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for src_text in sources:
            try:
                # Tokenize and generate
                input_ids = tokenizer(src_text, return_tensors="pt", max_length=cfg.max_seq_len, truncation=True).input_ids
                input_ids = input_ids.to(device)
                outputs = model.generate(input_ids, max_length=cfg.max_seq_len, num_beams=cfg.beam_size)
                hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
                hypotheses.append(hypothesis)
            except Exception as e:
                logging.warning("Generation failed for source: %s", e)
                hypotheses.append("")

    if hypotheses and all(hypotheses):
        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    else:
        bleu, chrf = 0.0, 0.0
    return bleu, chrf


def main():
    setup_logging()
    try:
        # Windows CPU optimization
        torch.set_num_threads(4)  # i7-8th gen has 4 physical cores
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Explicitly disable CUDA
        
        dataset_root = Path.home() / "eng_mni_nmt"
        cfg = FineTuneConfig()

        # Load config from model_config.py if available
        config_json = dataset_root / "model_config.json"
        if config_json.exists():
            with config_json.open("r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
                for k, v in cfg_dict.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)

        train_src = dataset_root / "data" / "prepared" / "train" / "eng_Latn.txt"
        train_tgt = dataset_root / "data" / "prepared" / "train" / "mni_Beng.txt"
        dev_src = dataset_root / "data" / "prepared" / "dev" / "eng_Latn.txt"
        dev_tgt = dataset_root / "data" / "prepared" / "dev" / "mni_Beng.txt"

        if not all(x.exists() for x in [train_src, train_tgt, dev_src, dev_tgt]):
            raise FileNotFoundError("Prepared data files are missing; run prepare_data.py first")

        # Load model from Hugging Face
        logging.info("Loading NLLB model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        device = torch.device("cpu")  # Force CPU (no CUDA on Windows)
        model = model.to(device)
        logging.info("Model loaded on device: CPU (i7-8th gen Windows)")
        logging.info("Batch size: %d | Max seq len: %d | Expected time: 24-48 hours", cfg.batch_size, cfg.max_seq_len)

        # Optimizer (Adam instead of AdamW for CPU speed)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        total_steps = cfg.max_epochs * 1000  # approximate
        
        best_bleu = -1.0
        checkpoint_dir = dataset_root / "fairseq2_experiments" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_path = dataset_root / "results" / "training_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load training data
        src_lines = [l.strip() for l in train_src.open("r", encoding="utf-8") if l.strip()]
        tgt_lines = [l.strip() for l in train_tgt.open("r", encoding="utf-8") if l.strip()]
        logging.info("Loaded %d training pairs", len(src_lines))

        with open(log_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "dev_bleu", "dev_chrf"])

            global_step = 0

            for epoch in range(1, cfg.max_epochs + 1):
                logging.info("Starting epoch %d/%d", epoch, cfg.max_epochs)
                model.train()

                epoch_loss = 0.0
                step = 0

                for i in range(0, len(src_lines), cfg.batch_size):
                    batch_src = src_lines[i : i + cfg.batch_size]
                    batch_tgt = tgt_lines[i : i + cfg.batch_size]

                    # Tokenize batch
                    inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len)
                    labels = tokenizer(batch_tgt, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len)

                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    label_ids = labels["input_ids"].to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    global_step += 1
                    step += 1
                    epoch_loss += loss.item()

                    if step % 100 == 0:
                        logging.info("  Step %d/%d loss=%.4f", step, (len(src_lines) + cfg.batch_size - 1) // cfg.batch_size, loss.item())

                avg_train_loss = epoch_loss / max(1, step)

                # Evaluation
                dev_bleu, dev_chrf = run_evaluation(model, tokenizer, dev_src, dev_tgt, device, cfg)

                logging.info("Epoch %d train_loss=%.4f dev_bleu=%.2f dev_chrf=%.2f", epoch, avg_train_loss, dev_bleu, dev_chrf)

                writer.writerow([epoch, avg_train_loss, dev_bleu, dev_chrf])
                csvfile.flush()

                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    ckpt_path = checkpoint_dir / f"best_epoch_{epoch}_bleu_{best_bleu:.2f}.pt"
                    model.save_pretrained(checkpoint_dir / f"best_epoch_{epoch}")
                    logging.info("Saved best checkpoint at epoch %d with BLEU=%.2f", epoch, best_bleu)

        logging.info("Training complete. Best BLEU=%.2f", best_bleu)

    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
