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
class DemoConfig:
    """Demo training config: 3 epochs, 4k samples for quick results."""
    max_seq_len: int = 256
    batch_size: int = 16
    max_epochs: int = 3  # Quick demo: 3 epochs instead of 30
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    label_smoothing: float = 0.1
    dropout: float = 0.3
    optimizer: str = "adamw"
    fp16: bool = True
    beam_size: int = 5
    sample_size: int = 4000  # Use only 4k examples for demo (instead of 16k)


def run_evaluation(model, tokenizer, src_path, tgt_path, device, cfg, max_samples=500):
    """Evaluate model on dev set using BLEU and chrF++."""
    references = [l.strip() for l in tgt_path.open("r", encoding="utf-8") if l.strip()]
    sources = [l.strip() for l in src_path.open("r", encoding="utf-8") if l.strip()]
    
    # Use only max_samples for demo speed
    if len(sources) > max_samples:
        step = len(sources) // max_samples
        sources = sources[::step][:max_samples]
        references = references[::step][:max_samples]
    
    hypotheses = []
    model.eval()
    with torch.no_grad():
        for src_text in sources:
            try:
                input_ids = tokenizer(src_text, return_tensors="pt", max_length=cfg.max_seq_len, truncation=True).input_ids
                input_ids = input_ids.to(device)
                outputs = model.generate(input_ids, max_length=cfg.max_seq_len, num_beams=cfg.beam_size)
                hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
                hypotheses.append(hypothesis)
            except Exception as e:
                logging.warning("Generation failed: %s", e)
                hypotheses.append("")

    if hypotheses and all(hypotheses) and all(references):
        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    else:
        bleu, chrf = 0.0, 0.0
    return bleu, chrf


def main():
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"
        cfg = DemoConfig()

        train_src = dataset_root / "data" / "prepared" / "train" / "eng_Latn.txt"
        train_tgt = dataset_root / "data" / "prepared" / "train" / "mni_Beng.txt"
        dev_src = dataset_root / "data" / "prepared" / "dev" / "eng_Latn.txt"
        dev_tgt = dataset_root / "data" / "prepared" / "dev" / "mni_Beng.txt"

        if not all(x.exists() for x in [train_src, train_tgt, dev_src, dev_tgt]):
            raise FileNotFoundError("Prepared data files are missing")

        # Load model from Hugging Face
        logging.info("Loading NLLB model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info("Model loaded on device: %s", device)

        # Load training data
        src_lines = [l.strip() for l in train_src.open("r", encoding="utf-8") if l.strip()]
        tgt_lines = [l.strip() for l in train_tgt.open("r", encoding="utf-8") if l.strip()]
        
        # Sample dataset for demo
        if len(src_lines) > cfg.sample_size:
            step = len(src_lines) // cfg.sample_size
            src_lines = src_lines[::step][:cfg.sample_size]
            tgt_lines = tgt_lines[::step][:cfg.sample_size]
        
        logging.info("Training on %d examples (demo sample) for %d epochs", len(src_lines), cfg.max_epochs)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        
        best_bleu = -1.0
        checkpoint_dir = dataset_root / "fairseq2_experiments" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_path = dataset_root / "results" / "training_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "dev_bleu", "dev_chrf"])

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

                    step += 1
                    epoch_loss += loss.item()

                    if step % 50 == 0:
                        logging.info("  Step %d loss=%.4f", step, loss.item())

                avg_train_loss = epoch_loss / max(1, step)

                # Evaluation
                dev_bleu, dev_chrf = run_evaluation(model, tokenizer, dev_src, dev_tgt, device, cfg, max_samples=200)

                logging.info("Epoch %d | train_loss=%.4f | dev_bleu=%.2f | dev_chrf=%.2f", 
                            epoch, avg_train_loss, dev_bleu, dev_chrf)

                writer.writerow([epoch, avg_train_loss, dev_bleu, dev_chrf])
                csvfile.flush()

                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    ckpt_name = f"best_epoch_{epoch}_bleu_{best_bleu:.2f}"
                    model.save_pretrained(checkpoint_dir / ckpt_name)
                    logging.info("✓ Saved best checkpoint: %s", ckpt_name)

        logging.info("\n✓ Training complete. Best BLEU=%.2f", best_bleu)
        logging.info("Results saved to:")
        logging.info("  - Log: %s", log_path)
        logging.info("  - Checkpoint: %s", checkpoint_dir)

    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
