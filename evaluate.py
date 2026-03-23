import json
import logging
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


def load_best_checkpoint(checkpoint_dir: Path):
    """Load the most recent checkpoint directory."""
    ckpts = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("best_epoch_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not ckpts:
        logging.warning("No fine-tuned checkpoint found; using base model")
        return None
    return ckpts[0]


def decode_batch(model, tokenizer, src_lines, device, beam_size=5):
    """Generate translations for source lines using beam search."""
    outcomes = []
    model.eval()
    with torch.no_grad():
        for sent in src_lines:
            try:
                input_ids = tokenizer(sent, return_tensors="pt", max_length=256, truncation=True).input_ids
                input_ids = input_ids.to(device)
                outputs = model.generate(input_ids, max_length=256, num_beams=beam_size)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                outcomes.append(decoded)
            except Exception as e:
                logging.warning("Decoding failed for source: %s", e)
                outcomes.append("")
    return outcomes


if __name__ == "__main__":
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"

        # Load model from Hugging Face
        logging.info("Loading NLLB model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        # Try to load fine-tuned checkpoint
        checkpoint_dir = dataset_root / "fairseq2_experiments" / "checkpoints"
        best_ckpt = load_best_checkpoint(checkpoint_dir)
        
        if best_ckpt is not None:
            logging.info("Loading fine-tuned model from %s", best_ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(best_ckpt)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info("Model loaded on device: %s", device)

        test_src = dataset_root / "data" / "prepared" / "test" / "eng_Latn.txt"
        test_tgt = dataset_root / "data" / "prepared" / "test" / "mni_Beng.txt"
        if not test_src.exists() or not test_tgt.exists():
            raise FileNotFoundError("Test set not found")

        source_lines = [l.strip() for l in test_src.open("r", encoding="utf-8") if l.strip()]
        reference_lines = [l.strip() for l in test_tgt.open("r", encoding="utf-8") if l.strip()]

        logging.info("Generating translations for %d test examples...", len(source_lines))
        translations = decode_batch(model, tokenizer, source_lines, device, beam_size=5)

        # Save translations
        out_path = dataset_root / "results" / "translations" / "test_output.mni_bng"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(translations) + "\n", encoding="utf-8")
        logging.info("Translations saved to %s", out_path)

        # Compute metrics using sacrebleu
        if all(translations) and all(reference_lines):
            bleu = sacrebleu.corpus_bleu(translations, [reference_lines]).score
            chrf = sacrebleu.corpus_chrf(translations, [reference_lines]).score
            ter = sacrebleu.corpus_ter(translations, [reference_lines]).score
        else:
            logging.warning("Empty translations or references; skipping metric computation")
            bleu, chrf, ter = 0.0, 0.0, 0.0

        metrics = {"BLEU": bleu, "chrF++": chrf, "TER": ter}
        score_path = dataset_root / "results" / "scores" / "test_scores.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Evaluation done. BLEU=%.2f chrF++=%.2f TER=%.2f", bleu, chrf, ter)

    except Exception as e:
        logging.exception("Evaluation failed: %s", e)
        raise
