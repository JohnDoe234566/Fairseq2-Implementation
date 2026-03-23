import json
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class FineTuneConfig:
    max_seq_len: int = 256
    batch_size: int = 16
    max_epochs: int = 30
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    dropout: float = 0.3
    optimizer: str = "adamw"
    fp16: bool = True

    def to_dict(self):
        return self.__dict__


def load_pretrained_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    if AutoModelForSeq2SeqLM is None:
        raise ImportError("transformers library is not installed")

    logging.info("Loading pretrained model %s from Hugging Face", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logging.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        raise


if __name__ == "__main__":
    setup_logging()
    try:
        cfg = FineTuneConfig()
        logging.info("Fine-tuning configuration:")
        for k, v in cfg.to_dict().items():
            logging.info("  %s: %s", k, v)

        # Load and verify model
        model, tokenizer = load_pretrained_nllb()
        
        # Save config for training script
        config_path = Path.home() / "eng_mni_nmt" / "model_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        logging.info("Config saved to %s", config_path)

        logging.info("Model configuration setup completed.")
    except Exception as e:
        logging.exception("Model configuration failed: %s", e)
        raise
