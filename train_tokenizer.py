import logging
from pathlib import Path
import sentencepiece as spm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_corpus_file(processed_dir: Path, tmp_corpus_path: Path):
    # combine all splits to one training file
    lines = []
    for split in ["train", "dev", "test"]:
        for lang in ["eng_Latn", "mni_Beng"]:
            path = processed_dir / f"{split}.{lang}"
            if not path.exists():
                raise FileNotFoundError(f"Missing split file: {path}")
            with path.open("r", encoding="utf-8") as f:
                lines.extend([l.strip() for l in f if l.strip()])

    with tmp_corpus_path.open("w", encoding="utf-8") as out_file:
        for line in lines:
            out_file.write(line + "\n")

    logging.info("Combined corpus text saved to %s", tmp_corpus_path)
    return tmp_corpus_path


def train_sentencepiece(input_file: Path, model_prefix: str, vocab_size: int = 8000):
    spm.SentencePieceTrainer.Train(
        input=str(input_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )
    logging.info("SentencePiece model trained: %s.model", model_prefix)


if __name__ == "__main__":
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"
        processed_dir = dataset_root / "data" / "processed"
        tok_dir = dataset_root / "tokenizer" / "sentencepiece"
        tok_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = tok_dir / "joint_corpus.txt"
        model_prefix = str(tok_dir / "spm")

        get_corpus_file(processed_dir, corpus_file)
        train_sentencepiece(corpus_file, model_prefix, vocab_size=8000)

        logging.info("Tokenizer training completed.")
    except Exception as e:
        logging.exception("Tokenizer training failed: %s", e)
        raise
