import logging
from pathlib import Path
import sentencepiece as spm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def tokenize_file(sp, input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            pieces = sp.encode(line, out_type=str)
            fout.write(" ".join(pieces) + "\n")

    logging.info("Tokenized %s -> %s", input_path, output_path)


if __name__ == "__main__":
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"
        tokens_dir = dataset_root / "data" / "tokenized"
        processed_dir = dataset_root / "data" / "processed"
        tokens_dir.mkdir(parents=True, exist_ok=True)

        sp_model = dataset_root / "tokenizer" / "sentencepiece" / "spm.model"
        if not sp_model.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {sp_model}")

        sp = spm.SentencePieceProcessor()
        sp.load(str(sp_model))

        for split in ["train", "dev", "test"]:
            for lang in ["eng_Latn", "mni_Beng"]:
                src = processed_dir / f"{split}.{lang}"
                tgt = tokens_dir / f"{split}.{lang}"
                tokenize_file(sp, src, tgt)

        logging.info("Tokenization applied to train/dev/test.")
    except Exception as e:
        logging.exception("Apply tokenizer failed: %s", e)
        raise
