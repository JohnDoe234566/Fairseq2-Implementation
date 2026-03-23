import logging
import random
import unicodedata2 as unicodedata
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_text(text: str, is_mni: bool = False) -> str:
    # Normalize Unicode and optionally apply Bengali script normalizations.
    norm = unicodedata.normalize("NFKC", text.strip())
    if is_mni:
        # Bengali script normalization: canonical composition + remove control spacing markers
        norm = unicodedata.normalize("NFC", norm)
        norm = norm.replace("\u200d", "").replace("\u200c", "")
    return norm


def load_parallel(src_path: Path, tgt_path: Path):
    with src_path.open("r", encoding="utf-8") as fsrc:
        src_lines = [l.strip() for l in fsrc]
    with tgt_path.open("r", encoding="utf-8") as ftgt:
        tgt_lines = [l.strip() for l in ftgt]

    if len(src_lines) != len(tgt_lines):
        logging.warning(
            "Source length (%d) != target length (%d), truncating to min length",
            len(src_lines),
            len(tgt_lines),
        )
    N = min(len(src_lines), len(tgt_lines))

    cleaned_src = []
    cleaned_tgt = []
    for i in range(N):
        src = normalize_text(src_lines[i], is_mni=False)
        tgt = normalize_text(tgt_lines[i], is_mni=True)

        if not src or not tgt:
            continue

        cleaned_src.append(src)
        cleaned_tgt.append(tgt)

    logging.info("Loaded %d valid sentence pairs", len(cleaned_src))
    return cleaned_src, cleaned_tgt


def split_and_write(src_lines, tgt_lines, output_dir: Path, seed=42):
    random.seed(seed)
    indices = list(range(len(src_lines)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    n_test = n - n_train - n_dev

    splits = {
        "train": indices[:n_train],
        "dev": indices[n_train : n_train + n_dev],
        "test": indices[n_train + n_dev :],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ids in splits.items():
        src_out = output_dir / f"{split_name}.eng_Latn"
        tgt_out = output_dir / f"{split_name}.mni_Beng"

        with src_out.open("w", encoding="utf-8") as fsrc, tgt_out.open("w", encoding="utf-8") as ftgt:
            for idx in split_ids:
                fsrc.write(src_lines[idx] + "\n")
                ftgt.write(tgt_lines[idx] + "\n")

        logging.info(
            "%s: %d examples written to %s and %s",
            split_name,
            len(split_ids),
            src_out,
            tgt_out,
        )


if __name__ == "__main__":
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"
        raw_dir = Path("/workspaces/Fairseq2-Implementation/eng_Latn-mni_Beng")
        out_dir = dataset_root / "data" / "processed"

        src_file = raw_dir / "train.eng_Latn"
        tgt_file = raw_dir / "train.mni_Beng"

        if not src_file.exists() or not tgt_file.exists():
            raise FileNotFoundError(f"Missing dataset files at {raw_dir}")

        src, tgt = load_parallel(src_file, tgt_file)

        split_and_write(src, tgt, out_dir)
        logging.info("Preprocessing complete.")

    except Exception as e:
        logging.exception("Preprocessing failed: %s", e)
        raise
