import json
import logging
from pathlib import Path

# fairseq2 imports - might need custom install path
try:
    from fairseq2.data import DataPipelineBuilder
    from fairseq2.models import nllb
except ImportError:
    DataPipelineBuilder = None
    nllb = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def save_manifest(manifest_data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in manifest_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info("Saved manifest to %s", output_path)


def init_mni_embedding(model, dict_tokens):
    # Create or update mni_Beng token at vocab index if missing.
    if "mni_Beng" not in dict_tokens:
        logging.info("Adding mni_Beng token to dict")
        dict_tokens.append("mni_Beng")

    # Copy from ben_Beng if available.
    if model is not None and hasattr(model, "encoder"):
        try:
            emb = model.encoder.embed_tokens
            if "ben_Beng" in dict_tokens and "mni_Beng" in dict_tokens:
                i_ben = dict_tokens.index("ben_Beng")
                i_mni = dict_tokens.index("mni_Beng")
                emb.weight.data[i_mni] = emb.weight.data[i_ben].detach().clone()
                logging.info("Initialized mni_Beng embedding from ben_Beng embedding")
        except Exception as e:
            logging.warning("Could not init mni_Beng embedding: %s", e)


if __name__ == "__main__":
    setup_logging()
    try:
        dataset_root = Path.home() / "eng_mni_nmt"
        tok_dir = dataset_root / "data" / "tokenized"
        prepared_dir = dataset_root / "data" / "prepared"

        source = "eng_Latn"
        target = "mni_Beng"

        splits = ["train", "dev", "test"]
        manifest_data = []

        for split in splits:
            src_path = tok_dir / f"{split}.{source}"
            tgt_path = tok_dir / f"{split}.{target}"
            if not src_path.exists() or not tgt_path.exists():
                raise FileNotFoundError(f"Missing tokenized files for {split}")

            src_lines = load_lines(src_path)
            tgt_lines = load_lines(tgt_path)
            if len(src_lines) != len(tgt_lines):
                raise ValueError(f"Lengths mismatch on {split}: {len(src_lines)} vs {len(tgt_lines)}")

            split_dir = prepared_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # write plain line files used by fairseq2 loader
            (split_dir / f"{source}.txt").write_text("\n".join(src_lines) + "\n", encoding="utf-8")
            (split_dir / f"{target}.txt").write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")

            logging.info("Prepared %d examples for %s", len(src_lines), split)

            # Create manifest record
            manifest_data.append({
                "split": split,
                "src_path": str(split_dir / f"{source}.txt"),
                "tgt_path": str(split_dir / f"{target}.txt"),
                "num_samples": len(src_lines),
            })

        manifest_file = prepared_dir / "manifest.jsonl"
        save_manifest(manifest_data, manifest_file)

        # DataPipelineBuilder usage placeholder; instantiate if available
        if DataPipelineBuilder is None:
            logging.warning("fairseq2.data.DataPipelineBuilder is unavailable in current environment")
        else:
            builder = DataPipelineBuilder()
            builder.add_source_field(source)
            builder.add_target_field(target)
            builder.build()
            logging.info("Initialized DataPipelineBuilder for %s->%s", source, target)

        # Attempt to initialize embedding from pretrained NLLB
        if nllb is not None:
            try:
                model = nllb.NLLB.from_pretrained("nllb-200-distilled-600M")
                dict_tokens = ["ben_Beng", "mni_Beng"]
                init_mni_embedding(model, dict_tokens)
            except Exception as e:
                logging.warning("Failed to load pretrained NLLB for embedding initialization: %s", e)

        logging.info("Data preparation complete.")

    except Exception as e:
        logging.exception("Data preparation failed: %s", e)
        raise
