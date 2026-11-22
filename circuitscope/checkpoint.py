"""Fetch exactly two files from the pinned public checkpoint."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHT_HASH = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
WEIGHT_BYTES = 548105171


def verify(path):
    config_digest = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    if config_digest != "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4":
        raise ValueError("GPT-2 configuration digest mismatch")
    with (path / "model.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHT_HASH:
            raise ValueError("GPT-2 checkpoint digest mismatch")


def fetch(path):
    path.mkdir(parents=True, exist_ok=True)
    for name, limit in (("model.safetensors", WEIGHT_BYTES), ("config.json", 8192)):
        final = path / name
        if final.exists():
            continue
        tmp = final.with_suffix(".part")
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        size = 0
        with urlopen(url, timeout=60) as source, tmp.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Checkpoint file exceeds expected size")
                dest.write(chunk)
        if name == "model.safetensors":
            with tmp.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHT_HASH:
                    raise ValueError("Downloaded weights failed verification")
        tmp.replace(final)
    verify(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path(".weights"))
    fetch(parser.parse_args().weights)
