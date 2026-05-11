"""Download selected HuggingFaceM4/FineVision subsets listed in subsets.txt.

Layout on HF: each config lives at `<config>/train-*.parquet` at the repo root.
snapshot_download is idempotent — rerun after editing subsets.txt.

For faster transfers, set HF_HUB_ENABLE_HF_TRANSFER=1 before running.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).parent
DATA = Path("/local/<user>/tool-vision/data/FineVision_Maybe")
REPO_ID = "HuggingFaceM4/FineVision"


def read_subsets(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main() -> None:
    subsets = read_subsets(HERE / "subsets.txt")
    if not subsets:
        raise SystemExit(f"No subsets listed in {HERE / 'subsets.txt'}")

    patterns = [f"{s}/*" for s in subsets] + ["README.md"]
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(subsets)} subsets to {DATA}")
    for s in subsets:
        print(f"  - {s}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA),
        allow_patterns=patterns,
        max_workers=8,
    )
    print(f"Done. Downloaded {len(subsets)} subsets to {DATA}")


if __name__ == "__main__":
    main()
