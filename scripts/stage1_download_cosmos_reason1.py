from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cosmos_reason1/raw", help="Output dir")
    p.add_argument("--repo-id", default="nvidia/Cosmos-Reason1-Benchmark")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("huggingface_hub not installed. Please pip install huggingface_hub") from e

    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=args.out)
    print(args.out)


if __name__ == "__main__":
    main()
