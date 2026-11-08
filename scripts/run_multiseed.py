"""Multi-seed runner — 3 seeds × 4 model = 12 runs.

Usage:
  python -m scripts.run_multiseed
  python -m scripts.run_multiseed --seeds 42 123 456
  python -m scripts.run_multiseed --seeds 123 456
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data", default="hf")
    parser.add_argument("--n-students", type=int, default=5000)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--bkt-mode", default="per-skill")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed: {len(args.seeds)} seeds × 4 models")
    print(f"Data: {args.data} | n={args.n_students} | max_len={args.max_len} "
          f"| epochs={args.epochs} | bkt={args.bkt_mode}")
    print("=" * 70)

    for i, seed in enumerate(args.seeds, 1):
        out_json = RESULTS / f"seed_{seed}.json"
        print(f"\n[{i}/{len(args.seeds)}] Seed {seed} -> {out_json.name}")
        print("=" * 70)

        cmd = [
            sys.executable, "-m", "scripts.run_pipeline",
            "--data", args.data,
            "--n-students", str(args.n_students),
            "--max-len", str(args.max_len),
            "--epochs", str(args.epochs),
            "--bkt-mode", args.bkt_mode,
            "--seed", str(seed),
            "--output-json", str(out_json),
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Seed {seed} failed (exit {result.returncode})")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"[OK] All {len(args.seeds)} seeds complete")
    print(f"Results in: {RESULTS}")
    print("\nNext: python -m scripts.make_paper_tables")


if __name__ == "__main__":
    main()