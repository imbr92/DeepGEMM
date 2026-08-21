#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests" / "test_mega_moe.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GLM-5.2 BF16 Mega-MoE EP matrix")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 1024, 4096])
    parser.add_argument("--skews", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num-correctness-tests", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for tokens in args.tokens:
        for skew in args.skews:
            command = [
                sys.executable,
                str(TEST),
                "--num-processes", "8",
                "--num-max-tokens-per-rank", str(tokens),
                "--num-tokens", str(tokens),
                "--hidden", "6144",
                "--intermediate-hidden", "2048",
                "--num-shared-experts", "1",
                "--num-experts", "256",
                "--num-topk", "8",
                "--activation-clamp", "10",
                "--fast-math", "1",
                "--mma-type", "bf16xbf16",
                "--routing-pattern", "balanced" if skew == 1 else "skew",
                "--skew-factor", str(skew),
                "--num-correctness-tests", str(args.num_correctness_tests),
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
