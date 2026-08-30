"""CLI: train peak or curve models."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.service import train_curve, train_peak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train RCC Column ML models")
    parser.add_argument("--mode", choices=["peak", "curve"], required=True)
    parser.add_argument("--data", type=Path, default=None, help="Peak dataset CSV")
    parser.add_argument("--curves", type=Path, default=None, help="Curve backbone CSV")
    parser.add_argument("--target", type=str, default="peak_load_kN")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args(argv)

    progress = not args.quiet
    if args.mode == "peak":
        meta = train_peak(
            data_path=args.data,
            target=args.target,
            models_out=args.models_dir,
            outputs=args.outputs_dir,
            show_progress=progress,
        )
    else:
        meta = train_curve(
            curves_path=args.curves,
            models_out=args.models_dir,
            outputs=args.outputs_dir,
            show_progress=progress,
        )
    print(f"Best model: {meta.get('best_model')}")
    for m in meta.get("metrics", []):
        print(
            f"  {m['model']}: test R2={m.get('test_r2', m.get('r2', 0)):.4f} "
            f"train R2={m.get('train_r2', float('nan')):.4f} "
            f"gap={m.get('overfit_r2_gap', float('nan')):.4f} "
            f"CV RMSE={m.get('cv_rmse_mean', float('nan')):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
