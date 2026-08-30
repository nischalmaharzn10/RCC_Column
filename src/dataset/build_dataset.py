"""CLI: build processed ML tables from PEER samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.service import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build RCC Column ML datasets from PEER samples")
    parser.add_argument(
        "--properties",
        type=Path,
        default=None,
        help="Properties CSV/TXT (default: samples/properties/rectangular_properties.csv)",
    )
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=None,
        help="Directory of force-displacement .txt files",
    )
    parser.add_argument(
        "--curve-index",
        type=Path,
        default=None,
        help="specimen ↔ curve_file index CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/processed)",
    )
    parser.add_argument("--backbone-points", type=int, default=40)
    args = parser.parse_args(argv)

    result = build_dataset(
        properties_path=args.properties,
        curves_dir=args.curves_dir,
        curve_index_path=args.curve_index,
        out_dir=args.out,
        backbone_points=args.backbone_points,
    )
    print("Wrote:")
    for key, val in result.items():
        print(f"  {key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
