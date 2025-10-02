from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Optional
from statistics import StatisticsError, median, quantiles

from datasets import load_dataset
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def iter_dataset_image_sizes(
    data_files: list[str],
    image_column: str = "image",
    text_column: Optional[str] = "latex_formula",
) -> Iterator[tuple[int, int, str, Optional[int]]]:
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    for row in dataset:
        image = row.get(image_column)
        text_len: Optional[int] = None
        if text_column and text_column in row and isinstance(row[text_column], str):
            text_len = len(row[text_column])
        if image is None:
            continue
        if isinstance(image, Image.Image):
            yield image.width, image.height, row.get("id", ""), text_len
            continue
        if isinstance(image, dict):
            if image.get("bytes"):
                try:
                    with Image.open(BytesIO(image["bytes"])) as img:
                        yield img.width, img.height, image.get("path") or "", text_len
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping unreadable image bytes for record: {row}", file=sys.stderr)
                continue
            if image.get("path"):
                path = Path(image["path"])
                try:
                    with Image.open(path) as img:
                        yield img.width, img.height, str(path), text_len
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping unreadable image: {path}", file=sys.stderr)
                continue


def find_images(root: Path, recursive: bool = True) -> Iterable[Path]:
    globber = root.rglob if recursive else root.glob
    for path in globber("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _quartiles(values: list[int]) -> tuple[float, float, float]:
    if len(values) <= 1:
        only = float(values[0]) if values else float("nan")
        return only, only, only
    try:
        q1, q2, q3 = quantiles(values, n=4, method="inclusive")
    except StatisticsError:
        mid = float(median(values))
        return mid, mid, mid
    return q1, q2, q3


def compute_dimension_stats(
    images: Iterable[tuple[int, int, str, Optional[int]]]
) -> tuple[dict[str, float], dict[str, float], Optional[dict[str, float]], int, dict[str, tuple[int, str]]]:
    total_width = 0
    total_height = 0
    count = 0
    widths: list[int] = []
    heights: list[int] = []
    text_lengths: list[int] = []
    extrema: dict[str, tuple[int, str]] = {
        "min_width": (sys.maxsize, ""),
        "max_width": (0, ""),
        "min_height": (sys.maxsize, ""),
        "max_height": (0, ""),
    }

    for width, height, origin, text_len in images:
        total_width += width
        total_height += height
        count += 1
        widths.append(width)
        heights.append(height)
        if text_len is not None:
            text_lengths.append(text_len)

        if width < extrema["min_width"][0]:
            extrema["min_width"] = (width, origin)
        if width > extrema["max_width"][0]:
            extrema["max_width"] = (width, origin)
        if height < extrema["min_height"][0]:
            extrema["min_height"] = (height, origin)
        if height > extrema["max_height"][0]:
            extrema["max_height"] = (height, origin)

    if count == 0:
        raise ValueError("No readable images found.")

    avg_width = total_width / count
    avg_height = total_height / count
    width_q1, width_median, width_q3 = _quartiles(widths)
    height_q1, height_median, height_q3 = _quartiles(heights)

    width_stats = {
        "min": float(min(widths)),
        "q1": width_q1,
        "median": width_median,
        "q3": width_q3,
        "max": float(max(widths)),
        "mean": avg_width,
    }
    height_stats = {
        "min": float(min(heights)),
        "q1": height_q1,
        "median": height_median,
        "q3": height_q3,
        "max": float(max(heights)),
        "mean": avg_height,
    }

    text_stats: Optional[dict[str, float]] = None
    if text_lengths:
        text_q1, text_median, text_q3 = _quartiles(text_lengths)
        avg_len = sum(text_lengths) / len(text_lengths)
        text_stats = {
            "min": float(min(text_lengths)),
            "q1": text_q1,
            "median": text_median,
            "q3": text_q3,
            "max": float(max(text_lengths)),
            "mean": avg_len,
        }

    return width_stats, height_stats, text_stats, count, extrema


def format_stats_line(label: str, stats: dict[str, float], unit: str = "") -> str:
    order = ("min", "q1", "median", "q3", "max", "mean")
    unit_suffix = f" {unit}" if unit else ""

    def fmt(value: float) -> str:
        if value != value:  # NaN guard
            return "nan"
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}{unit_suffix}"
        return f"{value:.2f}{unit_suffix}"

    parts = [f"{key}={fmt(stats[key])}" for key in order if key in stats]
    return f"{label}: {', '.join(parts)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute width/height summary statistics for images in a dataset.",
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("datasets"),
        help="Path to a dataset directory or parquet file (default: datasets).",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the image column inside the dataset (default: image).",
    )
    parser.add_argument(
        "--text-column",
        default="latex_formula",
        help="Name of the LaTeX text column for length stats (default: latex_formula).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="When reading image files directly, avoid recursing into subdirectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root

    if not root.exists():
        print(f"Dataset path not found: {root}", file=sys.stderr)
        return 1

    images_iter: Iterable[tuple[int, int, str, Optional[int]]]
    parquet_files: list[Path] = []
    direct_images: list[Path] = []

    if root.is_file():
        suffix = root.suffix.lower()
        if suffix == ".parquet":
            parquet_files = [root]
        elif suffix in SUPPORTED_EXTENSIONS:
            direct_images = [root]
        else:
            print(f"Unsupported file extension: {suffix}", file=sys.stderr)
            return 1
    elif root.is_dir():
        globber = root.rglob if not args.non_recursive else root.glob
        parquet_files = list(globber("*.parquet"))
    else:
        print(f"Unsupported path type: {root}", file=sys.stderr)
        return 1

    if parquet_files:
        data_files = [str(path) for path in parquet_files]

        def parquet_sizes() -> Iterator[tuple[int, int, str, Optional[int]]]:
            for width, height, origin, text_len in iter_dataset_image_sizes(
                data_files,
                image_column=args.image_column,
                text_column=args.text_column,
            ):
                yield width, height, origin or "parquet record", text_len

        images_iter = parquet_sizes()
    else:
        def file_sizes() -> Iterator[tuple[int, int, str, Optional[int]]]:
            paths = direct_images or list(find_images(root, recursive=not args.non_recursive))
            for path in paths:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping unreadable image: {path}", file=sys.stderr)
                    continue
                yield width, height, str(path), None

        images_iter = file_sizes()

    try:
        width_stats, height_stats, text_stats, count, extrema = compute_dimension_stats(images_iter)
    except ValueError as exc:  # pragma: no cover - purely defensive
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Processed {count} images.")
    print(format_stats_line("width", width_stats, unit="px"))
    print(format_stats_line("height", height_stats, unit="px"))
    if text_stats:
        print(format_stats_line("latex formula length", text_stats, unit="chars"))
    print("Extrema:")
    print(f"  Min width:  {extrema['min_width'][0]}px ({extrema['min_width'][1]})")
    print(f"  Max width:  {extrema['max_width'][0]}px ({extrema['max_width'][1]})")
    print(f"  Min height: {extrema['min_height'][0]}px ({extrema['min_height'][1]})")
    print(f"  Max height: {extrema['max_height'][0]}px ({extrema['max_height'][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
