"""Generate application icon set for Vector Dataset Toolkit.

This script downloads or loads the base PNG icon and produces PNG, ICO,
ICNS, and SVG outputs in ``resources/icons``.
"""
from __future__ import annotations

import argparse
import io
import sys
import textwrap
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "resources" / "icons"
DEFAULT_SOURCE_PATH = ROOT / "docs" / "milestones" / "vector_dataset_tool.png"
DEFAULT_ICON_URL = (
    "https://raw.githubusercontent.com/zeevbensender/vector-dataset-toolkit/main/"
    "docs/milestones/vector_dataset_tool.png"
)
ICON_BASENAME = "vector_dataset_tool"
SIZES = [16, 32, 48, 64, 128, 256, 512]
SVG_TEMPLATE = textwrap.dedent(
    """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img">
      <title>Vector Dataset Toolkit</title>
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#1b9ad7" />
          <stop offset="100%" stop-color="#084d8c" />
        </linearGradient>
        <linearGradient id="accent" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#ffd166" />
          <stop offset="100%" stop-color="#fca311" />
        </linearGradient>
      </defs>
      <rect x="32" y="32" width="448" height="448" rx="72" fill="url(#bg)" />
      <g stroke="#ffffff" stroke-width="18" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <path d="M132 348c32-80 80-120 144-144 22-8 44-12 72-12" />
        <path d="M152 224c24 16 48 24 72 24 48 0 76-28 112-56" />
        <circle cx="168" cy="356" r="22" fill="url(#accent)" stroke="none" />
        <circle cx="260" cy="296" r="22" fill="url(#accent)" stroke="none" />
        <circle cx="352" cy="192" r="22" fill="url(#accent)" stroke="none" />
      </g>
      <text x="256" y="416" text-anchor="middle" font-family="'Inter', 'Segoe UI', sans-serif" font-size="72" font-weight="700" fill="#ffffff">
        V
      </text>
    </svg>
    """
).strip()


def _load_image(source: Path | None, url: str) -> Image.Image:
    if Image is None:  # pragma: no cover - utility guard
        raise SystemExit(
            "Pillow is required to generate icons. Install it with `pip install Pillow`"
        ) from PIL_IMPORT_ERROR

    if source is None and DEFAULT_SOURCE_PATH.exists():
        source = DEFAULT_SOURCE_PATH

    if source and source.exists():
        return Image.open(source).convert("RGBA")

    with urlopen(url) as response:  # noqa: S310 - trusted URL owned by project
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _ensure_dirs() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)


def _save_base_png(image: Image.Image) -> Path:
    output_path = ICONS_DIR / f"{ICON_BASENAME}.png"
    image.save(output_path)
    return output_path


def _generate_resized_images(image: Image.Image) -> dict[int, Image.Image]:
    resized: dict[int, Image.Image] = {}
    for size in SIZES:
        resized[size] = image.resize((size, size), Image.Resampling.LANCZOS)
    return resized


def _save_png_variants(resized: dict[int, Image.Image]) -> list[Path]:
    outputs: list[Path] = []
    for size, icon in resized.items():
        output_path = ICONS_DIR / f"{ICON_BASENAME}_{size}x{size}.png"
        icon.save(output_path)
        outputs.append(output_path)
    return outputs


def _save_ico(resized: dict[int, Image.Image]) -> Path:
    output_path = ICONS_DIR / f"{ICON_BASENAME}.ico"
    base_image = resized[max(resized)]
    base_image.save(output_path, format="ICO", sizes=[(s, s) for s in SIZES])
    return output_path


def _save_icns(resized: dict[int, Image.Image]) -> Path:
    output_path = ICONS_DIR / f"{ICON_BASENAME}.icns"
    resized[max(resized)].save(output_path, format="ICNS", sizes=[(s, s) for s in SIZES])
    return output_path


def _save_svg_placeholder(resized: dict[int, Image.Image]) -> Path:
    """Write a standalone SVG that mirrors the shipped icon design."""

    output_path = ICONS_DIR / f"{ICON_BASENAME}.svg"
    output_path.write_text(SVG_TEMPLATE + "\n", encoding="utf-8")
    return output_path


def generate_icons(source: Path | None, url: str, skip_svg: bool = False) -> None:
    _ensure_dirs()

    try:
        source_image = _load_image(source, url)
    except (URLError, OSError) as exc:  # pragma: no cover - utility error handling
        message = f"Unable to load source image from {source or url}: {exc}"
        raise SystemExit(message) from exc

    resized = _generate_resized_images(source_image)
    base_png = _save_base_png(resized[max(resized)])
    png_variants = _save_png_variants(resized)
    ico = _save_ico(resized)
    icns = _save_icns(resized)
    svg = _save_svg_placeholder(resized) if not skip_svg else None

    print("Generated icon assets:")
    for path in [base_png, *png_variants, ico, icns]:
        print(f" - {path.relative_to(ROOT)}")
    if svg:
        print(f" - {svg.relative_to(ROOT)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate application icons")
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to a local PNG to use instead of downloading the default icon.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ICON_URL,
        help="URL for downloading the source icon (used when --source is not provided).",
    )
    parser.add_argument(
        "--skip-svg",
        action="store_true",
        help="Skip creation of the SVG placeholder if not required.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    generate_icons(args.source, args.url, skip_svg=args.skip_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
