import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path


def count_svg_paths(svg_path: str) -> int:
    """Count path elements in SVG. Returns -1 on parse error."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        paths = {e for e in root.iter()
                 if e.tag in ("path", "{http://www.w3.org/2000/svg}path")}
        return len(paths)
    except Exception:
        return -1


def get_aspect_ratio(png_path: str) -> float:
    """Return width/height ratio of PNG. Returns -1.0 on error."""
    try:
        img = Image.open(png_path)
        w, h = img.size
        return w / h
    except Exception:
        return -1.0


def filter_dataset(
    png_dir: str,
    svg_dir: str | None = None,
    min_paths: int = 3,
    max_paths: int = 500,
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
) -> list[dict]:
    """
    Filter PNGs (and optionally paired SVGs) by path count and aspect ratio.

    If svg_dir contains SVG siblings, each PNG must have a matching SVG with
    a path count in [min_paths, max_paths]. If svg_dir is None or empty
    (PNG-only datasets), the SVG check is skipped.

    Returns list of dicts: {png_path, svg_path or None}.
    """
    png_dir = Path(png_dir)
    svg_dir_path = Path(svg_dir) if svg_dir else None
    have_svgs = svg_dir_path is not None and any(svg_dir_path.glob("*.svg"))
    results = []

    for png_path in sorted(png_dir.glob("*.png")):
        svg_path = None
        if have_svgs:
            candidate = svg_dir_path / (png_path.stem + ".svg")
            if not candidate.exists():
                continue
            n_paths = count_svg_paths(str(candidate))
            if n_paths < min_paths or n_paths > max_paths:
                continue
            svg_path = str(candidate)

        ratio = get_aspect_ratio(str(png_path))
        if ratio < min_ratio or ratio > max_ratio:
            continue

        results.append({"png_path": str(png_path), "svg_path": svg_path})

    return results
