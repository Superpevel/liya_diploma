import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


SVG_NS = "{http://www.w3.org/2000/svg}"


def count_svg_paths(svg_path: str) -> int:
    """Сколько <path> в svg. -1 если не удалось распарсить."""
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return -1
    return sum(1 for e in root.iter() if e.tag in ("path", f"{SVG_NS}path"))


def get_aspect_ratio(png_path: str) -> float:
    """Отношение width/height. -1.0 если картинку не открыть."""
    try:
        w, h = Image.open(png_path).size
    except Exception:
        return -1.0
    return w / h


def filter_dataset(
    png_dir: str,
    svg_dir: str | None = None,
    min_paths: int = 3,
    max_paths: int = 500,
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
) -> list[dict]:
    """
    Отбирает PNG (и парные SVG если есть) по числу путей и пропорциям.
    Если svg_dir не задан или там нет svg — проверка по svg пропускается.
    """
    png_dir = Path(png_dir)
    svg_dir_path = Path(svg_dir) if svg_dir else None
    have_svgs = svg_dir_path is not None and any(svg_dir_path.glob("*.svg"))

    out = []
    for png_path in sorted(png_dir.glob("*.png")):
        svg_path = None
        if have_svgs:
            candidate = svg_dir_path / (png_path.stem + ".svg")
            if not candidate.exists():
                continue
            n = count_svg_paths(str(candidate))
            if not (min_paths <= n <= max_paths):
                continue
            svg_path = str(candidate)

        ratio = get_aspect_ratio(str(png_path))
        if not (min_ratio <= ratio <= max_ratio):
            continue

        out.append({"png_path": str(png_path), "svg_path": svg_path})
    return out
