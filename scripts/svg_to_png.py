import io
from pathlib import Path

from PIL import Image
from resvg_py import svg_to_bytes


def svg_to_png(svg_path: str, output_path: str, size: int = 512) -> bool:
    try:
        png_bytes = bytes(svg_to_bytes(
            svg_path=str(svg_path),
            width=size,
            height=size,
            background="white",
        ))
        Image.open(io.BytesIO(png_bytes)).convert("RGB").save(output_path, "PNG")
        return True
    except Exception:
        return False


def batch_convert(svg_dir: str, png_dir: str, size: int = 512) -> dict:
    """Конвертирует все svg из svg_dir в png в png_dir."""
    svg_dir = Path(svg_dir)
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    svgs = list(svg_dir.glob("**/*.svg"))
    success = 0
    for svg_path in svgs:
        out_path = png_dir / (svg_path.stem + ".png")
        if svg_to_png(str(svg_path), str(out_path), size):
            success += 1
    return {"total": len(svgs), "success": success, "failed": len(svgs) - success}
