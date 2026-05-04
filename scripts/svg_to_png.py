import cairosvg
from PIL import Image
import io
from pathlib import Path


def svg_to_png(svg_path: str, output_path: str, size: int = 512) -> bool:
    """Convert SVG to PNG with white background. Returns True on success."""
    try:
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.convert("RGB").save(output_path, "PNG")
        return True
    except Exception:
        return False


def batch_convert(svg_dir: str, png_dir: str, size: int = 512) -> dict:
    """Convert all SVGs in svg_dir to PNGs in png_dir. Returns stats dict."""
    svg_dir = Path(svg_dir)
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    svgs = list(svg_dir.glob("**/*.svg"))
    success, failed = 0, 0

    for svg_path in svgs:
        out_path = png_dir / (svg_path.stem + ".png")
        if svg_to_png(str(svg_path), str(out_path), size):
            success += 1
        else:
            failed += 1

    return {"total": len(svgs), "success": success, "failed": failed}
