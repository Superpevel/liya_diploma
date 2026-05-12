from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    PROJECT_ROOT = "/content/drive/MyDrive/liya_diploma"
    AI_TOOLKIT = "/content/ai-toolkit"
except ImportError:
    IN_COLAB = False
    PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())
    AI_TOOLKIT = str(Path(PROJECT_ROOT).parent / "ai-toolkit")
