from __future__ import annotations

from typing import Optional

from . import config
from .generator import ImageGenerator
from .storage import Storage

_storage: Optional[Storage] = None
_generator: Optional[ImageGenerator] = None
_settings: Optional[config.Settings] = None


def init(storage: Storage, generator: ImageGenerator,
         settings: config.Settings) -> None:
    global _storage, _generator, _settings
    _storage = storage
    _generator = generator
    _settings = settings


def storage() -> Storage:
    assert _storage is not None, "deps.init() not called"
    return _storage


def generator() -> ImageGenerator:
    assert _generator is not None, "deps.init() not called"
    return _generator


def settings() -> config.Settings:
    assert _settings is not None, "deps.init() not called"
    return _settings
