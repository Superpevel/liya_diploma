from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    model_key TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS variants (
    id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    seed INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    generation_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY,
    model_key TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gen_user_time
    ON generations(user_id, created_at DESC);
"""


@dataclass
class Variant:
    id: str
    idx: int
    file_path: Path
    seed: int


@dataclass
class Generation:
    id: str
    user_id: int
    prompt: str
    model_key: str
    created_at: int
    variants: List[Variant]


class Storage:
    def __init__(self, db_path: Path = config.DB_PATH,
                 images_dir: Path = config.STORAGE_DIR) -> None:
        self.db_path = db_path
        self.images_dir = images_dir
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- user settings ---

    async def get_user_model(self, user_id: int) -> str:
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT model_key FROM user_settings WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
        if row is None:
            return config.DEFAULT_MODEL_KEY
        return row["model_key"]

    async def set_user_model(self, user_id: int, model_key: str) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO user_settings(user_id, model_key) "
                    "VALUES(?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                    "model_key = excluded.model_key",
                    (user_id, model_key),
                )

    # --- generations ---

    def _save_images(self, gen_id: str, images: Sequence[Image.Image],
                     seeds: Sequence[int]) -> List[Variant]:
        out_dir = self.images_dir / gen_id
        out_dir.mkdir(parents=True, exist_ok=True)
        variants: List[Variant] = []
        for idx, (img, seed) in enumerate(zip(images, seeds)):
            variant_id = uuid.uuid4().hex
            file_path = out_dir / f"variant_{idx + 1}_{variant_id}.png"
            img.save(file_path, format="PNG")
            variants.append(Variant(id=variant_id, idx=idx,
                                    file_path=file_path, seed=seed))
        return variants

    async def save_generation(
        self,
        user_id: int,
        prompt: str,
        model_key: str,
        images: Sequence[Image.Image],
        seeds: Sequence[int],
    ) -> Generation:
        gen_id = uuid.uuid4().hex
        created_at = int(time.time())
        variants = await asyncio.to_thread(
            self._save_images, gen_id, images, seeds
        )
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO generations(id, user_id, prompt, model_key, "
                    "created_at) VALUES(?, ?, ?, ?, ?)",
                    (gen_id, user_id, prompt, model_key, created_at),
                )
                conn.executemany(
                    "INSERT INTO variants(id, generation_id, idx, file_path, "
                    "seed) VALUES(?, ?, ?, ?, ?)",
                    [
                        (v.id, gen_id, v.idx, str(v.file_path), v.seed)
                        for v in variants
                    ],
                )
        return Generation(
            id=gen_id,
            user_id=user_id,
            prompt=prompt,
            model_key=model_key,
            created_at=created_at,
            variants=variants,
        )

    async def last_generation(self, user_id: int) -> Optional[Generation]:
        async with self._lock:
            with self._connect() as conn:
                gen_row = conn.execute(
                    "SELECT * FROM generations WHERE user_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (user_id,),
                ).fetchone()
                if gen_row is None:
                    return None
                var_rows = conn.execute(
                    "SELECT * FROM variants WHERE generation_id = ? "
                    "ORDER BY idx ASC",
                    (gen_row["id"],),
                ).fetchall()
        return Generation(
            id=gen_row["id"],
            user_id=gen_row["user_id"],
            prompt=gen_row["prompt"],
            model_key=gen_row["model_key"],
            created_at=gen_row["created_at"],
            variants=[
                Variant(
                    id=r["id"],
                    idx=r["idx"],
                    file_path=Path(r["file_path"]),
                    seed=r["seed"],
                )
                for r in var_rows
            ],
        )

    async def recent_generations(
        self, user_id: int, ttl_hours: int
    ) -> List[Generation]:
        cutoff = int(time.time()) - ttl_hours * 3600
        async with self._lock:
            with self._connect() as conn:
                gen_rows = conn.execute(
                    "SELECT * FROM generations WHERE user_id = ? "
                    "AND created_at >= ? ORDER BY created_at DESC",
                    (user_id, cutoff),
                ).fetchall()
                if not gen_rows:
                    return []
                placeholders = ",".join("?" for _ in gen_rows)
                var_rows = conn.execute(
                    f"SELECT * FROM variants WHERE generation_id IN "
                    f"({placeholders}) ORDER BY idx ASC",
                    [g["id"] for g in gen_rows],
                ).fetchall()
        variants_by_gen: dict[str, List[Variant]] = {}
        for r in var_rows:
            variants_by_gen.setdefault(r["generation_id"], []).append(
                Variant(
                    id=r["id"],
                    idx=r["idx"],
                    file_path=Path(r["file_path"]),
                    seed=r["seed"],
                )
            )
        result: List[Generation] = []
        for g in gen_rows:
            variants = variants_by_gen.get(g["id"], [])
            variants = [v for v in variants if v.file_path.exists()]
            result.append(
                Generation(
                    id=g["id"],
                    user_id=g["user_id"],
                    prompt=g["prompt"],
                    model_key=g["model_key"],
                    created_at=g["created_at"],
                    variants=variants,
                )
            )
        return result

    async def save_feedback(
        self, user_id: int, generation_id: str, variant_id: str
    ) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO feedback(user_id, generation_id, variant_id, "
                    "created_at) VALUES(?, ?, ?, ?)",
                    (user_id, generation_id, variant_id, int(time.time())),
                )

    async def cleanup_expired(self, ttl_hours: int) -> int:
        cutoff = int(time.time()) - ttl_hours * 3600
        async with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id FROM generations WHERE created_at < ?",
                    (cutoff,),
                ).fetchall()
                expired_ids = [r["id"] for r in rows]
        for gen_id in expired_ids:
            gen_dir = self.images_dir / gen_id
            if gen_dir.exists():
                for f in gen_dir.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
                try:
                    gen_dir.rmdir()
                except OSError:
                    pass
        return len(expired_ids)


async def cleanup_loop(storage: Storage, ttl_hours: int) -> None:
    while True:
        try:
            await storage.cleanup_expired(ttl_hours)
        except Exception:
            pass
        await asyncio.sleep(3600)
