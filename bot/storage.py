import asyncio
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    variants: list


class Storage:
    def __init__(self, db_path: Path = config.DB_PATH,
                 images_dir: Path = config.STORAGE_DIR):
        self.db_path = db_path
        self.images_dir = images_dir
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def get_user_model(self, user_id: int) -> str:
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT model_key FROM user_settings WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
        return row["model_key"] if row else config.DEFAULT_MODEL_KEY

    async def set_user_model(self, user_id: int, model_key: str) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO user_settings(user_id, model_key) VALUES(?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET model_key = excluded.model_key",
                    (user_id, model_key),
                )

    def _save_images(self, gen_id, images, seeds):
        out_dir = self.images_dir / gen_id
        out_dir.mkdir(parents=True, exist_ok=True)
        variants = []
        for idx, (img, seed) in enumerate(zip(images, seeds)):
            vid = uuid.uuid4().hex
            path = out_dir / f"variant_{idx + 1}_{vid}.png"
            img.save(path, format="PNG")
            variants.append(Variant(id=vid, idx=idx, file_path=path, seed=seed))
        return variants

    async def save_generation(self, user_id, prompt, model_key, images, seeds) -> Generation:
        gen_id = uuid.uuid4().hex
        created_at = int(time.time())
        variants = await asyncio.to_thread(
            self._save_images, gen_id, images, seeds
        )
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO generations(id, user_id, prompt, model_key, created_at) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (gen_id, user_id, prompt, model_key, created_at),
                )
                conn.executemany(
                    "INSERT INTO variants(id, generation_id, idx, file_path, seed) "
                    "VALUES(?, ?, ?, ?, ?)",
                    [(v.id, gen_id, v.idx, str(v.file_path), v.seed) for v in variants],
                )
        return Generation(
            id=gen_id, user_id=user_id, prompt=prompt, model_key=model_key,
            created_at=created_at, variants=variants,
        )

    def _row_to_variant(self, r) -> Variant:
        return Variant(id=r["id"], idx=r["idx"],
                       file_path=Path(r["file_path"]), seed=r["seed"])

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
                    "SELECT * FROM variants WHERE generation_id = ? ORDER BY idx ASC",
                    (gen_row["id"],),
                ).fetchall()
        return Generation(
            id=gen_row["id"], user_id=gen_row["user_id"],
            prompt=gen_row["prompt"], model_key=gen_row["model_key"],
            created_at=gen_row["created_at"],
            variants=[self._row_to_variant(r) for r in var_rows],
        )

    async def recent_generations(self, user_id: int, ttl_hours: int) -> list:
        cutoff = int(time.time()) - ttl_hours * 3600
        async with self._lock:
            with self._connect() as conn:
                gen_rows = conn.execute(
                    "SELECT * FROM generations WHERE user_id = ? AND created_at >= ? "
                    "ORDER BY created_at DESC",
                    (user_id, cutoff),
                ).fetchall()
                if not gen_rows:
                    return []
                placeholders = ",".join("?" for _ in gen_rows)
                var_rows = conn.execute(
                    f"SELECT * FROM variants WHERE generation_id IN ({placeholders}) "
                    "ORDER BY idx ASC",
                    [g["id"] for g in gen_rows],
                ).fetchall()

        by_gen: dict = {}
        for r in var_rows:
            by_gen.setdefault(r["generation_id"], []).append(self._row_to_variant(r))

        out = []
        for g in gen_rows:
            variants = [v for v in by_gen.get(g["id"], []) if v.file_path.exists()]
            out.append(Generation(
                id=g["id"], user_id=g["user_id"], prompt=g["prompt"],
                model_key=g["model_key"], created_at=g["created_at"],
                variants=variants,
            ))
        return out

    async def save_feedback(self, user_id, generation_id, variant_id) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO feedback(user_id, generation_id, variant_id, created_at) "
                    "VALUES(?, ?, ?, ?)",
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
                expired = [r["id"] for r in rows]
        for gen_id in expired:
            gen_dir = self.images_dir / gen_id
            if gen_dir.exists():
                shutil.rmtree(gen_dir, ignore_errors=True)
        return len(expired)


async def cleanup_loop(storage: Storage, ttl_hours: int) -> None:
    while True:
        try:
            await storage.cleanup_expired(ttl_hours)
        except Exception:
            pass
        await asyncio.sleep(3600)
