from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class StoredTemplate:
    user_id: str
    tenant_id: str
    protected_template: np.ndarray
    embedding_dim: int
    model_name: str
    transform_version: str
    created_at: str


class TemplateStore:
    def __init__(self, db_path: str = "data/voxis.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    protected_template BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    transform_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, tenant_id)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        embedding = np.asarray(embedding, dtype=np.float32)
        return embedding.tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes, embedding_dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.shape[0] != embedding_dim:
            raise ValueError(
                f"Stored embedding length mismatch: expected {embedding_dim}, got {arr.shape[0]}"
            )
        return arr.copy()

    def upsert_template(
        self,
        user_id: str,
        tenant_id: str,
        protected_template: np.ndarray,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        transform_version: str = "v1",
    ) -> None:
        embedding_dim = int(protected_template.shape[0])
        blob = self._serialize_embedding(protected_template)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO templates (
                    user_id,
                    tenant_id,
                    protected_template,
                    embedding_dim,
                    model_name,
                    transform_version,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, tenant_id)
                DO UPDATE SET
                    protected_template=excluded.protected_template,
                    embedding_dim=excluded.embedding_dim,
                    model_name=excluded.model_name,
                    transform_version=excluded.transform_version,
                    created_at=excluded.created_at
                """,
                (
                    user_id,
                    tenant_id,
                    blob,
                    embedding_dim,
                    model_name,
                    transform_version,
                    created_at,
                ),
            )
            conn.commit()

    def get_template(self, user_id: str, tenant_id: str) -> Optional[StoredTemplate]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    user_id,
                    tenant_id,
                    protected_template,
                    embedding_dim,
                    model_name,
                    transform_version,
                    created_at
                FROM templates
                WHERE user_id = ? AND tenant_id = ?
                """,
                (user_id, tenant_id),
            ).fetchone()

        if row is None:
            return None

        protected_template = self._deserialize_embedding(row[2], row[3])

        return StoredTemplate(
            user_id=row[0],
            tenant_id=row[1],
            protected_template=protected_template,
            embedding_dim=row[3],
            model_name=row[4],
            transform_version=row[5],
            created_at=row[6],
        )