"""Сжатие embedding с vector(384) до vector(32) с пересчётом значений.

Признаков в build_embedding() всего 18 — остальные 366 позиций были нулями и
только раздували HNSW-индекс. Существующие строки пересчитываются из raw_payload,
поэтому поиск похожих продолжает работать после миграции.

Revision ID: 003
Revises: 002
Create Date: 2026-08-07
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

NEW_DIM = 32
OLD_DIM = 384


def _recompute_embeddings(dim: int) -> None:
    """Пересобирает embedding у всех строк из сохранённого raw_payload."""
    from src.db.embedding import build_embedding
    from src.models.candidate import Candidate

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT candidate_id, raw_payload FROM candidates WHERE raw_payload IS NOT NULL")
    ).fetchall()

    for candidate_id, raw_payload in rows:
        try:
            embedding = build_embedding(Candidate(**raw_payload))
        except Exception:
            # Payload не разбирается (схема менялась) — оставляем NULL,
            # строка просто не будет находиться поиском похожих.
            continue
        conn.execute(
            sa.text("UPDATE candidates SET embedding = :emb WHERE candidate_id = :cid"),
            {"emb": str(embedding), "cid": candidate_id},
        )


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_candidates_embedding")
    # Старые значения несовместимы по размерности — обнуляем и пересчитываем.
    op.execute(f"ALTER TABLE candidates ALTER COLUMN embedding TYPE vector({NEW_DIM}) USING NULL")
    _recompute_embeddings(NEW_DIM)
    op.execute(
        "CREATE INDEX ix_candidates_embedding ON candidates "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_candidates_embedding")
    op.execute(f"ALTER TABLE candidates ALTER COLUMN embedding TYPE vector({OLD_DIM}) USING NULL")
    op.execute(
        "CREATE INDEX ix_candidates_embedding ON candidates "
        "USING hnsw (embedding vector_cosine_ops)"
    )
