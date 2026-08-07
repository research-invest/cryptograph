"""
SQLAlchemy engine + session factory.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://btc_user:btc_pass@localhost:5432/btc_graph")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
# expire_on_commit=False: get_session() коммитит и закрывает сессию, а роуты
# читают поля ORM-объектов уже после выхода из блока with. С expire=True это
# приводило бы к DetachedInstanceError.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
