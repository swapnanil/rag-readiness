from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from agent.models import AuditSession, DataDescription, RAGArchitecture, RefinementRecord

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./rag_sessions.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class AuditSessionRow(Base):
    __tablename__ = "audit_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    data_description_json = Column(Text, nullable=False)
    architecture_json = Column(Text, nullable=False)
    refinement_count = Column(Integer, default=0)


class RefinementHistoryRow(Base):
    __tablename__ = "refinement_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    feedback = Column(Text, nullable=False)
    constraint_changes_json = Column(Text, nullable=False)
    previous_architecture_json = Column(Text, nullable=False)
    refined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


init_db()


def create_session(data: DataDescription, architecture: RAGArchitecture) -> str:
    session_id = str(uuid.uuid4())
    with SessionLocal() as db:
        row = AuditSessionRow(
            id=session_id,
            data_description_json=data.model_dump_json(),
            architecture_json=architecture.model_dump_json(),
        )
        db.add(row)
        db.commit()
    return session_id


def get_session(session_id: str) -> AuditSession | None:
    with SessionLocal() as db:
        row = db.query(AuditSessionRow).filter(AuditSessionRow.id == session_id).first()
        if row is None:
            return None
        history = (
            db.query(RefinementHistoryRow)
            .filter(RefinementHistoryRow.session_id == session_id)
            .order_by(RefinementHistoryRow.refined_at)
            .all()
        )
        refinement_records = [
            RefinementRecord(
                feedback=h.feedback,
                constraint_changes=json.loads(h.constraint_changes_json),
                previous_architecture=RAGArchitecture(**json.loads(h.previous_architecture_json)),
                refined_at=h.refined_at,
            )
            for h in history
        ]
        return AuditSession(
            id=row.id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            data_description=DataDescription(**json.loads(row.data_description_json)),
            architecture=RAGArchitecture(**json.loads(row.architecture_json)),
            refinement_count=row.refinement_count,
            refinement_history=refinement_records,
        )


def list_sessions() -> list[dict]:
    with SessionLocal() as db:
        rows = db.query(AuditSessionRow).order_by(AuditSessionRow.updated_at.desc()).all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                "refinement_count": row.refinement_count,
                "use_case": json.loads(row.data_description_json).get("use_case", ""),
            }
            for row in rows
        ]


def update_session(session_id: str, new_architecture: RAGArchitecture, feedback: str, constraint_changes: dict) -> None:
    with SessionLocal() as db:
        row = db.query(AuditSessionRow).filter(AuditSessionRow.id == session_id).first()
        if row is None:
            raise ValueError(f"Session {session_id} not found")

        history_row = RefinementHistoryRow(
            session_id=session_id,
            feedback=feedback,
            constraint_changes_json=json.dumps(constraint_changes),
            previous_architecture_json=row.architecture_json,
        )
        db.add(history_row)

        row.architecture_json = new_architecture.model_dump_json()
        row.refinement_count = (row.refinement_count or 0) + 1
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
