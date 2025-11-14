# app/db_models.py
from __future__ import annotations

from datetime import datetime, time as dt_time # 👈 [1. แก้ไข] เพิ่ม import time
from typing import Generator, List, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    Time, # 👈 [2. แก้ไข] เพิ่ม import Time
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)

from .config import settings


# ---------- Base / Engine / Session ----------
class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.db_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# ---------- Models ----------
class UserType(Base):
    __tablename__ = "user_types"
    user_type_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    users: Mapped[List["User"]] = relationship(back_populates="user_type")


class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (
        UniqueConstraint("subject_name", "section", "academic_year", name="uq_subject"),
    )

    subject_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_name: Mapped[str] = mapped_column(String(255), nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(100), default=None)
    schedule: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    cover_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    academic_year: Mapped[Optional[str]] = mapped_column(String(20), default=None)

    # --- [3. แก้ไข] เพิ่ม 2 บรรทัดนี้ ---
    class_start_time: Mapped[Optional[dt_time]] = mapped_column(Time) # เช่น 09:00:00
    class_end_time: Mapped[Optional[dt_time]] = mapped_column(Time)   # เช่น 12:00:00
    # ---

    users: Mapped[List["User"]] = relationship(back_populates="subject")
    logs: Mapped[List["AttendanceLog"]] = relationship(back_populates="subject")


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_code: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(Enum("admin", "operator", "viewer", name="user_role"), nullable=False)
    user_type_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user_types.user_type_id", ondelete="SET NULL"))
    password_hash: Mapped[Optional[str]] = mapped_column(String(255))
    subject_id: Mapped[Optional[int]] = mapped_column(ForeignKey("subjects.subject_id", ondelete="SET NULL"))
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_type: Mapped[Optional["UserType"]] = relationship(back_populates="users")
    subject: Mapped[Optional["Subject"]] = relationship(back_populates="users")
    faces: Mapped[List["UserFace"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    logs: Mapped[List["AttendanceLog"]] = relationship(back_populates="user")


class UserFace(Base):
    __tablename__ = "user_faces"
    __table_args__ = (UniqueConstraint("file_path", name="uq_user_faces_path"),)
    face_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user: Mapped["User"] = relationship(back_populates="faces")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    subject_id: Mapped[Optional[int]] = mapped_column(ForeignKey("subjects.subject_id", ondelete="SET NULL"),
                                                      index=True)
    action: Mapped[str] = mapped_column(Enum("enter", "exit", name="attendance_action"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    flags: Mapped[Optional[dict]] = mapped_column(JSON)
    user: Mapped[Optional["User"]] = relationship(back_populates="logs")
    subject: Mapped[Optional["Subject"]] = relationship(back_populates="logs")
    alerts: Mapped[List["Alert"]] = relationship(back_populates="log", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[int] = mapped_column(ForeignKey("attendance_logs.log_id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(
        Enum("short_stay", "multiple_faces", "low_confidence", "spoofing", name="alert_type"), nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Enum("open", "approved", "rejected", name="alert_status"), default="open",
                                        nullable=False)
    log: Mapped["AttendanceLog"] = relationship(back_populates="alerts")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()