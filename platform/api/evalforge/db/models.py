"""ORM models.

Design invariants:
- Prompt text is immutable: edits append a PromptVersion row; results FK the
  exact version, so cross-run diffs stay valid forever.
- Generation (Result) and judgment (JudgeEvaluation) are separate tables: one
  output can be scored by N judges without duplicating rows.
"""
import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampedBase(Base):
    __abstract__ = True
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultStatus(enum.Enum):
    OK = "ok"
    FAILED = "failed"


class Suite(TimestampedBase):
    __tablename__ = "suites"
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    prompts: Mapped[list["Prompt"]] = relationship(back_populates="suite")


class Prompt(TimestampedBase):
    __tablename__ = "prompts"
    suite_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("suites.id", ondelete="CASCADE"))
    suite: Mapped[Suite] = relationship(back_populates="prompts")
    versions: Mapped[list["PromptVersion"]] = relationship(back_populates="prompt")


class PromptVersion(TimestampedBase):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_id", "version_number"),)
    prompt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompts.id", ondelete="CASCADE"))
    version_number: Mapped[int] = mapped_column(Integer)
    input_text: Mapped[str] = mapped_column(Text)
    expected_output: Mapped[str | None] = mapped_column(Text, default=None)
    prompt: Mapped[Prompt] = relationship(back_populates="versions")


class CandidateModel(TimestampedBase):
    __tablename__ = "candidate_models"
    name: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))


class Run(TimestampedBase):
    __tablename__ = "runs"
    suite_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("suites.id"))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.QUEUED)
    concurrency_limit: Mapped[int] = mapped_column(Integer, default=3)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    suite: Mapped[Suite] = relationship()
    results: Mapped[list["Result"]] = relationship(back_populates="run")


class Result(TimestampedBase):
    __tablename__ = "results"
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    prompt_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompt_versions.id"))
    candidate_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidate_models.id"))
    status: Mapped[ResultStatus] = mapped_column(Enum(ResultStatus))
    generated_text: Mapped[str] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    latency_ms: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    run: Mapped[Run] = relationship(back_populates="results")
    prompt_version: Mapped[PromptVersion] = relationship()
    candidate_model: Mapped[CandidateModel] = relationship()
    judge_evaluations: Mapped[list["JudgeEvaluation"]] = relationship(back_populates="result")


class JudgeEvaluation(TimestampedBase):
    __tablename__ = "judge_evaluations"
    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    judge_name: Mapped[str] = mapped_column(String(100))
    score: Mapped[float] = mapped_column(Float)
    justification: Mapped[str | None] = mapped_column(Text, default=None)
    result: Mapped[Result] = relationship(back_populates="judge_evaluations")


class HumanRating(TimestampedBase):
    __tablename__ = "human_ratings"
    prompt_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="CASCADE")
    )
    result_a_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    result_b_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    chosen_result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("results.id", ondelete="CASCADE"), default=None
    )
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    rater_session: Mapped[str | None] = mapped_column(String(100), default=None)
