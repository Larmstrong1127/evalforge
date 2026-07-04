"""Shared path-param helpers for API routes."""
import uuid

from fastapi import HTTPException


def parse_uuid_or_404(id_str: str, resource_name: str) -> uuid.UUID:
    """Parse a path-param UUID string, raising a 404 (not 500) on garbage
    input. SQLAlchemy's generic Uuid type is emulated on SQLite rather than
    backed by a native column type, so an unparsed string reaching the query
    layer risks a driver-level crash instead of a clean "not found" — this
    project hit exactly that bug class in Phase 1 (CLI `results <run_id>`).
    """
    try:
        return uuid.UUID(id_str)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"{resource_name} {id_str} not found") from exc
