from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.api.params import parse_uuid_or_404
from evalforge.db.models import Prompt, PromptVersion
from evalforge.db.session import get_session
from evalforge.schemas.prompts import PromptVersionResponse
from evalforge.schemas.suites import PromptCreate

router = APIRouter(tags=["prompts"])


@router.post(
    "/prompts/{prompt_id}/versions",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_prompt_version(
    prompt_id: str, body: PromptCreate, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> PromptVersionResponse:
    prompt_uuid = parse_uuid_or_404(prompt_id, "prompt")
    prompt = await session.get(Prompt, prompt_uuid)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"prompt {prompt_id} not found")
    # This read-then-write is not itself race-safe under concurrent requests
    # for the same prompt (TOCTOU on the max() query) — acceptable given
    # ADR-001's single-process/no-distributed-writers model. The real safety
    # net is PromptVersion's (prompt_id, version_number) UniqueConstraint,
    # which would raise IntegrityError on a genuine race rather than
    # silently double-assigning a version number.
    max_version = (
        await session.execute(
            select(func.max(PromptVersion.version_number)).where(
                PromptVersion.prompt_id == prompt_uuid
            )
        )
    ).scalar_one()
    next_version = (max_version or 0) + 1
    version = PromptVersion(
        prompt=prompt,
        version_number=next_version,
        input_text=body.input_text,
        expected_output=body.expected_output,
    )
    session.add(version)
    await session.flush()
    return PromptVersionResponse(prompt_id=prompt.id, version_number=next_version)
