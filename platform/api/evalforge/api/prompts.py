import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

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
    try:
        prompt_uuid = uuid.UUID(prompt_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"prompt {prompt_id} not found") from exc
    prompt = await session.get(Prompt, prompt_uuid)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"prompt {prompt_id} not found")
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
