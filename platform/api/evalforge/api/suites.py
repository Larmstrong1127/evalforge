from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.api.params import parse_uuid_or_404
from evalforge.db.models import Prompt, PromptVersion, Suite
from evalforge.db.session import get_session
from evalforge.schemas.prompts import PromptVersionResponse
from evalforge.schemas.suites import PromptCreate, SuiteCreate, SuiteResponse

router = APIRouter(tags=["suites"])


@router.post("/suites", response_model=SuiteResponse, status_code=status.HTTP_201_CREATED)
async def create_suite(
    body: SuiteCreate, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> SuiteResponse:
    suite = Suite(name=body.name, description=body.description)
    session.add(suite)
    for p in body.prompts:
        prompt = Prompt(suite=suite)
        session.add(prompt)
        session.add(
            PromptVersion(
                prompt=prompt,
                version_number=1,
                input_text=p.input_text,
                expected_output=p.expected_output,
            )
        )
    await session.flush()
    return SuiteResponse(
        id=suite.id,
        name=suite.name,
        description=suite.description,
        prompt_count=len(body.prompts),
    )


@router.get("/suites", response_model=list[SuiteResponse])
async def list_suites(
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> list[SuiteResponse]:
    rows = await session.execute(
        select(Suite, func.count(Prompt.id))
        .outerjoin(Prompt, Prompt.suite_id == Suite.id)
        .group_by(Suite.id)
    )
    return [
        SuiteResponse(id=s.id, name=s.name, description=s.description, prompt_count=count)
        for s, count in rows.all()
    ]


@router.post(
    "/suites/{suite_id}/prompts",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_prompt(
    suite_id: str, body: PromptCreate, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> PromptVersionResponse:
    suite_uuid = parse_uuid_or_404(suite_id, "suite")
    suite = await session.get(Suite, suite_uuid)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id} not found")
    prompt = Prompt(suite=suite)
    session.add(prompt)
    version = PromptVersion(
        prompt=prompt,
        version_number=1,
        input_text=body.input_text,
        expected_output=body.expected_output,
    )
    session.add(version)
    await session.flush()
    return PromptVersionResponse(prompt_id=prompt.id, version_number=1)
