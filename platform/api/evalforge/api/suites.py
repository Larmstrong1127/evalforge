from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import Prompt, PromptVersion, Suite
from evalforge.db.session import get_session
from evalforge.schemas.suites import SuiteCreate, SuiteResponse

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
    suites = (await session.execute(select(Suite))).scalars().all()
    result = []
    for s in suites:
        count = (
            await session.execute(
                select(func.count(Prompt.id)).where(Prompt.suite_id == s.id)
            )
        ).scalar_one()
        result.append(
            SuiteResponse(id=s.id, name=s.name, description=s.description, prompt_count=count)
        )
    return result
