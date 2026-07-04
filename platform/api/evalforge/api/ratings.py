from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import HumanRating
from evalforge.db.session import get_session
from evalforge.schemas.ratings import RatingCreate, RatingResponse

router = APIRouter(tags=["ratings"])


@router.post("/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
async def create_rating(
    body: RatingCreate, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> RatingResponse:
    rating = HumanRating(
        prompt_version_id=body.prompt_version_id,
        result_a_id=body.result_a_id,
        result_b_id=body.result_b_id,
        chosen_result_id=body.chosen_result_id,
        skipped=body.skipped,
        rater_session=body.rater_session,
    )
    session.add(rating)
    await session.flush()
    return RatingResponse(id=rating.id)
