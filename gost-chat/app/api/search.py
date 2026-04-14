import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.retriever import EmptyQueryError, IndexLoadError, IndexNotFoundError, Retriever

logger = logging.getLogger(__name__)

router = APIRouter()


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    retriever = get_retriever(request)

    try:
        results, index_summary = retriever.search(payload.query, top_k=payload.top_k)
    except EmptyQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except IndexNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except IndexLoadError as exc:
        logger.exception("Failed to load retrieval index.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while searching retrieval index.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected backend error while searching the retrieval index.",
        ) from exc

    return SearchResponse(
        query=payload.query.strip(),
        top_k=payload.top_k,
        results=[SearchResult(**result.__dict__) for result in results],
        total_results=len(results),
        index_summary=index_summary,
    )

