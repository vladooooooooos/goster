import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.ask import AskCitation, AskRequest, AskResponse, AskRetrievedChunk
from app.services.llm_service import LlmServiceError
from app.services.rag_service import RagService
from app.services.retriever import EmptyQueryError, IndexLoadError, IndexNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    rag_service = get_rag_service(request)

    try:
        result = await rag_service.answer_question(payload.query, top_k=payload.top_k)
    except EmptyQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except IndexNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except IndexLoadError as exc:
        logger.exception("Failed to load retrieval index for RAG answer.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except LlmServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing RAG answer request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected backend error while processing the RAG answer request.",
        ) from exc

    return AskResponse(
        query=result.query,
        answer=result.answer,
        citations=[AskCitation(**citation.__dict__) for citation in result.citations],
        retrieved_results_count=result.retrieved_results_count,
        retrieval_used=result.retrieval_used,
        retrieved_chunks=[AskRetrievedChunk(**chunk.__dict__) for chunk in result.retrieved_chunks],
        retrieval_info=result.retrieval_info,
    )
