import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse, ModelResponse
from app.services.chat_service import ChatService, EmptyMessageError
from app.services.llm_service import LlmServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    logger.info("Health check requested.")
    return HealthResponse(status="ok", service=request.app.state.settings.app_name)


@router.get("/models", response_model=ModelResponse)
async def models(request: Request) -> ModelResponse:
    chat_service = get_chat_service(request)
    available = await chat_service.is_model_available()
    return ModelResponse(
        provider=chat_service.provider,
        model=chat_service.model,
        available=available,
        ollama_available=available,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    chat_service = get_chat_service(request)

    try:
        response = await chat_service.send_message(payload.message)
        return ChatResponse(response=response, model=chat_service.model)
    except EmptyMessageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LlmServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing chat request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected backend error while processing the chat request.",
        ) from exc
