import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.orchestration.chat_orchestrator import EmptyChatMessageError
from app.orchestration.chat_store import ChatSessionNotFoundError
from app.schemas.chat import (
    ChatMessagePayload,
    ChatRequest,
    ChatResponse,
    ChatSessionDetailResponse,
    ChatSessionResponse,
    CreateChatSessionRequest,
    HealthResponse,
    ModelResponse,
    SessionChatMessageRequest,
    SessionChatMessageResponse,
)
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


@router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(payload: CreateChatSessionRequest, request: Request) -> ChatSessionResponse:
    session = request.app.state.chat_store.create_session(title=payload.title or "New chat")
    return ChatSessionResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat_session(session_id: str, request: Request) -> ChatSessionDetailResponse:
    try:
        session = request.app.state.chat_store.get_session(session_id)
        messages = request.app.state.chat_store.list_messages(session_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.") from exc
    return ChatSessionDetailResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[ChatMessagePayload(**message.__dict__) for message in messages],
    )


@router.post("/chat/sessions/{session_id}/messages", response_model=SessionChatMessageResponse)
async def send_session_message(
    session_id: str,
    payload: SessionChatMessageRequest,
    request: Request,
) -> SessionChatMessageResponse:
    try:
        response = await request.app.state.chat_orchestrator.send_message(session_id, payload.message, top_k=payload.top_k)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.") from exc
    except EmptyChatMessageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return SessionChatMessageResponse(
        session_id=response.session_id,
        message_id=response.message_id,
        answer=response.answer,
        model=response.model,
        citations=list(getattr(response, "citations", []) or []),
        attachments=list(getattr(response, "attachments", []) or []),
        tool_events=list(getattr(response, "tool_events", []) or []),
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
