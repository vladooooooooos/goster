from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelResponse(BaseModel):
    provider: str
    model: str
    available: bool
    ollama_available: bool


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to send to the assistant.")


class ChatResponse(BaseModel):
    response: str
    model: str


class CreateChatSessionRequest(BaseModel):
    title: str | None = Field(None, description="Optional chat title.")


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessagePayload(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


class ChatSessionDetailResponse(ChatSessionResponse):
    messages: list[ChatMessagePayload] = Field(default_factory=list)


class SessionChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int = Field(12, ge=1, le=50)


class SessionChatMessageResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    model: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    tool_events: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
