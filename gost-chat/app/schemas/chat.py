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


class ErrorResponse(BaseModel):
    detail: str
