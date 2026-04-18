from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GOST Chat"
    llm_provider: str = "polza"
    llm_base_url: str = "https://polza.ai/api/v1"
    llm_model: str = "google/gemma-4-31b-it"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    llm_request_timeout_seconds: float = 120.0
    llm_api_key_env_var: str = "POLZA_API_KEY"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    ollama_request_timeout_seconds: float = 120.0
    indexer_output_dir: Path = Path("data")
    log_level: str = "INFO"
    retrieval_backend: str = "auto"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_https: bool = False
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 5.0
    qdrant_collection_name: str = "gost_blocks"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 4
    embedding_normalize_embeddings: bool = True
    reranker_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    reranker_batch_size: int = 2
    reranker_max_length: int = 512
    reranker_top_k: int = 30
    reranker_top_n: int = 12
    reranker_use_fp16_if_available: bool = True
    context_min_blocks: int = 2
    context_soft_target_blocks: int = 5
    context_simple_target_blocks: int = 5
    context_max_blocks: int = 12
    context_max_chars: int = 18000
    context_adaptive_score_threshold: float = 0.12
    visual_enable_decision: bool = True
    visual_crops_dir: Path = Path("data/crops")
    visual_crop_dpi: int = 160
    visual_max_crops_per_answer: int = 4
    visual_vision_enabled: bool = True
    visual_page_render_dpi: int = 120
    visual_candidate_limit: int = 8
    chat_store_path: Path = Path("data/chat_sessions.json")
    chat_history_limit: int = 12
    agent_max_tool_loops: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GOST_CHAT_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

