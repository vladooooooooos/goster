import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.ask import router as ask_router
from app.api.chat import router as chat_router
from app.api.search import router as search_router
from app.config import get_settings
from app.services.chat_service import ChatService
from app.services.context_builder import ContextBuilder
from app.services.local_embedding_service import LocalEmbeddingService, LocalEmbeddingSettings
from app.services.llm_service import create_llm_service
from app.services.qdrant_retriever import QdrantRetriever
from app.services.rag_service import RagService
from app.services.reranker_service import LocalRerankerService, RerankerSettings
from app.services.retrieval_pipeline import RetrievalPipeline
from app.services.retriever import Retriever

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.state.settings = settings
app.state.llm_service = create_llm_service(settings)
app.state.chat_service = ChatService(app.state.llm_service)
app.state.retriever = Retriever(settings.indexer_output_dir)
app.state.embedding_service = LocalEmbeddingService(
    LocalEmbeddingSettings(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize_embeddings,
    )
)
app.state.qdrant_retriever = QdrantRetriever(
    qdrant_url=settings.qdrant_url,
    qdrant_host=settings.qdrant_host,
    qdrant_port=settings.qdrant_port,
    qdrant_https=settings.qdrant_https,
    qdrant_api_key=settings.qdrant_api_key,
    qdrant_timeout_seconds=settings.qdrant_timeout_seconds,
    collection_name=settings.qdrant_collection_name,
    embedding_service=app.state.embedding_service,
)
app.state.reranker = LocalRerankerService(
    RerankerSettings(
        enabled=settings.reranker_enabled,
        model_name=settings.reranker_model_name,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        use_fp16_if_available=settings.reranker_use_fp16_if_available,
    )
)
app.state.retrieval_pipeline = RetrievalPipeline(
    retriever=app.state.retriever,
    settings=settings,
    qdrant_retriever=app.state.qdrant_retriever,
    reranker=app.state.reranker,
)
app.state.context_builder = ContextBuilder()
app.state.rag_service = RagService(
    llm_service=app.state.llm_service,
    retrieval_pipeline=app.state.retrieval_pipeline,
    context_builder=app.state.context_builder,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info("Request started: %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request failed: %s %s", request.method, request.url.path)
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed: %s %s %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name, "model": settings.llm_model},
    )


app.include_router(chat_router)
app.include_router(search_router)
app.include_router(ask_router)
