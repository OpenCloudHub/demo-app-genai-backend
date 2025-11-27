"""OpenCloudHub RAG API with prompt version management."""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from src._config import CONFIG
from src._logging import get_logger
from src.api.schemas import (
    APIStatus,
    ErrorResponse,
    HealthResponse,
    PromptInfo,
    QueryRequest,
    QueryResponse,
    ReloadPromptRequest,
    ReloadPromptResponse,
    RootResponse,
)
from src.rag.chain import RAGChain

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain on startup."""
    logger.info("🚀 Starting OpenCloudHub RAG API")
    app.state.status = APIStatus.LOADING
    app.state.chain = None
    app.state.prompt_name = CONFIG.prompt_name
    app.state.top_k = CONFIG.db_top_k
    app.state.prompt_version = None
    app.state.start_time = datetime.now(timezone.utc)

    try:
        logger.info(f"📦 Loading RAG chain with prompt v{app.state.prompt_version}")

        # Try to load production prompt version if not specified
        if app.state.prompt_version is None:
            try:
                prod_prompt = mlflow.genai.load_prompt(
                    f"prompts:/{app.state.prompt_name}@champion"
                )
                app.state.prompt_version = prod_prompt.version
                logger.info(
                    f"ℹ️ Loaded production prompt version: v{app.state.prompt_version}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load production prompt version: {e}")
                logger.warning("Service will start unhealthy until prompt is available")
                app.state.prompt_version = None
                app.state.status = APIStatus.UNHEALTHY
                app.state.chain = None
                yield  # Start service in unhealthy state
                return

        # Try to initialize RAG chain
        try:
            app.state.chain = RAGChain(
                db_connection_string=CONFIG.db_connection_string,
                table_name=CONFIG.db_table_name,
                embedding_model=CONFIG.embedding_model,
                llm_base_url=CONFIG.llm_base_url,
                llm_model=CONFIG.llm_model,
                api_key=CONFIG.api_key,
                prompt_name=app.state.prompt_name,
                prompt_version=app.state.prompt_version,
                top_k=app.state.top_k,
            )

            # Verify chain is properly initialized
            if app.state.chain is None:
                raise ValueError("RAG chain initialization returned None")

            app.state.status = APIStatus.HEALTHY
            logger.success("✅ RAG chain loaded successfully")

        except Exception as e:
            logger.error(f"⚠️ Failed to initialize RAG chain: {e}", exc_info=True)
            logger.warning(
                "Service will start unhealthy. Use /admin/reload-prompt to retry."
            )
            app.state.status = APIStatus.UNHEALTHY
            app.state.chain = None

    except Exception as e:
        logger.error(f"⚠️ Unexpected error during startup: {e}", exc_info=True)
        logger.warning("Service will start unhealthy")
        app.state.status = APIStatus.UNHEALTHY
        app.state.chain = None

    yield

    logger.info("👋 Shutting down OpenCloudHub RAG API")
    app.state.chain = None


description = """
**Demo RAG System for OpenCloudHub MLOps Platform Documentation**

This demo Retrieval-Augmented Generation (RAG) system demonstrates modern MLOps
practices applied to Generative AI workloads, showcasing key differences from traditional
ML/DL workflows.

## 🔍 How it Works

1. **Semantic Search**: User questions are embedded and matched against a vector database
    (PgVector) containing OpenCloudHub repository documentation
2. **Context Retrieval**: Top-K most relevant documentation chunks are retrieved
3. **LLM Generation**: Retrieved context + user question are sent to our self-hosted
    Qwen 2.5 (0.5B) model for answer generation
4. **Streaming Response**: Answers stream back token-by-token for responsive UX

## 🏗️ Infrastructure

- **Self-Hosted LLM**: Qwen 2.5-0.5B-Instruct served via Ray Serve on our Kubernetes cluster
- **Vector Store**: PostgreSQL with pgvector extension for semantic search
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) for document/query encoding
- **Orchestration**: LangChain for RAG pipeline composition

## 🔄 GenAI MLOps Workflow

Unlike traditional ML/DL where you version **models**, in GenAI you need to take care of versioning **prompts** as a new artifact type. This system showcases a full MLOps workflow around prompt versioning, evaluation, and deployment.

### Our MLOps Stack for GenAI

1. **Prompt Registry** (MLflow):
    - Store and version prompt templates like model artifacts
    - Track prompt lineage and metadata
    - Promote best-performing prompts to `@champion` alias

2. **Automated Evaluation**:
    - DVC for versioned evaluation datasets
    - MLflow Evaluate with custom LLM-as-judge metrics
    - A/B test multiple prompt versions automatically
    - GitHub Actions CI/CD for evaluation on every prompt change

3. **Tracing & Observability**:
    - Log every LLM call with input/output/latency
    - Track retrieval quality and context relevance
    - Monitor token usage and costs
    - Trace full RAG pipeline execution

4. **GitOps Deployment**:
    - Prompt versions stored in MLflow registry
    - ArgoCD + Image Updater for continuous deployment
    - Dynamic prompt reloading via `/admin/reload-prompt` endpoint
    - No service restart needed to test new prompts

## 🎯 Key GenAI Aspects Demonstrated

- **Prompt Versioning**: Treat prompts as first-class artifacts with semantic versioning
- **Evaluation-Driven Development**: Automated prompt testing on curated question sets
- **Dynamic Prompt Updates**: Reload prompts in production without container restarts
- **LLM Call Tracing**: Full observability into every generation request
- **Self-Hosted Models**: Complete control over LLM infrastructure and data privacy
- **Streaming Responses**: Real-time token generation for better UX

## 📊 Endpoints

- `POST /query` - Ask questions with optional streaming
- `GET /prompt` - View current prompt template and version
- `POST /admin/reload-prompt` - Hot-reload new prompt versions
- `GET /health` - Service health and chain status

Built with ❤️ by OpenCloudHub to showcase MLOps for GenAI workloads.
"""


app = FastAPI(
    title="🤖 OpenCloudHub RAG API",
    description=description,
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": {"error_type": type(exc).__name__, "message": str(exc)}},
    )


@app.get(
    "/",
    response_model=RootResponse,
    summary="Root endpoint",
    responses={
        200: {"description": "Service information"},
        503: {"description": "Service not healthy"},
    },
)
async def root():
    """Root endpoint with basic service information."""
    return RootResponse(
        service="OpenCloudHub RAG API",
        version="1.0.0",
        status=app.state.status.value,
        docs="/docs",
        health="/health",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is not ready or unhealthy"},
    },
)
async def health():
    """Health check endpoint."""
    uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()

    response = HealthResponse(
        status=app.state.status,
        chain_loaded=app.state.chain is not None,
        prompt_name=app.state.prompt_name,
        prompt_version=app.state.prompt_version,
        uptime_seconds=int(uptime),
    )

    # Return 503 if not healthy
    if app.state.status != APIStatus.HEALTHY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@app.get(
    "/prompt",
    response_model=PromptInfo,
    summary="Get Current Prompt",
    responses={
        200: {"description": "Current prompt information"},
        503: {"description": "Chain not loaded", "model": ErrorResponse},
    },
)
async def get_prompt():
    """Get the current prompt template information."""
    if app.state.chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not loaded",
        )

    try:
        # Get the prompt template text from the chain
        prompt_template = app.state.chain.prompt_template.messages[0].prompt.template

        return PromptInfo(
            prompt_name=app.state.prompt_name,
            prompt_version=app.state.prompt_version,
            prompt_template=prompt_template,
        )
    except Exception as e:
        logger.error(f"Failed to get prompt info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prompt info: {str(e)}",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a Question",
    responses={
        200: {"description": "Successful query"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Chain not loaded", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def query(request: QueryRequest):
    """
    Ask a question about the OpenCloudHub MLOps platform.

    The system will retrieve relevant context from the documentation and
    generate an answer using the configured LLM and current prompt template.
    """
    # Check if chain is loaded
    if app.state.chain is None or app.state.prompt_version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not loaded. Service may still be starting up.",
        )

    start_time = time.time()

    try:
        logger.info(f"Query: {request.question[:100]}... (stream={request.stream})")

        # Streaming response
        if request.stream:

            async def generate():
                """
                Async generator for streaming.

                CRITICAL: Must use asyncio.to_thread to run sync generator in thread pool
                to avoid blocking the event loop.
                """
                try:
                    chunk_count = 0

                    # Run sync stream in thread pool to avoid blocking
                    def sync_stream():
                        """Synchronous generator wrapper."""
                        for chunk in app.state.chain.stream(request.question):
                            yield chunk

                    # Convert sync generator to async
                    loop = asyncio.get_event_loop()

                    # Get the generator
                    gen = sync_stream()

                    # Stream chunks
                    while True:
                        try:
                            # Get next chunk in thread pool (non-blocking)
                            chunk = await loop.run_in_executor(
                                None, lambda: next(gen, None)
                            )

                            if chunk is None:
                                break

                            if chunk:  # Only send non-empty chunks
                                chunk_count += 1
                                logger.debug(
                                    f"Streaming chunk {chunk_count}: {repr(chunk)[:50]}"
                                )
                                yield f"data: {chunk}\n\n"

                        except StopIteration:
                            break

                    if chunk_count == 0:
                        logger.warning("No chunks generated")
                        yield "data: [ERROR] No response generated\n\n"
                    else:
                        processing_time = (time.time() - start_time) * 1000
                        logger.info(
                            f"Stream complete: {chunk_count} chunks in {processing_time:.0f}ms"
                        )
                        yield "data: [DONE]\n\n"

                except Exception as e:
                    logger.error(f"Streaming error: {e}", exc_info=True)
                    yield f"data: [ERROR] {str(e)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming response
        answer = app.state.chain.invoke(request.question)
        processing_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer,
            prompt_name=app.state.prompt_name,
            prompt_version=app.state.prompt_version,
            session_id=request.session_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@app.post(
    "/admin/reload-prompt",
    response_model=ReloadPromptResponse,
    summary="Reload Prompt",
    responses={
        200: {"description": "Prompt reloaded successfully"},
        500: {"description": "Reload failed", "model": ErrorResponse},
    },
)
async def reload_prompt(request: ReloadPromptRequest):
    """
    Reload the RAG chain with a different prompt version.

    This allows testing different prompt templates without restarting the service.
    """
    app.state.status = APIStatus.LOADING

    try:
        logger.info(f"🔄 Reloading RAG chain with prompt v{request.prompt_version}...")

        # Recreate the chain with new prompt version
        app.state.chain = RAGChain(
            db_connection_string=CONFIG.db_connection_string,
            table_name=CONFIG.db_table_name,
            embedding_model=CONFIG.embedding_model,
            llm_base_url=CONFIG.llm_base_url,
            llm_model=CONFIG.llm_model,
            api_key=CONFIG.api_key,
            prompt_name=app.state.prompt_name,
            prompt_version=request.prompt_version,
            top_k=request.top_k,
        )

        app.state.prompt_version = request.prompt_version
        app.state.status = APIStatus.HEALTHY
        logger.success(f"✅ Reloaded with prompt v{request.prompt_version}")

        return ReloadPromptResponse(
            status="reloaded",
            prompt_name=app.state.prompt_name,
            prompt_version=request.prompt_version,
            top_k=request.top_k,
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        app.state.status = APIStatus.UNHEALTHY
        logger.error(f"❌ Failed to reload prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload prompt: {str(e)}",
        )
