#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - FastAPI RAG Service (FIXED & PRODUCTION-READY)
# ============================================================================
# Purpose: Main FastAPI application with streaming RAG capabilities
# Guide Reference: Section 4.1 (Complete main.py Implementation)
# Last Updated: 2025-10-18
# CRITICAL FIXES:
#   - Retry logic with exponential backoff (3 attempts, max 10s wait)
#   - Fixed import paths with sys.path setup
#   - Error recovery on LLM/embeddings/vectorstore failures
#   - Rate limiting and exception handling
#   - Memory checks before model loading
# ============================================================================

import os
import sys
import time
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

# CRITICAL FIX: Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# System monitoring
import psutil

# Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configuration and dependencies
from config_loader import load_config, get_config_value
from logging_config import setup_logging, get_logger, PerformanceLogger
from dependencies import get_llm, get_embeddings, get_vectorstore, check_available_memory
from metrics import (
    start_metrics_server,
    record_request,
    record_error,
    record_tokens_generated,
    record_query_processed,
    update_token_rate,
    record_rag_retrieval,
    MetricsTimer,
    response_latency_ms
)

# Setup logging
setup_logging()
logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)

# Load configuration
CONFIG = load_config()

# ============================================================================
# GLOBAL STATE (Lazy Loading with Retry)
# ============================================================================

llm = None
embeddings = None
vectorstore = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Query request model. Guide Reference: Section 4.1 (Request Models)"""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="User query",
        examples=["What is Xoe-NovAi?"]
    )
    use_rag: bool = Field(True, description="Whether to use RAG context retrieval")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter")


class QueryResponse(BaseModel):
    """Query response model. Guide Reference: Section 4.1 (Response Models)"""
    response: str = Field(..., description="Generated response")
    sources: List[str] = Field(default_factory=list, description="RAG sources used")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")
    duration_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    token_rate_tps: Optional[float] = Field(None, description="Tokens per second")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status: healthy, degraded, or partial")
    version: str = Field(..., description="Stack version")
    memory_gb: float = Field(..., description="Current memory usage in GB")
    vectorstore_loaded: bool = Field(..., description="Whether vectorstore is available")
    components: Dict[str, bool] = Field(..., description="Component status map")

# ============================================================================
# RETRY DECORATORS (CRITICAL FIX)
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, OSError, MemoryError)),
    reraise=True
)
def _get_llm_with_retry() -> Any:
    """Get LLM with retry logic (3 attempts, exponential backoff max 10s)."""
    global llm
    
    if llm is not None:
        return llm
    
    logger.info("Initializing LLM (attempt with retry logic)...")
    
    # Check memory before loading
    try:
        check_available_memory(required_gb=4.0)
    except MemoryError as e:
        logger.error(f"Insufficient memory for LLM: {e}")
        raise
    
    llm = get_llm()
    logger.info("LLM initialized successfully")
    return llm


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, OSError)),
    reraise=True
)
def _get_embeddings_with_retry() -> Any:
    """Get embeddings with retry logic."""
    global embeddings
    
    if embeddings is not None:
        return embeddings
    
    logger.info("Initializing embeddings...")
    embeddings = get_embeddings()
    logger.info("Embeddings initialized successfully")
    return embeddings


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((RuntimeError, OSError)),
    reraise=False  # Don't re-raise - vectorstore is optional
)
def _get_vectorstore_with_retry() -> Optional[Any]:
    """Get vectorstore with retry logic. Returns None if unavailable."""
    global vectorstore
    
    if vectorstore is not None:
        return vectorstore
    
    logger.info("Loading vectorstore...")
    
    try:
        embeddings = _get_embeddings_with_retry()
        vectorstore = get_vectorstore(embeddings)
        
        if vectorstore:
            vector_count = vectorstore.index.ntotal
            logger.info(f"Vectorstore loaded: {vector_count} vectors")
        else:
            logger.warning("Vectorstore not found - RAG disabled (run ingest_library.py)")
        
        return vectorstore
        
    except Exception as e:
        logger.warning(f"Vectorstore loading failed (RAG will be unavailable): {e}")
        return None

# ============================================================================
# CONTEXT TRUNCATION & RAG LOGIC (unchanged from artifact)
# ============================================================================

def _build_truncated_context(
    docs: List,
    per_doc_chars: int = None,
    total_chars: int = None
) -> tuple:
    """Build truncated context from documents. Guide Reference: Section 4.1"""
    if per_doc_chars is None:
        per_doc_chars = CONFIG['performance']['per_doc_chars']
    
    if total_chars is None:
        total_chars = CONFIG['performance']['total_chars']
    
    context = ""
    sources = []
    
    for doc in docs:
        doc_text = doc.page_content[:per_doc_chars]
        source = doc.metadata.get("source", "unknown")
        formatted_doc = f"\n[Source: {source}]\n{doc_text}\n"
        
        if len(context + formatted_doc) > total_chars:
            logger.debug(f"Context truncation at {len(context)} chars")
            break
        
        context += formatted_doc
        if source not in sources:
            sources.append(source)
    
    context = context[:total_chars]
    logger.debug(f"Built context: {len(context)} chars from {len(sources)} sources")
    return context, sources


def retrieve_context(
    query: str,
    top_k: int = None,
    similarity_threshold: float = None
) -> tuple:
    """Retrieve relevant documents from FAISS vectorstore."""
    if vectorstore is None:
        logger.warning("Vectorstore not initialized, skipping RAG")
        return "", []
    
    if top_k is None:
        top_k = get_config_value('rag.top_k', 5)
    
    try:
        start_time = time.time()
        docs = vectorstore.similarity_search(query, k=top_k)
        retrieval_ms = (time.time() - start_time) * 1000
        record_rag_retrieval(retrieval_ms)
        
        if not docs:
            logger.warning(f"No documents retrieved for query: {query[:50]}...")
            return "", []
        
        per_doc_chars = int(os.getenv("RAG_PER_DOC_CHARS", CONFIG['performance']['per_doc_chars']))
        total_chars = int(os.getenv("RAG_TOTAL_CHARS", CONFIG['performance']['total_chars']))
        
        context, sources = _build_truncated_context(docs, per_doc_chars, total_chars)
        
        logger.info(f"Retrieved {len(sources)} documents in {retrieval_ms:.2f}ms")
        return context, sources
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}", exc_info=True)
        record_error("rag_retrieval", "vectorstore")
        return "", []


def generate_prompt(query: str, context: str = "") -> str:
    """Generate LLM prompt with optional RAG context."""
    if context:
        return f"""Based on the following context, answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    else:
        return f"""Question: {query}

Answer:"""

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management. Guide Reference: Section 4.1"""
    # Startup
    logger.info("=" * 70)
    logger.info("Starting Xoe-NovAi RAG API v0.1.2")
    logger.info("=" * 70)
    
    try:
        start_metrics_server()
        logger.info("✓ Prometheus metrics server started on port 8002")
    except Exception as e:
        logger.warning(f"Metrics server failed to start: {e}")
    
    memory_gb = psutil.virtual_memory().used / (1024 ** 3)
    logger.info(f"Current memory usage: {memory_gb:.2f}GB")
    
    if memory_gb > CONFIG['performance']['memory_warning_threshold_gb']:
        logger.warning(f"Memory usage high: {memory_gb:.2f}GB (threshold: {CONFIG['performance']['memory_warning_threshold_gb']}GB)")
    
    logger.info("=" * 70)
    logger.info("RAG API ready for requests")
    logger.info(f"  - API: http://0.0.0.0:{CONFIG['server']['port']}")
    logger.info(f"  - Metrics: http://0.0.0.0:{get_config_value('metrics.port', 8002)}/metrics")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Xoe-NovAi RAG API",
    description="CPU-optimized RAG service with streaming support",
    version=CONFIG['metadata']['stack_version'],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# RATE LIMITING
# ============================================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

cors_origins = CONFIG['server']['cors_origins']
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for origins: {cors_origins}")

# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    record_request(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    )
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }
    )
    
    return response

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Xoe-NovAi RAG API",
        "version": CONFIG['metadata']['stack_version'],
        "codename": CONFIG['metadata']['codename'],
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "stream": "/stream (POST)",
            "docs": "/docs",
            "metrics": f"http://localhost:{get_config_value('metrics.port', 8002)}/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status."""
    memory_gb = psutil.virtual_memory().used / (1024 ** 3)
    
    components = {
        "embeddings": embeddings is not None,
        "vectorstore": vectorstore is not None,
        "llm": llm is not None,
    }
    
    # Determine status
    if memory_gb > CONFIG['performance']['memory_limit_gb']:
        status = "degraded"
    elif not components['embeddings']:
        status = "degraded"
    elif not components['vectorstore']:
        status = "partial"
    else:
        status = "healthy"
    
    return HealthResponse(
        status=status,
        version=CONFIG['metadata']['stack_version'],
        memory_gb=round(memory_gb, 2),
        vectorstore_loaded=vectorstore is not None,
        components=components
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("60/minute")
async def query_endpoint(request: Request, query_req: QueryRequest):
    """Synchronous query endpoint. Guide Reference: Section 4.1"""
    with MetricsTimer(response_latency_ms, endpoint='/query', method='POST'):
        start_time = time.time()
        
        try:
            # Get LLM with retry
            try:
                llm_instance = _get_llm_with_retry()
            except Exception as e:
                logger.error(f"LLM initialization failed after retries: {e}")
                raise HTTPException(status_code=503, detail="LLM unavailable after retries")
            
            # Retrieve context if RAG enabled
            sources = []
            context = ""
            if query_req.use_rag and vectorstore is not None:
                context, sources = retrieve_context(query_req.query)
            
            # Generate prompt
            prompt = generate_prompt(query_req.query, context)
            
            # Generate response
            gen_start = time.time()
            response = llm_instance.invoke(
                prompt,
                max_tokens=query_req.max_tokens,
                temperature=query_req.temperature,
                top_p=query_req.top_p
            )
            gen_duration = time.time() - gen_start
            
            # Calculate metrics
            tokens_approx = len(response.split())
            token_rate = tokens_approx / gen_duration if gen_duration > 0 else 0
            
            record_tokens_generated(tokens_approx)
            record_query_processed(query_req.use_rag)
            update_token_rate(token_rate)
            
            perf_logger.log_token_generation(tokens=tokens_approx, duration_s=gen_duration)
            
            total_duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Query completed: {total_duration_ms:.0f}ms, {tokens_approx} tokens, "
                f"{token_rate:.1f} tok/s, {len(sources)} sources"
            )
            
            return QueryResponse(
                response=response,
                sources=sources,
                tokens_generated=tokens_approx,
                duration_ms=round(total_duration_ms, 2),
                token_rate_tps=round(token_rate, 2)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            record_error('query_failed', 'llm')
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)[:200]}")


@app.post("/stream")
@limiter.limit("60/minute")
async def stream_endpoint(request: Request, query_req: QueryRequest):
    """Streaming query endpoint (SSE). Guide Reference: Section 4.1"""
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE stream."""
        try:
            # Get LLM with retry
            try:
                llm_instance = _get_llm_with_retry()
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': f'LLM unavailable: {str(e)[:100]}'})}\n\n"
                return
            
            # Retrieve context if RAG enabled
            sources = []
            context = ""
            if query_req.use_rag and vectorstore is not None:
                context, sources = retrieve_context(query_req.query)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Generate prompt and stream tokens
            prompt = generate_prompt(query_req.query, context)
            token_count = 0
            gen_start = time.time()
            
            for token in llm_instance.stream(
                prompt,
                max_tokens=query_req.max_tokens,
                temperature=query_req.temperature,
                top_p=query_req.top_p
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                token_count += 1
                
                if token_count % 10 == 0:
                    await asyncio.sleep(0.01)
            
            gen_duration = time.time() - gen_start
            token_rate = token_count / gen_duration if gen_duration > 0 else 0
            
            record_tokens_generated(token_count)
            record_query_processed(query_req.use_rag)
            update_token_rate(token_rate)
            
            latency_ms = gen_duration * 1000
            yield f"data: {json.dumps({'type': 'done', 'tokens': token_count, 'latency_ms': latency_ms})}\n\n"
            
            logger.info(f"Stream complete: {token_count} tokens in {latency_ms:.0f}ms ({token_rate:.1f} tok/s)")
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            record_error('stream_failed', 'llm')
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)[:200]})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        f"Unhandled exception: {request.method} {request.url.path}",
        exc_info=exc
    )
    record_error("unhandled_exception", "api")
    
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)[:200] if debug_mode else "An unexpected error occurred"
        }
    )

# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    """Development entrypoint. Production: uvicorn main:app"""
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=CONFIG['server']['host'],
        port=CONFIG['server']['port'],
        log_level="info",
        reload=False
    )