#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.1 - FastAPI RAG Service (PRODUCTION-READY)
# ============================================================================
# Purpose: Main FastAPI application with streaming RAG capabilities
# Guide Reference: Section 4.1 (Complete main.py Implementation)
# Last Updated: 2025-10-11
# Features:
#   - SSE streaming for real-time responses
#   - Context truncation (<6GB memory target)
#   - Rate limiting (60 req/min)
#   - Prometheus metrics integration
#   - Redis caching support
#   - Lazy LLM loading
#   - Global exception handling
# ============================================================================

import os
import time
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

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

# Configuration and dependencies
from config_loader import load_config, get_config_value
from logging_config import setup_logging, get_logger, PerformanceLogger
from dependencies import get_llm, get_embeddings, get_vectorstore
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
# GLOBAL STATE (Lazy Loading)
# ============================================================================

llm = None
embeddings = None
vectorstore = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """
    Query request model.
    
    Guide Reference: Section 4.1 (Request Models)
    """
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="User query",
        examples=["What is Xoe-NovAi?"]
    )
    use_rag: bool = Field(
        True, 
        description="Whether to use RAG context retrieval"
    )
    max_tokens: int = Field(
        512, 
        ge=1, 
        le=2048, 
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature"
    )
    top_p: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )

class QueryResponse(BaseModel):
    """
    Query response model.
    
    Guide Reference: Section 4.1 (Response Models)
    """
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
# CONTEXT TRUNCATION & RAG LOGIC
# ============================================================================

def _build_truncated_context(
    docs: List,
    per_doc_chars: int = None,
    total_chars: int = None
) -> tuple:
    """
    Build truncated context from documents.
    
    Guide Reference: Section 4.1 (Context Truncation - CRITICAL)
    
    This ensures context stays within memory limits by truncating each
    document and enforcing a total character limit.
    
    Args:
        docs: List of Document objects from vectorstore
        per_doc_chars: Max characters per document
        total_chars: Max total characters
        
    Returns:
        Tuple of (context_text, source_list)
    """
    if per_doc_chars is None:
        per_doc_chars = CONFIG['performance']['per_doc_chars']
    
    if total_chars is None:
        total_chars = CONFIG['performance']['total_chars']
    
    context = ""
    sources = []
    
    for doc in docs:
        # Truncate document
        doc_text = doc.page_content[:per_doc_chars]
        source = doc.metadata.get("source", "unknown")
        
        # Add source header for clarity
        formatted_doc = f"\n[Source: {source}]\n{doc_text}\n"
        
        # Check if adding this doc would exceed total limit
        if len(context + formatted_doc) > total_chars:
            logger.debug(f"Context truncation at {len(context)} chars (limit: {total_chars})")
            break
        
        context += formatted_doc
        
        # Add source if not already present
        if source not in sources:
            sources.append(source)
    
    # Final truncation to ensure we're under limit
    context = context[:total_chars]
    
    logger.debug(f"Built context: {len(context)} chars from {len(sources)} sources")
    return context, sources

def retrieve_context(
    query: str,
    top_k: int = None,
    similarity_threshold: float = None
) -> tuple:
    """
    Retrieve relevant documents from FAISS vectorstore.
    
    Guide Reference: Section 2 (RAG Configuration)
    Best Practice: Configurable top_k with timing metrics
    
    Args:
        query: User query string
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score (unused in FAISS but kept for API)
        
    Returns:
        Tuple of (context_str, sources_list)
    """
    if not vectorstore:
        logger.warning("Vectorstore not initialized, skipping RAG")
        return "", []
    
    # Get config values
    if top_k is None:
        top_k = get_config_value('rag.top_k', 5)
    
    if similarity_threshold is None:
        similarity_threshold = get_config_value('rag.similarity_threshold', 0.7)
    
    try:
        # Similarity search with timing
        start_time = time.time()
        
        docs = vectorstore.similarity_search(query, k=top_k)
        
        retrieval_ms = (time.time() - start_time) * 1000
        record_rag_retrieval(retrieval_ms)
        
        if not docs:
            logger.warning(f"No documents retrieved for query: {query[:50]}...")
            return "", []
        
        # Build truncated context
        per_doc_chars = int(os.getenv("RAG_PER_DOC_CHARS", CONFIG['performance']['per_doc_chars']))
        total_chars = int(os.getenv("RAG_TOTAL_CHARS", CONFIG['performance']['total_chars']))
        
        context, sources = _build_truncated_context(docs, per_doc_chars, total_chars)
        
        logger.info(f"Retrieved {len(sources)} relevant documents in {retrieval_ms:.2f}ms")
        return context, sources
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}", exc_info=True)
        record_error("rag_retrieval", "vectorstore")
        return "", []

def generate_prompt(query: str, context: str = "") -> str:
    """
    Generate LLM prompt with optional RAG context.
    
    Guide Reference: Section 4.1 (Prompt Engineering)
    Best Practice: Clear instructions with context separation
    
    Args:
        query: User query
        context: Optional RAG context
        
    Returns:
        Formatted prompt string
    """
    if context:
        # RAG prompt template
        prompt = f"""Based on the following context, answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    else:
        # Direct query (no RAG)
        prompt = f"""Question: {query}

Answer:"""
    
    return prompt

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Guide Reference: Section 4.1 (Startup/Shutdown)
    
    This runs on startup and shutdown to initialize/cleanup resources.
    """
    # Startup
    logger.info("=" * 70)
    logger.info("Starting Xoe-NovAi RAG API v0.1.1")
    logger.info("=" * 70)
    
    # Start metrics server
    try:
        start_metrics_server()
        logger.info("✓ Prometheus metrics server started")
    except Exception as e:
        logger.warning(f"Metrics server failed to start: {e}")
    
    # Check memory
    memory_gb = psutil.virtual_memory().used / (1024 ** 3)
    logger.info(f"Current memory usage: {memory_gb:.2f}GB")
    
    if memory_gb > CONFIG['performance']['memory_warning_threshold_gb']:
        logger.warning(f"Memory usage high: {memory_gb:.2f}GB (warning threshold: {CONFIG['performance']['memory_warning_threshold_gb']}GB)")
    
    # Initialize embeddings and vectorstore (LLM is lazy loaded)
    global embeddings, vectorstore
    
    try:
        logger.info("Initializing embeddings...")
        embeddings = get_embeddings()
        logger.info("✓ Embeddings initialized successfully")
        
        logger.info("Loading vectorstore...")
        vectorstore = get_vectorstore(embeddings)
        if vectorstore:
            vector_count = vectorstore.index.ntotal
            logger.info(f"✓ Vectorstore loaded: {vector_count} vectors")
        else:
            logger.warning("⚠ No vectorstore found - RAG disabled (run ingest_library.py)")
        
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}", exc_info=True)
        # Continue anyway - services can still function without RAG
    
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
    """
    Log all requests with timing.
    
    Guide Reference: Section 4.1 (Request Logging)
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Record metrics
    record_request(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    )
    
    # Log
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": get_remote_address(request)
        }
    )
    
    return response

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Guide Reference: Section 4.1 (Root Endpoint)
    """
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
    """
    Health check endpoint with integrated healthcheck.py results.
    
    Guide Reference: Section 5.1 (rev_1.3 - Integrated Health Checks)
    
    Returns:
        Health status with component information
    """
    # Import healthcheck functions (rev_1.3)
    try:
        from healthcheck import run_health_checks
        ERROR_RECOVERY_ENABLED = os.getenv("ERROR_RECOVERY_ENABLED", "true").lower() == "true"
    except ImportError:
        run_health_checks = None
        ERROR_RECOVERY_ENABLED = False
    
    # Get memory
    memory_gb = psutil.virtual_memory().used / (1024 ** 3)
    
    # Basic component checks
    components = {
        "embeddings": embeddings is not None,
        "vectorstore": vectorstore is not None,
        "llm": llm is not None,
    }
    
    # Run healthcheck.py checks if available (rev_1.3)
    if run_health_checks and ERROR_RECOVERY_ENABLED:
        try:
            # Run subset of checks (non-blocking)
            health_results = await asyncio.to_thread(
                run_health_checks,
                targets=['memory', 'redis', 'ryzen'],
                critical_only=False
            )
            
            # Merge results into components
            for target, (success, message) in health_results.items():
                components[f"health_{target}"] = success
                
            logger.info(f"Health checks completed: {len(health_results)} checks")
            
        except Exception as e:
            logger.warning(f"Health check integration failed: {e}")
            # Continue with basic checks
    
    # Determine status
    if memory_gb > CONFIG['performance']['memory_limit_gb']:
        status = "degraded"
    elif not components['embeddings']:
        status = "degraded"
    elif not components.get('health_memory', True):
        status = "degraded"
    elif not components['vectorstore']:
        status = "partial"  # Can work without vectorstore (no RAG)
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
    """
    Synchronous query endpoint.
    
    Guide Reference: Section 4.1 (Query Endpoint)
    
    Args:
        request: FastAPI request
        query_req: Query request model
        
    Returns:
        Query response with sources
    """
    global llm
    
    with MetricsTimer(response_latency_ms, endpoint='/query', method='POST'):
        start_time = time.time()
        
        try:
            # Initialize LLM (lazy loading)
            if llm is None:
                logger.info("Lazy loading LLM...")
                llm = get_llm()
                logger.info("✓ LLM loaded successfully")
            
            # Retrieve context if RAG enabled
            sources = []
            context = ""
            if query_req.use_rag and vectorstore:
                context, sources = retrieve_context(query_req.query)
            
            # Generate prompt
            prompt = generate_prompt(query_req.query, context)
            
            # Generate response
            gen_start = time.time()
            response = llm.invoke(
                prompt,
                max_tokens=query_req.max_tokens,
                temperature=query_req.temperature,
                top_p=query_req.top_p
            )
            gen_duration = time.time() - gen_start
            
            # Calculate metrics
            tokens_approx = len(response.split())
            token_rate = tokens_approx / gen_duration if gen_duration > 0 else 0
            
            # Record metrics
            record_tokens_generated(tokens_approx)
            record_query_processed(query_req.use_rag)
            update_token_rate(token_rate)
            
            # Log performance
            perf_logger.log_token_generation(
                tokens=tokens_approx,
                duration_s=gen_duration
            )
            
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
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            record_error('query_failed', 'llm')
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)[:200]}")

@app.post("/stream")
@limiter.limit("60/minute")
async def stream_endpoint(request: Request, query_req: QueryRequest):
    """
    Streaming query endpoint (SSE).
    
    Guide Reference: Section 4.1 (SSE Streaming)
    
    Args:
        request: FastAPI request
        query_req: Query request model
        
    Returns:
        StreamingResponse with SSE events
    """
    global llm
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE stream."""
        try:
            # Initialize LLM (lazy loading)
            if llm is None:
                logger.info("Lazy loading LLM for streaming...")
                llm = get_llm()
                logger.info("✓ LLM loaded successfully")
            
            # Retrieve context if RAG enabled
            sources = []
            context = ""
            if query_req.use_rag and vectorstore:
                context, sources = retrieve_context(query_req.query)
                
                # Send sources first
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Generate prompt
            prompt = generate_prompt(query_req.query, context)
            
            # Stream tokens
            token_count = 0
            gen_start = time.time()
            
            for token in llm.stream(
                prompt,
                max_tokens=query_req.max_tokens,
                temperature=query_req.temperature,
                top_p=query_req.top_p
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                token_count += 1
                
                # Yield control periodically (prevents blocking)
                if token_count % 10 == 0:
                    await asyncio.sleep(0.01)
            
            gen_duration = time.time() - gen_start
            
            # Calculate token rate
            token_rate = token_count / gen_duration if gen_duration > 0 else 0
            
            # Record metrics
            record_tokens_generated(token_count)
            record_query_processed(query_req.use_rag)
            update_token_rate(token_rate)
            
            # Send completion event
            latency_ms = gen_duration * 1000
            yield f"data: {json.dumps({'type': 'done', 'tokens': token_count, 'latency_ms': latency_ms})}\n\n"
            
            logger.info(
                f"Stream complete: {token_count} tokens in {latency_ms:.0f}ms "
                f"({token_rate:.1f} tok/s)"
            )
            
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    Guide Reference: Section 4.1 (Error Handling)
    Best Practice: Structured error responses with logging
    """
    logger.error(
        f"Unhandled exception: {request.method} {request.url.path}",
        exc_info=exc
    )
    record_error("unhandled_exception", "api")
    
    # Show full error in debug mode only
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
    """
    Development entrypoint.
    
    Production deployment uses: uvicorn main:app
    """
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=CONFIG['server']['host'],
        port=CONFIG['server']['port'],
        log_level="info",
        reload=False
    )
