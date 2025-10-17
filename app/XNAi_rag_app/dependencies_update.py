#!/usr/bin/env python3
# ============================================================================
# XNAi Phase 1 v0.1.2 - Dependencies Module
# ============================================================================
# Purpose: Centralized initialization of LLM, embeddings, vectorstore,
#          Redis, and HTTP clients for RAG and UI services.
# Guide Reference: Sections 4 & 5 (Dependencies & Performance)
# Last Updated: 2025-10-16
# ============================================================================

import os
import logging
from pathlib import Path
from typing import Optional, Any

# Third-party dependencies
import redis
import httpx
from llama_cpp import Llama
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS

# Local modules
from XNAi_rag_app.config_loader import load_config, get_config_value
from XNAi_rag_app.metrics import (
    record_error,
    record_tokens_generated,
    record_query_processed,
    update_token_rate,
)

logger = logging.getLogger(__name__)
CONFIG = load_config()

# ============================================================================
# REDIS CLIENT INITIALIZATION
# ============================================================================

def init_redis_client() -> redis.Redis:
    """Initialize Redis client for caching and streams."""
    try:
        client = redis.Redis(
            host=CONFIG['REDIS_HOST'],
            port=int(CONFIG['REDIS_PORT']),
            password=CONFIG.get('REDIS_PASSWORD', None),
            decode_responses=True,
            socket_timeout=int(CONFIG.get('REDIS_TIMEOUT', 60)),
            max_connections=int(CONFIG.get('REDIS_MAX_CONNECTIONS', 50)),
        )
        # Test connection
        client.ping()
        logger.info("Redis client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        record_error('init', 'redis')
        raise

REDIS_CLIENT: Optional[redis.Redis] = None

def get_redis_client() -> redis.Redis:
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        REDIS_CLIENT = init_redis_client()
    return REDIS_CLIENT

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def init_llm() -> Llama:
    """Initialize the LLM with optimized Ryzen threading and F16_KV memory."""
    try:
        llm_path = CONFIG['LLM_MODEL_PATH']
        n_ctx = int(CONFIG['LLM_CONTEXT_WINDOW'])
        n_batch = int(CONFIG['LLAMA_CPP_N_BATCH'])
        n_threads = int(CONFIG['LLAMA_CPP_N_THREADS'])
        f16_kv = CONFIG.get('LLAMA_CPP_F16_KV', True)
        
        llm = Llama(
            model_path=llm_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            f16_kv=f16_kv,
            use_mlock=CONFIG.get('LLAMA_CPP_USE_MLOCK', True),
            use_mmap=CONFIG.get('LLAMA_CPP_USE_MMAP', True),
            seed=int(CONFIG.get('LLM_SEED', -1)),
        )
        logger.info(f"LLM initialized: {llm_path}")
        return llm
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        record_error('init', 'llm')
        raise

LLM_MODEL: Optional[Llama] = None

def get_llm_model() -> Llama:
    global LLM_MODEL
    if LLM_MODEL is None:
        LLM_MODEL = init_llm()
    return LLM_MODEL

# ============================================================================
# EMBEDDINGS INITIALIZATION
# ============================================================================

def init_embeddings() -> LlamaCppEmbeddings:
    """Initialize embeddings model."""
    try:
        embedding_path = CONFIG['EMBEDDING_MODEL_PATH']
        embeddings = LlamaCppEmbeddings(
            model_path=embedding_path,
            n_ctx=int(CONFIG['LLM_CONTEXT_WINDOW']),
            n_threads=int(CONFIG['LLAMA_CPP_N_THREADS']),
            batch_size=int(CONFIG['EMBEDDING_BATCH_SIZE']),
            normalize_embeddings=CONFIG.get('EMBEDDING_NORMALIZE', True),
        )
        logger.info(f"Embeddings initialized: {embedding_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Embeddings initialization failed: {e}")
        record_error('init', 'embeddings')
        raise

EMBEDDINGS_MODEL: Optional[LlamaCppEmbeddings] = None

def get_embeddings_model() -> LlamaCppEmbeddings:
    global EMBEDDINGS_MODEL
    if EMBEDDINGS_MODEL is None:
        EMBEDDINGS_MODEL = init_embeddings()
    return EMBEDDINGS_MODEL

# ============================================================================
# VECTORSTORE INITIALIZATION
# ============================================================================

def init_vectorstore() -> FAISS:
    """Initialize FAISS vectorstore."""
    try:
        embeddings = get_embeddings_model()
        index_path = Path(CONFIG.get('VECTORSTORE_PATH', '/data/faiss_index'))
        if index_path.exists():
            vectorstore = FAISS.load_local(str(index_path), embeddings)
            logger.info(f"FAISS vectorstore loaded from {index_path}")
        else:
            vectorstore = FAISS.from_texts([], embeddings)
            logger.info("FAISS vectorstore created empty")
        return vectorstore
    except Exception as e:
        logger.error(f"Vectorstore initialization failed: {e}")
        record_error('init', 'vectorstore')
        raise

VECTORSTORE: Optional[FAISS] = None

def get_vectorstore() -> FAISS:
    global VECTORSTORE
    if VECTORSTORE is None:
        VECTORSTORE = init_vectorstore()
    return VECTORSTORE

# ============================================================================
# HTTP CLIENT INITIALIZATION
# ============================================================================

def init_http_client() -> httpx.AsyncClient:
    """Initialize shared HTTP client for API calls."""
    timeout = int(CONFIG.get('HTTP_TIMEOUT', 30))
    try:
        client = httpx.AsyncClient(timeout=timeout)
        logger.info("HTTPX async client initialized")
        return client
    except Exception as e:
        logger.error(f"HTTP client initialization failed: {e}")
        record_error('init', 'http_client')
        raise

HTTP_CLIENT: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = init_http_client()
    return HTTP_CLIENT

# ============================================================================
# DEPENDENCIES READY CHECK
# ============================================================================

def check_dependencies_ready() -> bool:
    """Check all critical dependencies are initialized and healthy."""
    try:
        # Redis
        get_redis_client().ping()
        # LLM
        _ = get_llm_model()
        # Embeddings
        _ = get_embeddings_model()
        # Vectorstore
        _ = get_vectorstore()
        # HTTP client
        _ = get_http_client()
        logger.info("All dependencies are ready")
        return True
    except Exception as e:
        logger.error(f"Dependency readiness check failed: {e}")
        return False

# ============================================================================
# CLEANUP / SHUTDOWN
# ============================================================================

async def shutdown_dependencies():
    """Cleanly shutdown async clients and free resources."""
    global HTTP_CLIENT
    try:
        if HTTP_CLIENT:
            await HTTP_CLIENT.aclose()
            logger.info("HTTP client closed")
            HTTP_CLIENT = None
    except Exception as e:
        logger.warning(f"Error shutting down HTTP client: {e}")

# ============================================================================
# MODULE TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    print("="*70)
    print("XNAi Dependencies Module - Test Suite")
    print("="*70)

    # Test Redis
    try:
        r = get_redis_client()
        r.ping()
        print("✓ Redis OK")
    except Exception as e:
        print(f"✗ Redis failed: {e}")

    # Test LLM
    try:
        llm = get_llm_model()
        print("✓ LLM OK")
    except Exception as e:
        print(f"✗ LLM failed: {e}")

    # Test Embeddings
    try:
        emb = get_embeddings_model()
        print("✓ Embeddings OK")
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")

    # Test Vectorstore
    try:
        vs = get_vectorstore()
        print("✓ Vectorstore OK")
    except Exception as e:
        print(f"✗ Vectorstore failed: {e}")

    # Test HTTP client
    try:
        client = get_http_client()
        print("✓ HTTP client OK")
        asyncio.run(shutdown_dependencies())
    except Exception as e:
        print(f"✗ HTTP client failed: {e}")

    print("="*70)
    print("Dependencies module test complete")
    print("="*70)
