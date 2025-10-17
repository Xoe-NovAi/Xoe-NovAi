#!/usr/bin/env python3
"""
Compatibility wrapper for dependencies_update improvements.

This file implements the enhanced logic from the update but preserves the
original public API (`get_llm`, `get_embeddings`, `get_vectorstore`,
`get_redis_client`) and uses nested config lookups via
`XNAi_rag_app.config_loader.get_config_value`.

Heavy imports (LLM, FAISS) are lazy to avoid import-time failures in tests.
Optional async wrappers and an async http client are provided as opt-in
helpers; Redis async is guarded by runtime availability.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Any
import asyncio

logger = logging.getLogger(__name__)

# Local config loader
from XNAi_rag_app.config_loader import get_config_value, load_config

# Retry utilities
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# System monitoring
import psutil

# HTTP async client (optional)
import httpx
_http_client: Optional[httpx.AsyncClient] = None


def check_available_memory(required_gb: float = 6.0) -> bool:
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    logger.debug("Memory available: %.2fGB (required %.2fGB)", available_gb, required_gb)
    if available_gb < required_gb:
        raise MemoryError(f"Insufficient memory: {available_gb:.2f}GB available, {required_gb:.1f}GB required")
    return True


# -----------------------------
# Redis (sync default, async optional)
# -----------------------------
def get_redis_client():
    try:
        import redis
    except Exception as e:
        logger.error("redis import failed: %s", e)
        raise

    host = get_config_value("redis.host") or os.getenv("REDIS_HOST", "localhost")
    port = int(get_config_value("redis.port", default=6379))
    password = get_config_value("redis.password", default=None)

    client = redis.Redis(host=host, port=port, password=password, decode_responses=False)
    # Do not ping on import; caller may test connectivity
    return client


async def get_redis_client_async():
    try:
        import redis.asyncio as redis_async
    except Exception:
        raise RuntimeError("Async redis not available; upgrade redis package to enable async support")

    host = get_config_value("redis.host") or os.getenv("REDIS_HOST", "localhost")
    port = int(get_config_value("redis.port", default=6379))
    password = get_config_value("redis.password", default=None)

    return redis_async.Redis(host=host, port=port, password=password, decode_responses=False)


# -----------------------------
# HTTP client lifecycle
# -----------------------------
def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        timeout = float(get_config_value("server.http_timeout", default=10))
        _http_client = httpx.AsyncClient(timeout=timeout)
    return _http_client


async def shutdown_dependencies():
    global _http_client
    if _http_client is not None:
        try:
            await _http_client.aclose()
        except Exception:
            logger.exception("Error closing http client")
        finally:
            _http_client = None


# -----------------------------
# LLM initialization (lazy imports)
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type((RuntimeError, OSError)), reraise=True)
def _init_llm_internal(model_path: Optional[str] = None, **kwargs) -> Any:
    """Lazy LLM initializer with memory check and guarded imports."""
    # Determine model path
    model_path = model_path or get_config_value("models.llm_path") or os.getenv("LLM_MODEL_PATH")
    if model_path is None:
        raise FileNotFoundError("LLM model path not provided in config or env")

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"LLM model not found: {model_path}")

    # Memory check
    mem_required = float(get_config_value("performance.memory_limit_gb", default=6.0))
    check_available_memory(mem_required)

    # Lazy import langchain-community wrapper
    try:
        from langchain_community.llms import LlamaCpp
        LlamaImpl = LlamaCpp
    except Exception:
        # Fallback: try low-level binding
        try:
            from llama_cpp import Llama as LlamaImpl
        except Exception as e:
            logger.exception("Failed to import LLM bindings: %s", e)
            raise

    # Build params
    n_ctx = int(get_config_value("models.llm_context_window", default=2048))
    n_threads = int(get_config_value("performance.cpu_threads", default=6))
    f16_kv = get_config_value("performance.f16_kv_enabled", default=True)

    params = {
        "model_path": str(model_file),
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "f16_kv": f16_kv,
    }
    params.update(kwargs)

    # Instantiate using wrapper's signature
    try:
        llm = LlamaImpl(**params)
        logger.info("LLM initialized: %s", model_path)
        return llm
    except Exception as e:
        logger.exception("LLM initialization failed: %s", e)
        raise


def get_llm(model_path: Optional[str] = None, **kwargs) -> Any:
    return _init_llm_internal(model_path=model_path, **kwargs)


async def get_llm_async(model_path: Optional[str] = None, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_llm, model_path, **kwargs)


# -----------------------------
# Embeddings (lazy)
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type((RuntimeError, OSError)), reraise=True)
def get_embeddings(model_path: Optional[str] = None, **kwargs) -> Any:
    mp = model_path or get_config_value("models.embedding_path")
    if mp is None:
        raise FileNotFoundError("Embedding model path not provided in config or env")

    if not Path(mp).exists():
        raise FileNotFoundError(f"Embedding model not found: {mp}")

    try:
        from langchain_community.embeddings import LlamaCppEmbeddings
        EmbeddingsImpl = LlamaCppEmbeddings
    except Exception:
        logger.exception("Failed to import embeddings implementation")
        raise

    params = {
        "model_path": mp,
        "n_ctx": int(get_config_value("models.embedding_n_ctx", default=512)),
        "n_threads": int(get_config_value("performance.embedding_threads", default=2)),
    }
    params.update(kwargs)

    emb = EmbeddingsImpl(**params)
    logger.info("Embeddings initialized: %s", mp)
    return emb


async def get_embeddings_async(model_path: Optional[str] = None, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embeddings, model_path, **kwargs)


# -----------------------------
# Vectorstore (FAISS) with backup fallback
# -----------------------------
def get_vectorstore(embeddings: Any = None, index_path: Optional[str] = None, backup_path: Optional[str] = None) -> Optional[Any]:
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        logger.exception("FAISS import failed")
        return None

    index_path = index_path or get_config_value("vectorstore.index_path")
    backup_path = backup_path or get_config_value("vectorstore.backup_path")

    if index_path is None:
        logger.warning("No index_path configured for vectorstore")
        return None

    index_dir = Path(index_path)

    # Try primary
    try:
        if index_dir.exists() and (index_dir / "index.faiss").exists():
            logger.info("Loading FAISS index from %s", index_dir)
            vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            if get_config_value("backup.faiss.verify_on_load", default=True):
                try:
                    vs.similarity_search("__verify__", k=1)
                except Exception:
                    raise RuntimeError("FAISS verification failed after load")
            return vs
    except Exception:
        logger.exception("Primary FAISS load failed; will try backups if configured")

    # Backup fallback
    if backup_path:
        try:
            backup_dir = Path(backup_path)
            backups = sorted(backup_dir.glob("**/*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for b in backups:
                # naive attempt: if it's a dir with index.faiss
                if b.is_dir() and (b / "index.faiss").exists():
                    try:
                        # copy/restore to index_dir
                        import shutil
                        if index_dir.exists():
                            shutil.rmtree(index_dir)
                        shutil.copytree(b, index_dir)
                        vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
                        logger.info("Restored FAISS from backup %s", b)
                        return vs
                    except Exception:
                        logger.exception("Failed to restore backup %s", b)
        except Exception:
            logger.exception("Error while searching for backups")

    logger.warning("No FAISS vectorstore available (index not found and backups not usable)")
    return None


async def get_vectorstore_async(embeddings: Any = None, index_path: Optional[str] = None, backup_path: Optional[str] = None) -> Optional[Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_vectorstore, embeddings, index_path, backup_path)


# Expose public API names
__all__ = [
    "get_redis_client",
    "get_redis_client_async",
    "get_http_client",
    "shutdown_dependencies",
    "get_llm",
    "get_llm_async",
    "get_embeddings",
    "get_embeddings_async",
    "get_vectorstore",
    "get_vectorstore_async",
]
