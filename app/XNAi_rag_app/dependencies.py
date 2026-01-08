#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.4-stable - Dependencies Module (PRODUCTION-READY)
# ============================================================================
# Purpose: Centralized dependency management for LLM, embeddings, vectorstore, curator
# Guide Reference: Section 4 (Core Dependencies Module)
# Last Updated: 2025-10-18
# ============================================================================
# Features:
#   - @retry decorators (3 attempts, exponential backoff)
#   - FAISS backup fallback (/backups/*.bak)
#   - LlamaCppEmbeddings (50% memory savings vs HuggingFace)
#   - Kwarg filtering for Pydantic compatibility
#   - Memory checks before loading (<6GB threshold)
#   - get_curator() for CrawlModule integration
#   - Async wrapper functions for all components
#   - No HuggingFace dependencies
# ============================================================================

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

# Retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# LangChain imports (lazy loaded where possible)
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# System monitoring
import psutil

# HTTP client
import httpx

# Logging setup
logger = logging.getLogger(__name__)

# Configuration loader
from config_loader import load_config, get_config_value

# Load config once at module level
CONFIG = load_config()

# Global instances for singleton pattern
_redis_client: Optional[Any] = None
_http_client: Optional[httpx.AsyncClient] = None

# ============================================================================
# LLAMA CPP PARAMETER FILTERING
# ============================================================================

def filter_llama_kwargs(**kwargs) -> dict:
    """
    Filter kwargs to only valid LlamaCpp parameters.
    
    Guide Reference: Section 4.2 (Pydantic Compatibility)
    
    Prevents Pydantic validation errors from extra kwargs.
    
    Args:
        **kwargs: Raw kwargs from environment/config
        
    Returns:
        Filtered kwargs safe for LlamaCpp initialization
    """
    valid_params = {
        'model_path', 'n_ctx', 'n_batch', 'n_gpu_layers', 'n_threads',
        'n_parts', 'seed', 'f16_kv', 'logits_all', 'vocab_only',
        'use_mlock', 'use_mmap', 'embedding', 'last_n_tokens_size',
        'lora_base', 'lora_path', 'verbose', 'max_tokens', 'temperature',
        'top_p', 'top_k', 'repeat_penalty', 'stop', 'streaming'
    }
    
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Log filtered parameters for debugging
    removed = set(kwargs.keys()) - set(filtered.keys())
    if removed:
        logger.debug(f"Filtered out invalid llama-cpp params: {removed}")
    
    return filtered

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def check_available_memory(required_gb: float = 6.0) -> bool:
    """
    Check if sufficient memory available before loading models.
    
    Guide Reference: Section 4.2 (Memory Management)
    
    Args:
        required_gb: Required memory in GB (default: 6.0)
        
    Returns:
        True if sufficient memory available
        
    Raises:
        MemoryError: If insufficient memory
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    used_gb = memory.used / (1024 ** 3)
    
    logger.info(
        f"Memory status: {used_gb:.2f}GB used, {available_gb:.2f}GB available "
        f"(required: {required_gb:.1f}GB)"
    )
    
    if available_gb < required_gb:
        raise MemoryError(
            f"Insufficient memory: {available_gb:.2f}GB available, "
            f"{required_gb:.1f}GB required. Close other applications or increase system RAM."
        )
    
    return True

# ============================================================================
# REDIS CLIENT
# ============================================================================

def get_redis_client():
    """
    Get Redis client (singleton pattern).
    
    Guide Reference: Section 4.1 (Redis Client)
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            import redis
        except ImportError:
            logger.error("redis package not installed")
            raise
        
        host = get_config_value("redis.host") or os.getenv("REDIS_HOST", "redis")
        port = int(get_config_value("redis.port", default=6379))
        password = get_config_value("redis.password") or os.getenv("REDIS_PASSWORD")
        timeout = int(get_config_value("redis.timeout_seconds", default=60))
        
        _redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=False,
            socket_timeout=timeout,
            max_connections=int(get_config_value("redis.max_connections", default=50))
        )
        
        # Test connection
        try:
            _redis_client.ping()
            logger.info(f"Redis client connected: {host}:{port}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            _redis_client = None
            raise
    
    return _redis_client

async def get_redis_client_async():
    """
    Get async Redis client (requires redis[asyncio]).
    
    Returns:
        Async Redis client
    """
    try:
        import redis.asyncio as redis_async
    except ImportError:
        raise RuntimeError(
            "Async redis not available. Install: pip install redis[asyncio]"
        )
    
    host = get_config_value("redis.host") or os.getenv("REDIS_HOST", "redis")
    port = int(get_config_value("redis.port", default=6379))
    password = get_config_value("redis.password") or os.getenv("REDIS_PASSWORD")
    
    return redis_async.Redis(
        host=host,
        port=port,
        password=password,
        decode_responses=False
    )

# ============================================================================
# HTTP CLIENT
# ============================================================================

def get_http_client() -> httpx.AsyncClient:
    """
    Get shared HTTP client (singleton pattern).
    
    Returns:
        Async HTTP client
    """
    global _http_client
    
    if _http_client is None:
        timeout = float(get_config_value("server.timeout_seconds", default=30))
        _http_client = httpx.AsyncClient(timeout=timeout)
        logger.info("HTTP client initialized")
    
    return _http_client

async def shutdown_dependencies():
    """
    Cleanly shutdown async clients and free resources.
    """
    global _http_client
    
    if _http_client is not None:
        try:
            await _http_client.aclose()
            logger.info("HTTP client closed")
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")
        finally:
            _http_client = None

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, OSError, ConnectionError, TimeoutError)),
    reraise=True
)
def get_llm(model_path: Optional[str] = None, **kwargs) -> LlamaCpp:
    """
    Initialize LlamaCpp LLM with Ryzen optimization.
    
    Guide Reference: Section 4.2.1 (LLM Configuration)
    
    Critical optimizations:
    - f16_kv=true: Halves KV cache memory (~1GB savings)
    - n_threads=6: Optimal for Ryzen 7 5700U (75% of 8C/16T)
    - use_mlock=true: Lock model in RAM (prevent swapping)
    - use_mmap=true: Memory-mapped file access for efficiency
    
    Args:
        model_path: Path to GGUF model (default: from config)
        **kwargs: Additional llama-cpp parameters
        
    Returns:
        Initialized LlamaCpp instance
        
    Raises:
        MemoryError: If insufficient memory
        FileNotFoundError: If model not found
        RuntimeError: If initialization fails after 3 retries
    """
    # Check memory first (fail-fast)
    check_available_memory(required_gb=CONFIG['performance']['memory_limit_gb'])
    
    # Load model path from config if not provided
    if model_path is None:
        model_path = os.getenv(
            "LLM_MODEL_PATH",
            CONFIG["models"]["llm_path"]
        )
    
    # Verify model exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"LLM model not found: {model_path}\n"
            f"Please ensure the model file exists or update LLM_MODEL_PATH in .env"
        )
    
    logger.info(f"Loading LLM from {model_path} ({model_file.stat().st_size / (1024**3):.2f}GB)")
    
    # Ryzen optimization
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Build parameters with environment variable overrides
    llm_params = {
        'model_path': model_path,
        'n_ctx': int(os.getenv('LLAMA_CPP_N_CTX', CONFIG['models']['llm_context_window'])),
        'n_batch': int(os.getenv('LLAMA_CPP_N_BATCH', 512)),
        'n_threads': int(os.getenv('LLAMA_CPP_N_THREADS', CONFIG['performance']['cpu_threads'])),
        'n_gpu_layers': 0,  # CPU-only
        'f16_kv': os.getenv('LLAMA_CPP_F16_KV', 'true').lower() == 'true',
        'use_mlock': os.getenv('LLAMA_CPP_USE_MLOCK', 'true').lower() == 'true',
        'use_mmap': os.getenv('LLAMA_CPP_USE_MMAP', 'true').lower() == 'true',
        'verbose': os.getenv('LLM_VERBOSE', 'false').lower() == 'true',
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', 512)),
        'temperature': float(os.getenv('LLM_TEMPERATURE', 0.7)),
        'top_p': float(os.getenv('LLM_TOP_P', 0.95)),
        'top_k': int(os.getenv('LLM_TOP_K', 40)),
        'repeat_penalty': float(os.getenv('LLM_REPEAT_PENALTY', 1.1)),
    }
    
    # Merge with provided kwargs
    llm_params.update(kwargs)
    
    # Filter to valid params
    filtered_params = filter_llama_kwargs(**llm_params)
    
    logger.info(
        f"LLM initialization: n_ctx={filtered_params['n_ctx']}, "
        f"n_threads={filtered_params['n_threads']}, "
        f"f16_kv={filtered_params['f16_kv']}, "
        f"use_mlock={filtered_params['use_mlock']}"
    )
    
    try:
        llm = LlamaCpp(**filtered_params)
        logger.info("LLM initialized successfully")
        return llm
        
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to initialize LLM after retries: {e}\n"
            f"Check model path, memory availability, and system resources."
        )

async def get_llm_async(model_path: Optional[str] = None, **kwargs) -> LlamaCpp:
    """
    Async wrapper for LLM initialization.
    
    Args:
        model_path: Path to model
        **kwargs: Additional parameters
        
    Returns:
        Initialized LLM
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: get_llm(model_path, **kwargs))

# ============================================================================
# EMBEDDINGS INITIALIZATION
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, OSError)),
    reraise=True
)
def get_embeddings(model_path: Optional[str] = None, **kwargs) -> LlamaCppEmbeddings:
    """
    Initialize LlamaCppEmbeddings model.
    
    Guide Reference: Section 4.2.2 (Embeddings - 50% memory savings)
    
    LlamaCppEmbeddings advantages:
    - 50% memory savings vs HuggingFaceEmbeddings
    - No PyTorch dependency
    - CPU-optimized for Ryzen architecture
    - 384 dimensions (all-MiniLM-L12-v2 model)
    
    Args:
        model_path: Path to embedding model (default: from config)
        **kwargs: Additional parameters
        
    Returns:
        Initialized LlamaCppEmbeddings instance
    """
    if model_path is None:
        model_path = os.getenv(
            "EMBEDDING_MODEL_PATH",
            CONFIG["models"]["embedding_path"]
        )
    
    # Verify model exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {model_path}\n"
            f"Please ensure the model file exists or update EMBEDDING_MODEL_PATH in .env"
        )
    
    logger.info(f"Loading embeddings from {model_path} ({model_file.stat().st_size / (1024**2):.1f}MB)")
    
    # Embeddings use fewer threads
    embed_params = {
        'model_path': model_path,
        'n_ctx': int(os.getenv('EMBEDDING_N_CTX', 512)),
        'n_threads': int(os.getenv('EMBEDDING_N_THREADS', 2)),
    }
    
    embed_params.update(kwargs)
    
    try:
        embeddings = LlamaCppEmbeddings(**embed_params)
        logger.info(
            f"Embeddings initialized: {CONFIG['models']['embedding_dimensions']} dimensions, "
            f"n_threads={embed_params['n_threads']}"
        )
        return embeddings
        
    except Exception as e:
        logger.error(f"Embeddings initialization failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to initialize embeddings after retries: {e}\n"
            f"Check model path and system resources."
        )

async def get_embeddings_async(model_path: Optional[str] = None, **kwargs) -> LlamaCppEmbeddings:
    """
    Async wrapper for embeddings initialization.
    
    Args:
        model_path: Path to model
        **kwargs: Additional parameters
        
    Returns:
        Initialized embeddings
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: get_embeddings(model_path, **kwargs))

# ============================================================================
# VECTORSTORE INITIALIZATION WITH BACKUP FALLBACK
# ============================================================================

def get_vectorstore(
    embeddings: Optional[LlamaCppEmbeddings] = None,
    index_path: Optional[str] = None,
    backup_path: Optional[str] = None
) -> Optional[FAISS]:
    """
    Load FAISS vectorstore with backup fallback.
    
    Guide Reference: Section 4.2.3 (FAISS Backup Strategy)
    
    Loading strategy:
    1. Try primary index at /app/XNAi_rag_app/faiss_index
    2. If primary fails, try backups (most recent first, up to 5)
    3. If backup succeeds, restore it to primary location
    4. If verify_on_load=true, validate with test search
    
    Args:
        embeddings: Embeddings instance (will be created if None)
        index_path: Primary index path (default: from config)
        backup_path: Backup directory path (default: from config)
        
    Returns:
        Loaded FAISS vectorstore or None if not found
    """
    # Initialize embeddings if not provided
    if embeddings is None:
        try:
            embeddings = get_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings for vectorstore: {e}")
            return None
    
    if index_path is None:
        index_path = os.getenv(
            "FAISS_INDEX_PATH",
            CONFIG["vectorstore"]["index_path"]
        )
    
    if backup_path is None:
        backup_path = os.getenv(
            "FAISS_BACKUP_PATH",
            CONFIG["vectorstore"]["backup_path"]
        )
    
    index_dir = Path(index_path)
    backup_dir = Path(backup_path)
    
    # Try loading primary index
    if index_dir.exists() and (index_dir / "index.faiss").exists():
        logger.info(f"Loading FAISS index from {index_path}")
        
        try:
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Validate if enabled
            if CONFIG["backup"]["faiss"].get("verify_on_load", True):
                try:
                    test_result = vectorstore.similarity_search("test", k=1)
                    vector_count = vectorstore.index.ntotal
                    logger.info(
                        f"FAISS index validated: {vector_count} vectors, "
                        f"search functional"
                    )
                except Exception as e:
                    logger.error(f"FAISS validation failed: {e}")
                    raise
            else:
                vector_count = vectorstore.index.ntotal
                logger.info(f"FAISS index loaded: {vector_count} vectors (validation skipped)")
            
            logger.warning("FAISS loaded with allow_dangerous_deserialization=True")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to load primary FAISS index: {e}")
            logger.info("Attempting backup fallback...")
    
    # Try loading from backups
    if backup_dir.exists():
        backup_dirs = sorted(
            [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("faiss_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        max_backups_to_try = CONFIG["backup"]["faiss"].get("max_count", 5)
        
        for backup in backup_dirs[:max_backups_to_try]:
            backup_index = backup / "index.faiss"
            
            if not backup_index.exists():
                continue
            
            logger.info(f"Trying backup: {backup}")
            
            try:
                vectorstore = FAISS.load_local(
                    str(backup),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                vector_count = vectorstore.index.ntotal
                logger.info(f"Loaded from backup: {backup} ({vector_count} vectors)")
                
                # Restore backup to primary location
                logger.info(f"Restoring backup to primary location: {index_path}")
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                shutil.copytree(backup, index_dir)
                logger.info("Backup restored successfully")
                
                return vectorstore
                
            except Exception as e:
                logger.warning(f"Backup {backup} failed to load: {e}")
                continue
    
    # No valid index found
    logger.warning(
        "No valid FAISS index found (primary or backups). "
        "Run ingestion to create: python3 scripts/ingest_library.py"
    )
    return None

async def get_vectorstore_async(
    embeddings: Optional[LlamaCppEmbeddings] = None,
    index_path: Optional[str] = None,
    backup_path: Optional[str] = None
) -> Optional[FAISS]:
    """
    Async wrapper for vectorstore initialization.
    
    Args:
        embeddings: Embeddings instance
        index_path: Primary index path
        backup_path: Backup directory
        
    Returns:
        Loaded vectorstore or None
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_vectorstore(embeddings, index_path, backup_path)
    )

# ============================================================================
# CRAWLMODULE INTEGRATION
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RuntimeError, OSError)),
    reraise=True
)
def get_curator(cache_dir: Optional[str] = None, **kwargs) -> Any:
    """
    Initialize CrawlModule for library curation.
    
    Guide Reference: Section 4.3 (CrawlModule Integration)
    Guide Reference: Section 9.2 (CrawlModule Architecture)
    
    NEW in v0.1.4: Provides access to CrawlModule for:
    - Library curation from 4 sources (Gutenberg, arXiv, PubMed, YouTube)
    - Rate limiting (30 req/min)
    - URL allowlist validation
    - Metadata tracking in knowledge/curator/index.toml
    - Redis caching
    - Auto-embed to FAISS (optional)
    
    Args:
        cache_dir: Cache directory (default: /app/cache)
        **kwargs: Additional crawler parameters
        
    Returns:
        Initialized crawler instance (from crawl.py functions)
        
    Note:
        Returns a dict of functions from crawl.py module, not a class instance
    """
    try:
        # Import crawl module
        sys.path.insert(0, '/app/XNAi_rag_app')
        import crawl
        
        # Return module itself - it has all the functions we need
        logger.info("CrawlModule functions loaded successfully")
        return crawl
        
    except ImportError as e:
        logger.error("CrawlModule not found - crawl.py may be missing")
        raise ImportError(
            "CrawlModule requires crawl.py in app/XNAi_rag_app/. "
            "Ensure crawl.py exists."
        ) from e

# ============================================================================
# CLEANUP UTILITIES
# ============================================================================

def cleanup_old_backups(
    backup_path: str,
    max_count: int = 5,
    retention_days: int = 7
):
    """
    Clean up old FAISS backups based on retention policy.
    
    Guide Reference: Section 4.2.3 (Backup Retention)
    
    Args:
        backup_path: Backup directory
        max_count: Maximum number of backups to keep
        retention_days: Maximum age in days
    """
    backup_dir = Path(backup_path)
    
    if not backup_dir.exists():
        return
    
    backups = sorted(
        [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("faiss_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    # Remove backups beyond max_count
    removed_count = 0
    for backup in backups[max_count:]:
        try:
            shutil.rmtree(backup)
            removed_count += 1
            logger.info(f"Removed old backup: {backup}")
        except Exception as e:
            logger.warning(f"Failed to remove backup {backup}: {e}")
    
    # Remove backups older than retention_days
    cutoff_time = datetime.now().timestamp() - (retention_days * 86400)
    
    for backup in backups[:max_count]:
        if backup.stat().st_mtime < cutoff_time:
            try:
                shutil.rmtree(backup)
                removed_count += 1
                logger.info(f"Removed expired backup: {backup}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleanup complete: removed {removed_count} old backups")

# ============================================================================
# HEALTH CHECKS
# ============================================================================

def check_dependencies_ready() -> Dict[str, bool]:
    """
    Check all critical dependencies are initialized and healthy.
    
    Returns:
        Dict with status of each component
    """
    status = {
        "redis": False,
        "llm": False,
        "embeddings": False,
        "vectorstore": False,
        "http_client": False,
    }
    
    # Redis
    try:
        client = get_redis_client()
        client.ping()
        status["redis"] = True
    except Exception as e:
        logger.error(f"Redis check failed: {e}")
    
    # LLM (expensive, skip in health checks)
    # status["llm"] = True  # Assume OK if loaded once
    
    # Embeddings (expensive, skip in health checks)
    # status["embeddings"] = True  # Assume OK if loaded once
    
    # Vectorstore (check file existence)
    try:
        index_path = Path(CONFIG["vectorstore"]["index_path"])
        status["vectorstore"] = (index_path / "index.faiss").exists()
    except Exception as e:
        logger.error(f"Vectorstore check failed: {e}")
    
    # HTTP client
    try:
        _ = get_http_client()
        status["http_client"] = True
    except Exception as e:
        logger.error(f"HTTP client check failed: {e}")
    
    return status

# ============================================================================
# EXPOSED API
# ============================================================================

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
    "get_curator",
    "cleanup_old_backups",
    "check_dependencies_ready",
    "check_available_memory",
]