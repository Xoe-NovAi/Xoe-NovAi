#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - pytest Configuration (FIXED)
# ============================================================================
# Purpose: Shared fixtures and pytest configuration
# Guide Reference: Section 11 (Testing Infrastructure)
# Last Updated: 2025-10-18
# CRITICAL FIX: Added all missing fixtures (mock_redis, mock_crawler, etc.)
# ============================================================================

import pytest
import sys
import os
import toml
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "XNAi_rag_app"))

# ============================================================================
# SESSION-SCOPED FIXTURES (created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Session-scoped test configuration."""
    return {
        'metadata': {
            'stack_version': 'v0.1.2-test',
            'codename': 'Test Build',
            'architecture': 'testing'
        },
        'performance': {
            'memory_limit_gb': 6.0,
            'memory_warning_threshold_gb': 5.5,
            'memory_critical_threshold_gb': 5.8,
            'cpu_threads': 6,
            'f16_kv_enabled': True,
            'token_rate_min': 15,
            'token_rate_target': 20,
            'token_rate_max': 25,
            'latency_target_ms': 1000,
            'per_doc_chars': 500,
            'total_chars': 2048,
        },
        'models': {
            'llm_path': '/models/test-model.gguf',
            'embedding_path': '/embeddings/test-embed.gguf',
            'embedding_dimensions': 384,
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'cors_origins': ['http://localhost:8001'],
        },
        'redis': {
            'host': 'redis',
            'port': 6379,
            'cache': {
                'enabled': True,
                'ttl_seconds': 3600,
            }
        },
        'backup': {
            'faiss': {
                'enabled': True,
                'retention_days': 7,
            }
        },
        'logging': {
            'level': 'INFO',
            'file_path': '/app/logs/xnai.log',
        },
        'healthcheck': {
            'thresholds': {
                'memory_max_gb': 6.0,
            }
        },
        'crawl': {
            'rate_limit_per_min': 30,
            'sanitize_scripts': True,
        }
    }

# ============================================================================
# FUNCTION-SCOPED FIXTURES (recreated for each test)
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for unit tests."""
    mock = MagicMock()
    mock.invoke.return_value = "Mock response from LLM"
    mock.stream.return_value = iter(["Mock", " stream", " response"])
    mock.__call__.return_value = "Mock response"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    mock = MagicMock()
    # Return 384-dimensional vectors (all-MiniLM-L12-v2)
    mock.embed_query.return_value = [0.1] * 384
    mock.embed_documents.return_value = [[0.1] * 384] * 5
    return mock


@pytest.fixture
def mock_vectorstore():
    """Mock FAISS vectorstore."""
    mock = MagicMock()
    # Mock the index
    mock.index = MagicMock()
    mock.index.ntotal = 10  # 10 vectors in index
    
    # Mock similarity search
    from langchain_core.documents import Document
    mock.similarity_search.return_value = [
        Document(page_content="Test doc 1", metadata={"id": "1"}),
        Document(page_content="Test doc 2", metadata={"id": "2"}),
        Document(page_content="Test doc 3", metadata={"id": "3"}),
    ]
    
    # Mock add_documents
    mock.add_documents.return_value = None
    mock.save_local.return_value = None
    mock.load_local.return_value = mock
    
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.xadd.return_value = b'0-0'
    mock.info.return_value = {'redis_version': '7.4.1'}
    return mock


@pytest.fixture
def mock_crawler():
    """Mock CrawlModule/crawl4ai crawler."""
    mock = MagicMock()
    mock.warmup.return_value = None
    mock.crawl.return_value = [
        {
            'id': 'test_001',
            'content': 'Test crawled content 1',
            'metadata': {'source': 'test'}
        }
    ]
    return mock


@pytest.fixture
def mock_psutil():
    """Mock psutil Process for memory tests."""
    mock = MagicMock()
    memory_info = MagicMock()
    # 4GB memory usage
    memory_info.rss = 4 * 1024 ** 3
    mock.memory_info.return_value = memory_info
    return mock


# ============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ============================================================================

@pytest.fixture
def temp_library(tmp_path):
    """Create temporary library directory structure."""
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    
    # Create multiple categories with test documents
    categories = ["classics", "physics", "psychology", "technical-manuals", "esoteric"]
    for category in categories:
        cat_dir = lib_dir / category
        cat_dir.mkdir()
        
        # Create 5 documents per category
        for i in range(5):
            doc_file = cat_dir / f"doc_{i:04d}.txt"
            doc_file.write_text(
                f"Test document {i} in category {category}\n"
                f"This is sample content for testing purposes.\n"
                f"It contains multiple lines of text.\n"
            )
    
    return lib_dir


@pytest.fixture
def temp_knowledge(tmp_path):
    """Create temporary knowledge directory with curator metadata."""
    know_dir = tmp_path / "knowledge"
    know_dir.mkdir()
    
    # Create curator subdirectory with metadata index
    curator_dir = know_dir / "curator"
    curator_dir.mkdir()
    
    # Create metadata index.toml
    metadata = {
        f"doc_{i:04d}": {
            "source": "test",
            "category": "test",
            "timestamp": datetime.now().isoformat()
        }
        for i in range(5)
    }
    
    with open(curator_dir / "index.toml", "w") as f:
        toml.dump(metadata, f)
    
    # Create other agent knowledge directories (Phase 2)
    for agent in ["coder", "editor", "manager", "learner"]:
        (know_dir / agent).mkdir(exist_ok=True)
    
    return know_dir


@pytest.fixture
def temp_faiss_index(tmp_path):
    """Create temporary FAISS index directory."""
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    
    # Create mock FAISS files
    (index_dir / "index.faiss").write_bytes(b"MOCK_FAISS_INDEX_FILE")
    (index_dir / "index.pkl").write_bytes(b"MOCK_PICKLE_FILE")
    
    return index_dir


# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================

@pytest.fixture
def ryzen_env(monkeypatch):
    """Set Ryzen optimization environment variables."""
    env_vars = {
        "LLAMA_CPP_N_THREADS": "6",
        "LLAMA_CPP_F16_KV": "true",
        "LLAMA_CPP_USE_MLOCK": "true",
        "LLAMA_CPP_USE_MMAP": "true",
        "OPENBLAS_CORETYPE": "ZEN",
        "OMP_NUM_THREADS": "1",
        "MEMORY_LIMIT_GB": "6.0",
        "CHAINLIT_NO_TELEMETRY": "true",
        "CRAWL4AI_NO_TELEMETRY": "true",
        "LANGCHAIN_TRACING_V2": "false",
        "PYDANTIC_NO_TELEMETRY": "true",
        "FASTAPI_NO_TELEMETRY": "true",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def telemetry_env(monkeypatch):
    """Verify telemetry is fully disabled."""
    telemetry_vars = {
        "CHAINLIT_NO_TELEMETRY": "true",
        "CRAWL4AI_NO_TELEMETRY": "true",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_API_KEY": "",
        "PYDANTIC_NO_TELEMETRY": "true",
        "FASTAPI_NO_TELEMETRY": "true",
        "OPENAI_API_KEY": "",
        "SCARF_NO_ANALYTICS": "true",
    }
    
    for key, value in telemetry_vars.items():
        monkeypatch.setenv(key, value)
    
    return telemetry_vars


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest at startup."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: mark test as slow (integration tests)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "ryzen: mark test as Ryzen-specific")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow tests (integration, benchmarks)"
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="run performance benchmark tests"
    )
    parser.addoption(
        "--security",
        action="store_true",
        default=False,
        help="run security tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="need --slow option to run")
        for item in items:
            if "slow" in item.keywords or "integration" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--benchmark"):
        skip_bench = pytest.mark.skip(reason="need --benchmark option to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_bench)
    
    if not config.getoption("--security"):
        skip_sec = pytest.mark.skip(reason="need --security option to run")
        for item in items:
            if "security" in item.keywords:
                item.add_marker(skip_sec)


# ============================================================================
# ASSERTION HELPERS (for custom test assertions)
# ============================================================================

def assert_ryzen_config(env_vars):
    """Assert Ryzen optimization environment variables are set."""
    required_vars = {
        "LLAMA_CPP_N_THREADS": "6",
        "LLAMA_CPP_F16_KV": "true",
        "OPENBLAS_CORETYPE": "ZEN",
    }
    
    for key, expected in required_vars.items():
        actual = env_vars.get(key)
        assert actual == expected, f"{key}={actual}, expected {expected}"


def assert_telemetry_disabled(env_vars):
    """Assert all telemetry disables are properly configured."""
    required_disables = [
        ("CHAINLIT_NO_TELEMETRY", "true"),
        ("CRAWL4AI_NO_TELEMETRY", "true"),
        ("LANGCHAIN_TRACING_V2", "false"),
        ("PYDANTIC_NO_TELEMETRY", "true"),
        ("FASTAPI_NO_TELEMETRY", "true"),
    ]
    
    for key, expected in required_disables:
        actual = env_vars.get(key)
        assert actual == expected, f"Telemetry not disabled: {key}={actual}"


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Auto-cleanup after each test."""
    yield
    # pytest's tmp_path handles file cleanup automatically
    pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Session-level cleanup."""
    yield
    # Cleanup after all tests in session
    pass
