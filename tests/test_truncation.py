#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 rev_1.4 - Context Truncation Tests
# ============================================================================
# Purpose: Unit tests for context truncation logic in RAG pipeline
# Guide Reference: Section 11 (Testing Infrastructure)
# Last Updated: 2025-10-13
# Features:
#   - Test per_doc and total truncation limits
#   - Memory usage validation (<6GB target)
#   - Edge cases (empty docs, oversized context)
#   - Integration with Document objects
# ============================================================================

import pytest
import psutil
from pathlib import Path
from unittest.mock import Mock
from typing import List

# System under test
from ingest_library import _build_truncated_context
from langchain_core.documents import Document

# Configuration
from config_loader import load_config
CONFIG = load_config()

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(page_content="This is a short document.", metadata={"id": "doc1"}),
        Document(page_content="This is a longer document with more content that will be truncated during processing.", metadata={"id": "doc2"}),
        Document(page_content="Another short document for testing.", metadata={"id": "doc3"}),
        Document(page_content="Very long document content that exceeds truncation limits and will be cut off significantly.", metadata={"id": "doc4"}),
        Document(page_content="Final short document.", metadata={"id": "doc5"})
    ]

@pytest.fixture
def empty_documents() -> List[Document]:
    """Empty documents for edge case testing."""
    return [
        Document(page_content="", metadata={"id": "empty1"}),
        Document(page_content="", metadata={"id": "empty2"})
    ]

@pytest.fixture
def oversized_documents() -> List[Document]:
    """Oversized documents for truncation testing."""
    long_content = "A" * 2000  # Exceeds per_doc limit
    return [
        Document(page_content=long_content, metadata={"id": "oversized1"}),
        Document(page_content=long_content, metadata={"id": "oversized2"}),
        Document(page_content=long_content, metadata={"id": "oversized3"})
    ]

# ============================================================================
# UNIT TESTS
# ============================================================================

def test_build_truncated_context_basic(sample_documents):
    """
    Test basic truncation with default limits.
    
    Guide Reference: Section 11.2 (Truncation Tests)
    """
    # Defaults: per_doc=500, total=2048
    context = _build_truncated_context(sample_documents)
    
    # Check length
    assert len(context) <= 2048
    
    # Check all docs represented (at least partially)
    doc_contents = [doc.page_content for doc in sample_documents]
    for content in doc_contents:
        assert content[:100] in context  # First 100 chars of each doc
    
    # Check no excessive truncation
    assert len(context) > 500  # Should have substantial content

def test_build_truncated_context_per_doc_limit(sample_documents):
    """
    Test per-document truncation.
    
    Guide Reference: Section 11.2
    """
    context = _build_truncated_context(sample_documents, per_doc=100)
    
    # Each doc should be truncated to 100 chars
    doc_counts = [context.count(doc.page_content[:100]) for doc in sample_documents]
    for count in doc_counts:
        assert count == 1  # Each doc appears once, truncated
    
    assert len(context) <= 2048

def test_build_truncated_context_total_limit(sample_documents):
    """
    Test total context length limit.
    
    Guide Reference: Section 11.2
    """
    # Force total limit to be small
    context = _build_truncated_context(sample_documents, total=300)
    
    # Should truncate after ~1-2 docs
    assert len(context) <= 300
    assert context.count(sample_documents[0].page_content[:100]) == 1
    assert len(context.split('\n')) <= 3  # Limited docs

def test_build_truncated_context_empty_docs(empty_documents):
    """
    Test with empty documents.
    
    Guide Reference: Section 11.2 (Edge Cases)
    """
    context = _build_truncated_context(empty_documents)
    
    assert context == ""  # No content
    assert len(context) == 0

def test_build_truncated_context_oversized_docs(oversized_documents):
    """
    Test with oversized documents.
    
    Guide Reference: Section 11.2 (Memory Safety)
    """
    # Each doc is 2000 chars, per_doc=500
    context = _build_truncated_context(oversized_documents, per_doc=500)
    
    # First doc should be truncated to 500 chars
    assert context.startswith("A" * 500)
    assert len(context) <= 2048
    assert context.count("A" * 500) <= 4  # Max 4 full chunks

def test_build_truncated_context_memory_safety(oversized_documents):
    """
    Test memory usage during truncation.
    
    Guide Reference: Section 11.2 (<6GB Target)
    """
    memory_before = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
    
    context = _build_truncated_context(oversized_documents, per_doc=500, total=1000)
    
    memory_after = psutil.Process().memory_info().rss / (1024 ** 2)
    memory_delta = memory_after - memory_before
    
    # Should use minimal memory (<100MB for 3 docs)
    assert memory_delta < 100, f"Memory delta too high: {memory_delta}MB"
    
    logger.info(f"Memory test: +{memory_delta:.1f}MB for truncation")

def test_build_truncated_context_no_docs():
    """
    Test with no documents.
    
    Guide Reference: Section 11.2 (Edge Cases)
    """
    context = _build_truncated_context([])
    
    assert context == ""
    assert len(context) == 0

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_truncation_with_real_documents():
    """
    Test with real document-like content.
    
    Guide Reference: Section 11.3 (Integration)
    """
    docs = [
        Document(page_content="This is a test document for truncation validation.", metadata={"source": "test"}),
        Document(page_content="Another document with more content that might get truncated if too long.", metadata={"source": "test2"}),
        Document(page_content="Short document that fits easily within limits.", metadata={"source": "test3"})
    ]
    
    context = _build_truncated_context(docs, per_doc=200, total=400)
    
    # Should include parts of first 2 docs
    assert "test document" in context
    assert "might get truncated" in context
    assert "Short document" not in context  # Third doc truncated
    assert len(context) <= 400

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_truncation_performance():
    """
    Test truncation performance with large context.
    
    Guide Reference: Section 11.4 (Performance)
    """
    # Create 100 documents
    docs = [Document(page_content="A" * 1000, metadata={"id": i}) for i in range(100)]
    
    start_time = time.time()
    context = _build_truncated_context(docs, per_doc=500, total=2048)
    duration = time.time() - start_time
    
    # Should be fast (<50ms for 100 docs)
    assert duration < 0.05, f"Truncation too slow: {duration}s"
    
    # Should respect limits
    assert len(context) <= 2048
    assert context.count("A" * 500) <= 4  # Max 4 full chunks

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run all tests."""
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])