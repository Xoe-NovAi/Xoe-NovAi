# Xoe-NovAi v0.1.3-beta: Library Ingestion with Batch Checkpointing

**Section**: 11 (Data Ingestion + Pattern 4 Implementation)  
**Purpose**: Resilient document ingestion with crash recovery  
**Cross-References**: Artifact 1 (Pattern 4), Artifact 4 (Docker), Artifact 5 (Health Checks)

---

## 11.1 Ingestion Architecture

### 11.1.1 Problem Statement

**v0.1.2 Issue**: Long-running ingestion (1000+ documents) saved FAISS index only at completion. System crash at 80% progress = 100% data loss.

**v0.1.3 Solution**: Batch checkpointing (Pattern 4) saves progress incrementally every N documents, enabling crash recovery.

**Impact**:
- 0% data loss on crash (vs 100% in v0.1.2)
- Resume capability (continue from last checkpoint)
- Progress visibility (checkpoint count tracking)
- Target rate: 50-200 items/hour maintained

### 11.1.2 Checkpoint Strategy

```
Ingestion Flow with Checkpointing:
=====================================

Start Ingestion
  ↓
Load Existing FAISS Index (if present)
  ├─ Exists → Resume from checkpoint
  └─ None → Create new index
  ↓
Process Documents in Batches (size=100)
  ↓
[Batch 1: docs 0-99]
  ├─ Generate embeddings (384 dims each)
  ├─ Add to FAISS index
  └─ SAVE CHECKPOINT → disk
  ↓
[Batch 2: docs 100-199]
  ├─ Generate embeddings
  ├─ Add to existing index
  └─ SAVE CHECKPOINT → disk (overwrites batch 1)
  ↓
... repeat ...
  ↓
[CRASH at doc 450]
  ↓
RESUME (restart same command)
  ├─ Load Checkpoint 4 (docs 0-399 already in index)
  ├─ Skip docs 0-399 (already processed)
  └─ Continue from doc 400
  ↓
[Batch 5: docs 400-499]
  ├─ Add to existing index
  └─ SAVE CHECKPOINT
  ↓
Ingestion Complete
```

---

## 11.2 Implementation (ingest_library.py)

**File**: `scripts/ingest_library.py` (complete with Pattern 1 + Pattern 4)

```python
#!/usr/bin/env python3
"""
Xoe-NovAi v0.1.3-beta: Library Ingestion with Batch Checkpointing
Guide Ref: Section 11.2

Features:
- Batch checkpointing (Pattern 4): Save every N documents
- Crash recovery: Resume from last checkpoint
- Progress tracking: Checkpoint count, vector count, rate
- Parallel processing: ThreadPoolExecutor for embeddings
- Logging: JSON structured logs with performance metrics

Usage:
  python3 scripts/ingest_library.py --library-path /library --batch-size 100
  python3 scripts/ingest_library.py --library-path /library --force  # Rebuild from scratch
"""

# Pattern 1: Import Path Resolution (CRITICAL for container execution)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app' / 'XNAi_rag_app'))

import argparse
import time
import json
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings

from config_loader import load_config, get_config_value
from logging_config import get_logger

logger = get_logger(__name__)
CONFIG = load_config()

def load_documents(library_path: Path, file_types: List[str] = None) -> List[Document]:
    """
    Load all documents from library directory.
    
    Args:
        library_path: Root library path (e.g., /library)
        file_types: List of extensions (default: ['txt', 'md', 'json'])
    
    Returns:
        List of Document objects with metadata
    """
    if file_types is None:
        file_types = ['txt', 'md', 'json']
    
    documents = []
    
    for file_type in file_types:
        for file_path in library_path.rglob(f"*.{file_type}"):
            try:
                # Read content
                if file_type == 'json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = data.get('content', data.get('text', ''))
                        title = data.get('title', file_path.stem)
                else:
                    content = file_path.read_text(encoding='utf-8')
                    title = file_path.stem
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(file_path),
                        'title': title,
                        'category': file_path.parent.name,
                        'filename': file_path.name,
                        'file_type': file_type
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
    
    return documents

def ingest_library_with_checkpoints(
    library_path: str,
    batch_size: int = 100,
    force: bool = False,
    max_workers: int = 6
) -> Tuple[int, float]:
    """
    Ingest documents with automatic batch checkpointing (Pattern 4).
    
    Args:
        library_path: Root path to /library/ directory
        batch_size: Checkpoint frequency (default: 100 documents)
        force: If True, rebuild from scratch (ignore existing index)
        max_workers: ThreadPoolExecutor workers for parallel embedding
    
    Returns:
        Tuple of (total_documents_ingested, duration_seconds)
    
    Example:
        total, duration = ingest_library_with_checkpoints('/library', batch_size=100)
        # Output: Ingested 500 documents in 245.3s (122.4 docs/min)
    """
    # Guide Ref: Section 11.2 (Pattern 4: Batch Checkpointing)
    
    start_time = time.time()
    
    # =========================================================================
    # Step 1: Load embeddings model (retry-enabled in dependencies.py)
    # =========================================================================
    logger.info("Step 1: Loading embeddings model...")
    
    embedding_model_path = get_config_value('embedding_model_path', '/embeddings/all-MiniLM-L12-v2.Q8_0.gguf')
    
    embeddings = LlamaCppEmbeddings(
        model_path=embedding_model_path,
        n_threads=2,
        verbose=False
    )
    
    logger.info(f"✓ Embeddings loaded: {embedding_model_path}")
    
    # =========================================================================
    # Step 2: Define FAISS index path
    # =========================================================================
    index_path = Path('/app/XNAi_rag_app/faiss_index')
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Step 2: FAISS index path: {index_path}")
    
    # =========================================================================
    # Step 3: Load existing checkpoint OR create new (Pattern 4)
    # =========================================================================
    vectorstore = None
    initial_count = 0
    
    if index_path.exists() and not force:
        logger.info("Step 3: Checking for existing checkpoint...")
        try:
            vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            initial_count = vector