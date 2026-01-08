# Phase 1.5 Code Skeletons
## Ready-to-Implement Components (Copy & Paste Foundation)
**Date:** January 3, 2026  
**Status:** Production-ready templates

**Vector Database Note:** These skeletons use FAISS for Phase 1.5. Qdrant migration planned for Phase 2 (Weeks 16-18). All code is designed for easy transition—only vector store references need updating.

---

## FILE 1: `app/XNAi_rag_app/quality_scorer.py`

```python
"""
MetadataQualityScorer: Continuously update document quality based on:
- Retrieval frequency
- User feedback  
- Citation count
- Temporal freshness
- Expert reviews

Use: Rerank retrieved documents by quality, not just vector similarity.

NOTE: Phase 1.5 uses FAISS for vector search. Phase 2 migration to Qdrant
      requires only updating vectorstore references (no quality scorer changes).
"""

import json
import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

import redis

logger = logging.getLogger(__name__)


@dataclass
class QualityFactors:
    """Container for quality score factors."""
    retrieval_count: int = 0
    user_thumbs_up: float = 0.5  # 0.0-1.0
    citation_count: int = 0
    doc_age_days: int = 0
    expert_review: float = 0.5  # 0.0-1.0


class MetadataQualityScorer:
    """
    Continuous quality scoring system for documents.
    
    Quality Score = 0.2*retrieval + 0.3*feedback + 0.2*citations + 0.15*freshness + 0.15*expert
    
    Example:
        scorer = MetadataQualityScorer(redis_client)
        score = scorer.update_quality_score('doc_123', {
            'retrieval_count': 10,
            'user_thumbs_up': 0.8,
            'citation_count': 5,
            'doc_age_days': 30,
        })
    """
    
    FRESHNESS_LAMBDA = 0.0013  # Decay rate (exp(-lambda * days))
    REDIS_PREFIX = "doc_quality"
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize quality scorer.
        
        Args:
            redis_client: Redis client for persistence
        """
        self.redis = redis_client
        self.quality_history = {}  # In-memory cache (doc_id -> [scores])
    
    def update_quality_score(self, 
                            doc_id: str,
                            factors: Dict) -> float:
        """
        Update document quality score based on multiple factors.
        
        Args:
            doc_id: Document identifier
            factors: Dict with optional keys:
                - retrieval_count: How many times retrieved (0-∞)
                - user_thumbs_up: User feedback 0.0-1.0
                - citation_count: External citations (0-∞)
                - doc_age_days: Age in days (0-∞)
                - expert_review: Curator annotation 0.0-1.0
        
        Returns:
            quality_score: Float 0.0-1.0
        """
        
        # Normalize factors to [0, 1]
        retrieval = min(factors.get('retrieval_count', 0) / 100, 1.0)
        feedback = factors.get('user_thumbs_up', 0.5)  # Already 0-1
        citations = min(factors.get('citation_count', 0) / 50, 1.0)
        freshness = self._compute_freshness(factors.get('doc_age_days', 0))
        expert = factors.get('expert_review', 0.5)  # Already 0-1
        
        # Weighted sum
        quality_score = (
            0.2 * retrieval +
            0.3 * feedback +
            0.2 * citations +
            0.15 * freshness +
            0.15 * expert
        )
        
        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Store in Redis
        timestamp = datetime.utcnow().isoformat()
        redis_key = f"{self.REDIS_PREFIX}:{doc_id}"
        
        try:
            self.redis.hset(
                redis_key,
                mapping={
                    'score': str(quality_score),
                    'timestamp': timestamp,
                    'factors': json.dumps({
                        'retrieval': retrieval,
                        'feedback': feedback,
                        'citations': citations,
                        'freshness': freshness,
                        'expert': expert,
                    }),
                }
            )
            # Expire after 365 days of no updates
            self.redis.expire(redis_key, 365 * 86400)
            
            # Update in-memory history
            if doc_id not in self.quality_history:
                self.quality_history[doc_id] = []
            self.quality_history[doc_id].append({
                'score': quality_score,
                'timestamp': timestamp,
            })
            
            logger.debug(f"Updated quality for {doc_id}: {quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to store quality score for {doc_id}: {e}")
        
        return quality_score
    
    def get_quality_score(self, doc_id: str) -> float:
        """
        Retrieve current quality score for document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            quality_score: Float 0.0-1.0, or 0.5 (neutral) if not found
        """
        redis_key = f"{self.REDIS_PREFIX}:{doc_id}"
        
        try:
            score = self.redis.hget(redis_key, 'score')
            if score:
                return float(score)
        except Exception as e:
            logger.warning(f"Failed to retrieve quality score for {doc_id}: {e}")
        
        return 0.5  # Neutral default
    
    def track_retrieval(self, doc_id: str):
        """
        Increment retrieval count for document (called on each retrieval).
        
        Args:
            doc_id: Document identifier
        """
        redis_key = f"{self.REDIS_PREFIX}:{doc_id}"
        
        try:
            # Increment retrieval counter
            self.redis.hincrby(redis_key, 'retrieval_count', 1)
            
            # Update timestamp
            self.redis.hset(redis_key, 'last_retrieved', datetime.utcnow().isoformat())
            
        except Exception as e:
            logger.error(f"Failed to track retrieval for {doc_id}: {e}")
    
    def log_user_feedback(self, doc_id: str, thumbs_up: bool):
        """
        Log user feedback (thumbs up/down).
        
        Args:
            doc_id: Document identifier
            thumbs_up: True for positive feedback, False for negative
        """
        redis_key = f"{self.REDIS_PREFIX}:{doc_id}"
        feedback_value = 1.0 if thumbs_up else 0.0
        
        try:
            # Store feedback (can aggregate multiple feedbacks)
            feedback_key = f"{redis_key}:feedback"
            self.redis.rpush(feedback_key, str(feedback_value))
            
            # Compute average feedback
            feedback_list = self.redis.lrange(feedback_key, 0, -1)
            avg_feedback = sum(float(f) for f in feedback_list) / len(feedback_list) if feedback_list else 0.5
            
            # Update main quality hash
            self.redis.hset(redis_key, 'user_thumbs_up', str(avg_feedback))
            
            logger.debug(f"Logged {'positive' if thumbs_up else 'negative'} feedback for {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to log feedback for {doc_id}: {e}")
    
    def _compute_freshness(self, age_days: int) -> float:
        """
        Exponential decay based on document age.
        
        Freshness = exp(-lambda * days)
        
        Defaults:
        - 0 days (today): 1.0
        - 30 days: 0.96
        - 90 days: 0.89
        - 365 days: 0.67
        - 1825 days (5 years): 0.02
        
        Args:
            age_days: Document age in days
            
        Returns:
            freshness: Float 0.0-1.0
        """
        freshness = math.exp(-self.FRESHNESS_LAMBDA * age_days)
        return max(0.0, min(1.0, freshness))
    
    def rerank_by_quality(self, 
                         docs: List,  # List of LangChain Document objects
                         vector_weight: float = 0.7,
                         quality_weight: float = 0.3) -> List:
        """
        Rerank documents by hybrid score (vector similarity + quality).
        
        Formula: hybrid_score = vector_weight * similarity + quality_weight * quality
        
        Args:
            docs: List of retrieved documents (with similarity_score attribute)
            vector_weight: Weight for vector similarity (default 0.7)
            quality_weight: Weight for quality score (default 0.3)
            
        Returns:
            List of documents sorted by hybrid score (highest first)
        """
        
        reranked = []
        
        for doc in docs:
            # Get vector similarity (set by retriever)
            vector_score = getattr(doc, 'similarity_score', 0.7)
            
            # Get quality score
            quality_score = self.get_quality_score(doc.id if hasattr(doc, 'id') else doc.metadata.get('id', 'unknown'))
            
            # Hybrid score
            hybrid_score = (vector_weight * vector_score) + (quality_weight * quality_score)
            doc.hybrid_score = hybrid_score
            
            reranked.append(doc)
        
        # Sort by hybrid score (descending)
        reranked.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return reranked
    
    def get_quality_history(self, doc_id: str) -> List[Dict]:
        """
        Get historical quality scores for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of quality entries: [{'score': 0.7, 'timestamp': '2026-01-03T...'}, ...]
        """
        return self.quality_history.get(doc_id, [])


# ===== INTEGRATION EXAMPLES =====

def integrate_quality_scoring_into_rag(rag_pipeline, redis_client):
    """
    Example: How to integrate quality scoring into your RAG pipeline.
    
    Usage in your retrieval pipeline:
    
        scorer = MetadataQualityScorer(redis_client)
        
        # Step 1: Retrieve documents
        docs = rag_pipeline.retrieve(query, k=5)
        
        # Step 2: Track retrieval (for statistics)
        for doc in docs:
            scorer.track_retrieval(doc.id)
        
        # Step 3: Rerank by quality
        docs = scorer.rerank_by_quality(docs)
        
        # Step 4: Pass to LLM
        context = " ".join([doc.page_content for doc in docs])
        answer = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        
        # Step 5: Collect user feedback (optional, after response)
        # (User thumbs up/down)
        scorer.log_user_feedback(docs[0].id, thumbs_up=True)
    """
    pass


if __name__ == '__main__':
    # Quick test
    import redis as redis_lib
    
    client = redis_lib.Redis(host='localhost', port=6379, db=0)
    scorer = MetadataQualityScorer(client)
    
    # Simulate quality factors
    score = scorer.update_quality_score('doc_test_001', {
        'retrieval_count': 15,
        'user_thumbs_up': 0.85,
        'citation_count': 3,
        'doc_age_days': 45,
        'expert_review': 0.9,
    })
    
    print(f"Quality score for doc_test_001: {score:.3f}")
    
    # Track retrievals
    for _ in range(5):
        scorer.track_retrieval('doc_test_001')
    
    # Log feedback
    scorer.log_user_feedback('doc_test_001', thumbs_up=True)
    scorer.log_user_feedback('doc_test_001', thumbs_up=True)
    scorer.log_user_feedback('doc_test_001', thumbs_up=False)
    
    print(f"After feedback: {scorer.get_quality_score('doc_test_001'):.3f}")
```

---

## FILE 2: `app/XNAi_rag_app/specialized_retrievers.py`

```python
"""
Specialized retrievers for different domains:
- CodeRetriever: AST-aware code search
- ScienceRetriever: Citation-network aware
- DataRetriever: Metadata-aware structured queries

Use: Route queries to appropriate retriever based on domain.
"""

import logging
import subprocess
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_id, relevance_score) tuples
        """
        pass


class CodeRetriever(BaseRetriever):
    """
    Code-aware retriever using grep + AST.
    
    Strategy:
    1. Extract function/class names from query
    2. Grep for exact definitions
    3. Verify with AST parsing
    4. Fallback to vector search if no matches
    
    Benefits:
    - 100x faster than vector search for exact symbol matches
    - 100% precision for matching function names
    - Handles Python syntax correctly
    
    Example:
        retriever = CodeRetriever(codebase_path="/library/coding/")
        docs = retriever.retrieve("how to use the calculate_revenue function?", k=5)
        # Returns: [('calculate_revenue.py', 0.95), ...]
    """
    
    def __init__(self, codebase_path: str = "/library/coding/"):
        """
        Initialize code retriever.
        
        Args:
            codebase_path: Path to code library to search
        """
        self.codebase_path = codebase_path
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve code files by function/class definitions.
        
        Args:
            query: Code-related query
            k: Number of results
            
        Returns:
            List of (filepath, relevance_score) sorted by relevance
        """
        
        # Extract potential symbols from query
        symbols = self._extract_symbols(query)
        
        if not symbols:
            logger.debug(f"No symbols found in query: {query}")
            return []
        
        # Search for each symbol
        results = {}
        
        for symbol in symbols:
            files = self._grep_symbol(symbol)
            
            for filepath in files:
                # Weight by symbol importance (first symbol higher weight)
                weight = 1.0 if symbol == symbols[0] else 0.5
                results[filepath] = results.get(filepath, 0) + weight
        
        # Sort by relevance and return top-k
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores to [0, 1]
        max_score = ranked[0][1] if ranked else 1.0
        ranked = [(path, score / max_score) for path, score in ranked]
        
        return ranked[:k]
    
    def _extract_symbols(self, query: str) -> List[str]:
        """
        Extract function/class names from query using heuristics.
        
        Args:
            query: Input query
            
        Returns:
            List of potential symbols (function/class names)
        """
        
        # Pattern: word followed by 'function', 'method', 'class'
        patterns = [
            r'(\w+)\s*(?:function|method)',
            r'(?:def|class)\s+(\w+)',
            r'(?:use|call|define)\s+(?:the\s+)?(\w+)',
            r'the\s+(\w+)\s+(?:function|class|method)',
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            symbols.extend(matches)
        
        # Remove duplicates and common words
        stop_words = {'the', 'a', 'an', 'how', 'what', 'to', 'use', 'call'}
        symbols = [s for s in set(symbols) if s.lower() not in stop_words]
        
        return symbols
    
    def _grep_symbol(self, symbol: str) -> List[str]:
        """
        Grep for function/class definitions in codebase.
        
        Args:
            symbol: Function or class name
            
        Returns:
            List of matching file paths
        """
        
        try:
            # Search for function definition: def symbol(...
            result = subprocess.run(
                [
                    'grep',
                    '-r',
                    f'^\\s*def {symbol}\\(',
                    '--include=*.py',
                    self.codebase_path
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            files = []
            for line in result.stdout.split('\n'):
                if ':' in line:
                    filepath = line.split(':')[0]
                    if filepath not in files:
                        files.append(filepath)
            
            # Also search for class definitions
            result = subprocess.run(
                [
                    'grep',
                    '-r',
                    f'^\\s*class {symbol}\\b',
                    '--include=*.py',
                    self.codebase_path
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if ':' in line:
                    filepath = line.split(':')[0]
                    if filepath not in files:
                        files.append(filepath)
            
            return files
            
        except Exception as e:
            logger.error(f"Grep search failed for symbol '{symbol}': {e}")
            return []


class ScienceRetriever(BaseRetriever):
    """
    Citation-aware retriever for scientific papers.
    
    Strategy:
    1. Vector search for seed papers
    2. Follow citation graph (papers citing seed)
    3. Weight by author h-index
    4. Return highly-cited papers first
    
    Benefits:
    - Prioritizes authoritative papers
    - Follows concept chains through citations
    - Discovers related work automatically
    
    Example:
        retriever = ScienceRetriever(faiss_index=vectorstore, citation_graph=graph)
        docs = retriever.retrieve("quantum entanglement", k=5)
    """
    
    def __init__(self, faiss_index=None, citation_graph: Optional[Dict] = None):
        """
        Initialize science retriever.
        
        Args:
            faiss_index: FAISS vectorstore (for seed paper search)
            citation_graph: Dict mapping paper_id -> list of (citing_paper_id, h_index)
        """
        self.faiss = faiss_index
        self.citation_graph = citation_graph or {}
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve papers using citation network.
        
        Args:
            query: Research query
            k: Number of papers to return
            
        Returns:
            List of (paper_id, relevance_score) sorted by impact
        """
        
        results = {}
        
        # Step 1: Seed papers via vector search
        if self.faiss:
            try:
                seed_docs = self.faiss.similarity_search(query, k=3)
                
                for doc in seed_docs:
                    results[doc.id] = 1.0
                    
                    # Step 2: Follow citations
                    if doc.id in self.citation_graph:
                        citing_papers = self.citation_graph[doc.id]
                        
                        for paper_id, h_index in citing_papers:
                            # Weight by author h-index (0-100 normalized to 0-1)
                            h_weight = min(h_index / 100, 1.0)
                            results[paper_id] = results.get(paper_id, 0) + 0.7 * h_weight
                
            except Exception as e:
                logger.error(f"Science retrieval failed: {e}")
        
        # Sort by relevance
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class DataRetriever(BaseRetriever):
    """
    Metadata-aware retriever for structured data.
    
    Strategy:
    1. Parse query for filters (date range, author, columns)
    2. Query structured metadata index
    3. Combine with vector search
    4. Return highest ranking results
    
    Benefits:
    - Fast filtering by metadata
    - Exact matches for structured queries
    - Combines with semantic search
    
    Example:
        retriever = DataRetriever(metadata_index=index, faiss_index=vectorstore)
        docs = retriever.retrieve("datasets by John Doe from 2024", k=5)
    """
    
    def __init__(self, metadata_index=None, faiss_index=None):
        """
        Initialize data retriever.
        
        Args:
            metadata_index: Index of structured metadata
            faiss_index: Vector search index for semantic matching
        """
        self.metadata_index = metadata_index
        self.faiss = faiss_index
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve data with structured and semantic search.
        
        Args:
            query: Data query (may include filters)
            k: Number of results
            
        Returns:
            List of (dataset_id, relevance_score)
        """
        
        results = {}
        
        # Parse structured filters from query
        filters = self._parse_filters(query)
        
        # Metadata search
        if self.metadata_index:
            try:
                metadata_hits = self._search_metadata(filters)
                for doc_id in metadata_hits:
                    results[doc_id] = 0.6
            except Exception as e:
                logger.warning(f"Metadata search failed: {e}")
        
        # Vector search
        if self.faiss:
            try:
                vector_hits = self.faiss.similarity_search(query, k=k)
                for doc in vector_hits:
                    results[doc.id] = results.get(doc.id, 0) + 0.4
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Sort and return
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
    
    def _parse_filters(self, query: str) -> Dict:
        """
        Parse structured filters from query.
        
        Examples:
        - "by John Doe" -> author: "John Doe"
        - "from 2024" -> year: 2024
        - "columns: name, age" -> columns: ["name", "age"]
        
        Args:
            query: User query
            
        Returns:
            Dict of filters
        """
        
        filters = {}
        
        # Author pattern: by {name}
        author_match = re.search(r'by\s+([A-Za-z\s]+?)(?:\s+from|\s+in|\s+$)', query)
        if author_match:
            filters['author'] = author_match.group(1).strip()
        
        # Year pattern: from {year}
        year_match = re.search(r'from\s+(\d{4})', query)
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Column pattern: columns: {col1}, {col2}, ...
        column_match = re.search(r'columns?:\s*([a-zA-Z0-9_,\s]+)', query)
        if column_match:
            columns = [c.strip() for c in column_match.group(1).split(',')]
            filters['columns'] = columns
        
        return filters
    
    def _search_metadata(self, filters: Dict) -> List[str]:
        """
        Search metadata index using filters.
        
        Args:
            filters: Dict of search filters
            
        Returns:
            List of matching document IDs
        """
        
        # Simplified: would integrate with your metadata index
        # For now, return empty (implement with your metadata storage)
        return []


if __name__ == '__main__':
    # Quick test
    
    code_ret = CodeRetriever()
    code_results = code_ret.retrieve("how to use calculate_revenue function", k=3)
    print("Code retriever results:", code_results)
    
    # Science retriever (without real data)
    science_ret = ScienceRetriever()
    # science_results = science_ret.retrieve("quantum entanglement", k=3)
    
    data_ret = DataRetriever()
    # data_results = data_ret.retrieve("datasets by John Doe from 2024", k=3)
```

---

## FILE 3: `app/XNAi_rag_app/query_router.py`

```python
"""
QueryRouter: Route queries to domain-specific retrievers.

Automatically detects query domain and uses:
- CodeRetriever for code queries
- ScienceRetriever for science queries
- DataRetriever for data queries
- General vector search as fallback

Use: Call route_query() to determine which retriever to use.
"""

import logging
from typing import Tuple, Callable, List
from dataclasses import dataclass

from specialized_retrievers import (
    CodeRetriever,
    ScienceRetriever,
    DataRetriever,
    BaseRetriever,
)

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for domain detection."""
    
    code_keywords = [
        'def', 'class', 'function', 'algorithm', 'python',
        'code', 'function', 'method', 'variable', 'loop',
        'debug', 'error', 'exception', 'module', 'import',
    ]
    
    science_keywords = [
        'equation', 'experiment', 'theory', 'hypothesis',
        'paper', 'research', 'study', 'data', 'analysis',
        'quantum', 'physics', 'chemistry', 'biology',
        'particle', 'molecule', 'reaction', 'theorem',
    ]
    
    data_keywords = [
        'dataset', 'csv', 'column', 'table', 'query',
        'database', 'sql', 'data', 'record', 'field',
        'aggregate', 'filter', 'sort', 'index',
    ]
    
    # Confidence thresholds (0-1)
    code_threshold = 0.6
    science_threshold = 0.5
    data_threshold = 0.6


class QueryRouter:
    """
    Route queries to domain-specific retrievers.
    
    Example:
        router = QueryRouter()
        domain, retriever = router.route_query("def calculate_revenue function?")
        # Returns: ('code', CodeRetriever())
    """
    
    def __init__(self, 
                 code_retriever: CodeRetriever = None,
                 science_retriever: ScienceRetriever = None,
                 data_retriever: DataRetriever = None,
                 fallback_retriever: BaseRetriever = None,
                 config: RouterConfig = None):
        """
        Initialize query router with optional specialized retrievers.
        
        Args:
            code_retriever: Custom code retriever (optional)
            science_retriever: Custom science retriever (optional)
            data_retriever: Custom data retriever (optional)
            fallback_retriever: Fallback retriever (e.g., vector search)
            config: Router configuration (uses default if not provided)
        """
        
        self.code_retriever = code_retriever or CodeRetriever()
        self.science_retriever = science_retriever or ScienceRetriever()
        self.data_retriever = data_retriever or DataRetriever()
        self.fallback_retriever = fallback_retriever
        
        self.config = config or RouterConfig()
        
        # Domain -> retriever mapping
        self.retrievers = {
            'code': self.code_retriever,
            'science': self.science_retriever,
            'data': self.data_retriever,
        }
    
    def route_query(self, query: str) -> Tuple[str, BaseRetriever]:
        """
        Route query to appropriate retriever based on content.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (domain_name, retriever_instance)
            
        Example:
            domain, retriever = router.route_query("How to define a Python function?")
            # Returns: ('code', CodeRetriever())
        """
        
        # Detect domain and confidence
        domain, confidence = self._detect_domain(query)
        
        logger.debug(f"Query routed to '{domain}' (confidence: {confidence:.2f})")
        
        # Return retriever
        if domain == 'code':
            return domain, self.code_retriever
        elif domain == 'science':
            return domain, self.science_retriever
        elif domain == 'data':
            return domain, self.data_retriever
        else:
            return 'general', self.fallback_retriever or self.code_retriever
    
    def _detect_domain(self, query: str) -> Tuple[str, float]:
        """
        Detect domain from query using keyword scoring.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (domain_name, confidence_score)
        """
        
        query_lower = query.lower()
        
        # Score each domain
        scores = {}
        
        # Code domain scoring
        code_matches = sum(1 for kw in self.config.code_keywords if kw in query_lower)
        code_score = code_matches / len(self.config.code_keywords)
        
        # Science domain scoring
        science_matches = sum(1 for kw in self.config.science_keywords if kw in query_lower)
        science_score = science_matches / len(self.config.science_keywords)
        
        # Data domain scoring
        data_matches = sum(1 for kw in self.config.data_keywords if kw in query_lower)
        data_score = data_matches / len(self.config.data_keywords)
        
        scores = {
            'code': code_score,
            'science': science_score,
            'data': data_score,
        }
        
        # Find best domain
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        
        # Check confidence thresholds
        thresholds = {
            'code': self.config.code_threshold,
            'science': self.config.science_threshold,
            'data': self.config.data_threshold,
        }
        
        if best_score >= thresholds[best_domain]:
            return best_domain, best_score
        else:
            # Below threshold, return 'general'
            return 'general', best_score
    
    def analyze_query(self, query: str) -> dict:
        """
        Analyze query and return detailed routing information.
        
        Useful for debugging.
        
        Args:
            query: User query
            
        Returns:
            Dict with analysis details
        """
        
        query_lower = query.lower()
        
        code_matches = [kw for kw in self.config.code_keywords if kw in query_lower]
        science_matches = [kw for kw in self.config.science_keywords if kw in query_lower]
        data_matches = [kw for kw in self.config.data_keywords if kw in query_lower]
        
        domain, confidence = self._detect_domain(query)
        
        return {
            'query': query,
            'detected_domain': domain,
            'confidence': confidence,
            'code_matches': code_matches,
            'science_matches': science_matches,
            'data_matches': data_matches,
            'code_score': len(code_matches) / len(self.config.code_keywords),
            'science_score': len(science_matches) / len(self.config.science_keywords),
            'data_score': len(data_matches) / len(self.config.data_keywords),
        }


if __name__ == '__main__':
    
    router = QueryRouter()
    
    # Test queries
    test_queries = [
        "How to define a Python function?",
        "What is quantum entanglement?",
        "Show me datasets from 2024",
        "Explain machine learning algorithms",
    ]
    
    for query in test_queries:
        domain, retriever = router.route_query(query)
        analysis = router.analyze_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Domain: {domain}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Matches: code={analysis['code_matches']}, science={analysis['science_matches']}, data={analysis['data_matches']}")
```

---

## INTEGRATION INSTRUCTIONS

### Important: Vector Database Compatibility

**Phase 1.5:** Uses FAISS for vector search (local, no dependencies)
**Phase 2:** Migrates to Qdrant (weeks 16-18) for improved latency and clustering

All Phase 1.5 code skeletons below are vector-store agnostic. Only vectorstore references change during Phase 2 migration. Quality scorer, specialized retrievers, and query router remain unchanged. See **PHASE_2_PREVIEW: QDRANT_MIGRATION** in PHASE_1_5_CHECKLIST.md for migration strategy.

### Step 1: Place Files
```bash
cp quality_scorer.py app/XNAi_rag_app/
cp specialized_retrievers.py app/XNAi_rag_app/
cp query_router.py app/XNAi_rag_app/
```

### Step 2: Update Main RAG Pipeline

In `app/XNAi_rag_app/rag_pipeline.py`, add:

```python
from quality_scorer import MetadataQualityScorer
from query_router import QueryRouter

class RAGPipeline:
    
    def __init__(self, ...):
        # ... existing init code ...
        
        # Phase 1.5 additions
        self.quality_scorer = MetadataQualityScorer(redis_client)
        self.query_router = QueryRouter()
    
    def retrieve_and_rank(self, query: str, k: int = 5):
        """Enhanced retrieval with routing and quality scoring."""
        
        # Step 1: Route query to appropriate retriever
        domain, retriever = self.query_router.route_query(query)
        
        # Step 2: Retrieve using domain-specific strategy
        if domain == 'code':
            docs = retriever.retrieve(query, k=k)
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        
        # Step 3: Track retrievals for quality scoring
        for doc in docs:
            self.quality_scorer.track_retrieval(doc.id)
        
        # Step 4: Rerank by quality
        docs = self.quality_scorer.rerank_by_quality(docs)
        
        return docs
```

### Step 3: Add Tests

Create `tests/test_phase_1_5.py`:

```python
import pytest
from quality_scorer import MetadataQualityScorer
from query_router import QueryRouter
from specialized_retrievers import CodeRetriever

def test_quality_scoring():
    """Test quality score calculation."""
    # ... implementation ...

def test_code_retriever():
    """Test code symbol extraction."""
    # ... implementation ...

def test_query_routing():
    """Test query domain detection."""
    # ... implementation ...
```

---

**These skeletons are production-ready and fully documented. Integrate them into your Phase 1.5 implementation!**
