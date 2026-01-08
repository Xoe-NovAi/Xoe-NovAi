# Phase 2 & 3: Advanced Implementation
## Multi-Adapter Retrieval, Reranking, Monitoring, and Versioning
**Xoe-NovAi v0.1.4-stable** | **Estimated Duration:** Week 2-4+

---

# PHASE 2: Advanced Retrieval Architecture
## Multi-Adapter Retrieval + Relevance Reranking
**Estimated Duration:** Week 2-3 | **Impact:** 35-50% precision improvement

---

## PHASE 2 COMPONENTS

### Component 5: Multi-Adapter Retrieval (Hybrid Search)

**File:** `app/XNAi_rag_app/multi_adapter_retriever.py`

```python
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Unified retrieval result from any adapter"""
    content: str
    metadata: Dict[str, Any]
    score: float  # 0.0-1.0 normalized
    adapter: str  # 'semantic' | 'keyword' | 'structure'
    rank: int

class RetrievalAdapter(Enum):
    """Available retrieval strategies"""
    SEMANTIC = "semantic"      # Dense vector similarity
    KEYWORD = "keyword"        # BM25/sparse
    STRUCTURE = "structure"    # Document structure/hierarchy
    HYBRID = "hybrid"          # Combine multiple
    RERANKED = "reranked"      # Final ranking with LLM

class MultiAdapterRetriever:
    """
    Retrieve chunks using multiple strategies, then intelligently combine
    
    Strategy 1: Semantic (vector similarity) - Best for conceptual questions
    Strategy 2: Keyword (BM25) - Best for specific term queries
    Strategy 3: Structure (heading hierarchy) - Best for navigating docs
    Strategy 4: Hybrid - Combine all with weighting
    """
    
    def __init__(self, 
                 faiss_index,  # Vector store
                 bm25_index,   # Keyword index
                 metadata_db,  # Redis or similar
                 llm_reranker=None):
        self.faiss = faiss_index
        self.bm25 = bm25_index
        self.metadata_db = metadata_db
        self.llm_reranker = llm_reranker
    
    def retrieve(self, 
                 query: str,
                 top_k: int = 10,
                 strategy: RetrievalAdapter = RetrievalAdapter.HYBRID,
                 domain: str = None) -> List[RetrievalResult]:
        """Retrieve using specified strategy"""
        
        if strategy == RetrievalAdapter.SEMANTIC:
            return self._retrieve_semantic(query, top_k, domain)
        elif strategy == RetrievalAdapter.KEYWORD:
            return self._retrieve_keyword(query, top_k, domain)
        elif strategy == RetrievalAdapter.STRUCTURE:
            return self._retrieve_structure(query, top_k, domain)
        elif strategy == RetrievalAdapter.HYBRID:
            return self._retrieve_hybrid(query, top_k, domain)
        elif strategy == RetrievalAdapter.RERANKED:
            return self._retrieve_reranked(query, top_k, domain)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _retrieve_semantic(self, query: str, top_k: int, domain: str) -> List[RetrievalResult]:
        """Retrieve using vector similarity (dense retrieval)"""
        
        # Encode query to embedding
        query_embedding = self.faiss.get_embeddings().embed_query(query)
        
        # Search FAISS
        scores, indices = self.faiss.index.search(
            np.array([query_embedding], dtype='float32'),
            k=top_k
        )
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # Invalid result
                continue
            
            normalized_score = float(score)
            chunk_data = self.metadata_db.hgetall(f"chunk:{idx}")
            
            if domain and chunk_data.get('domain') != domain:
                continue
            
            results.append(RetrievalResult(
                content=chunk_data.get('content', ''),
                metadata=chunk_data,
                score=normalized_score,
                adapter=RetrievalAdapter.SEMANTIC.value,
                rank=len(results) + 1
            ))
        
        return results[:top_k]
    
    def _retrieve_keyword(self, query: str, top_k: int, domain: str) -> List[RetrievalResult]:
        """Retrieve using keyword matching (BM25)"""
        
        bm25_results = self.bm25.retrieve(query, k=top_k)
        
        results = []
        for bm25_result in bm25_results:
            normalized_score = min(1.0, bm25_result.score / 10.0)
            
            results.append(RetrievalResult(
                content=bm25_result.page_content,
                metadata=bm25_result.metadata,
                score=normalized_score,
                adapter=RetrievalAdapter.KEYWORD.value,
                rank=len(results) + 1
            ))
        
        return results[:top_k]
    
    def _retrieve_structure(self, query: str, top_k: int, domain: str) -> List[RetrievalResult]:
        """Retrieve by document structure (code examples, definitions, etc)"""
        
        intent = self._classify_query_intent(query)
        
        if intent == 'how':
            section_types = ['code', 'example', 'tutorial']
        elif intent == 'what':
            section_types = ['definition', 'concept', 'explanation']
        elif intent == 'why':
            section_types = ['rationale', 'motivation', 'background']
        else:
            section_types = None
        
        results = []
        
        if section_types:
            cursor = self.metadata_db.scan_iter(f"chunk:*")
            for key in cursor:
                chunk_data = self.metadata_db.hgetall(key)
                
                if chunk_data.get('chunk_type') in section_types:
                    if query.lower() in chunk_data.get('content', '').lower():
                        results.append(RetrievalResult(
                            content=chunk_data.get('content', ''),
                            metadata=chunk_data,
                            score=0.7,
                            adapter=RetrievalAdapter.STRUCTURE.value,
                            rank=len(results) + 1
                        ))
        
        return results[:top_k]
    
    def _retrieve_hybrid(self, query: str, top_k: int, domain: str) -> List[RetrievalResult]:
        """Hybrid retrieval: combine semantic + keyword + structure with weights"""
        
        semantic_results = self._retrieve_semantic(query, top_k, domain)
        keyword_results = self._retrieve_keyword(query, top_k, domain)
        structure_results = self._retrieve_structure(query, top_k, domain)
        
        all_results = {}
        
        for result in semantic_results:
            chunk_id = result.metadata.get('chunk_id', id(result))
            if chunk_id not in all_results:
                all_results[chunk_id] = {'content': result.content, 'metadata': result.metadata}
            all_results[chunk_id]['semantic_score'] = result.score
        
        for result in keyword_results:
            chunk_id = result.metadata.get('chunk_id', id(result))
            if chunk_id not in all_results:
                all_results[chunk_id] = {'content': result.content, 'metadata': result.metadata}
            all_results[chunk_id]['keyword_score'] = result.score
        
        for result in structure_results:
            chunk_id = result.metadata.get('chunk_id', id(result))
            if chunk_id not in all_results:
                all_results[chunk_id] = {'content': result.content, 'metadata': result.metadata}
            all_results[chunk_id]['structure_score'] = result.score
        
        hybrid_results = []
        for chunk_id, data in all_results.items():
            semantic_score = data.get('semantic_score', 0.0)
            keyword_score = data.get('keyword_score', 0.0)
            structure_score = data.get('structure_score', 0.0)
            
            # WEIGHTS (tunable):
            hybrid_score = (
                0.50 * semantic_score +   # Conceptual understanding
                0.30 * keyword_score +    # Term matching
                0.20 * structure_score    # Navigation
            )
            
            hybrid_results.append(RetrievalResult(
                content=data['content'],
                metadata=data['metadata'],
                score=hybrid_score,
                adapter=RetrievalAdapter.HYBRID.value,
                rank=0
            ))
        
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(hybrid_results[:top_k]):
            result.rank = i + 1
        
        return hybrid_results[:top_k]
    
    def _retrieve_reranked(self, query: str, top_k: int, domain: str) -> List[RetrievalResult]:
        """Final reranking using LLM judgment"""
        
        if not self.llm_reranker:
            return self._retrieve_hybrid(query, top_k, domain)
        
        # Get hybrid candidates (retrieve more to have good pool)
        candidates = self._retrieve_hybrid(query, top_k * 2, domain)
        reranked = self.llm_reranker.rerank(query, candidates, k=top_k)
        
        for i, result in enumerate(reranked):
            result.rank = i + 1
            result.adapter = RetrievalAdapter.RERANKED.value
        
        return reranked
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify query intent: 'how', 'what', 'why', 'list', 'compare'"""
        
        query_lower = query.lower()
        
        if query_lower.startswith(('how to', 'how do', 'how can')):
            return 'how'
        elif query_lower.startswith(('what is', 'what are', 'define')):
            return 'what'
        elif query_lower.startswith(('why', 'explain')):
            return 'why'
        elif query_lower.startswith(('list', 'show', 'give me')):
            return 'list'
        elif query_lower.startswith(('compare', 'difference', 'vs')):
            return 'compare'
        else:
            return 'general'
```

### Component 6: LLM-based Reranking

**File:** `app/XNAi_rag_app/llm_reranker.py`

```python
import json
from typing import List
from langchain.llms import BaseLLM
import logging

logger = logging.getLogger(__name__)

class LLMReranker:
    """Use LLM to intelligently rerank retrieval results by relevance"""
    
    def __init__(self, llm: BaseLLM, model_name: str = "gpt-4"):
        self.llm = llm
        self.model_name = model_name
    
    def rerank(self, 
               query: str,
               results: List['RetrievalResult'],
               k: int = 10) -> List['RetrievalResult']:
        """
        Rerank results using LLM judgment
        
        Prompt LLM to evaluate each result's relevance to query
        """
        
        candidates_text = self._format_candidates(results)
        
        prompt = f"""
        Evaluate the relevance of each document to the user's query on a scale of 0-10.
        Higher scores indicate better relevance. Consider:
        - Direct answer to the question
        - Authoritative source
        - Clarity and completeness
        - Recency if applicable
        
        QUERY: {query}
        
        CANDIDATES:
        {candidates_text}
        
        For each candidate (by index), assign a relevance score from 0 to 10.
        Return JSON: {{"0": 9, "1": 7, "2": 5, ...}}
        
        Output ONLY valid JSON, no explanation.
        """
        
        response = self.llm.predict(prompt)
        
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM reranking response, using original scores")
            return results[:k]
        
        # Apply LLM scores
        for i, result in enumerate(results):
            llm_score = scores.get(str(i), result.score * 10)
            result.score = min(1.0, llm_score / 10.0)
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    def _format_candidates(self, results: List['RetrievalResult']) -> str:
        """Format candidates for LLM evaluation"""
        
        text = ""
        for i, result in enumerate(results):
            text += f"\n{i}. [{result.metadata.get('title', 'Untitled')}]\n"
            text += f"   Source: {result.metadata.get('source', 'unknown')}\n"
            text += f"   Type: {result.adapter}\n"
            text += f"   {result.content[:150]}...\n"
        
        return text
```

---

## PHASE 2 INTEGRATION FLOW

```
USER QUERY
    ↓
CLASSIFY INTENT (how / what / why / compare)
    ↓
MULTI-ADAPTER RETRIEVAL
    ├─ Semantic: Dense vector similarity (50% weight)
    ├─ Keyword: BM25 term matching (30% weight)
    ├─ Structure: Document hierarchy (20% weight)
    └─ Hybrid: Weighted combination
        ↓
    Top 20 candidates (scored 0.0-1.0)
    ↓
LLM RERANKING (Optional)
    (Evaluate relevance with LLM judgment)
    ↓
TOP 10 RESULTS (Ranked by relevance)
    ↓
RESPONSE GENERATION
    (Ground response in top results)
    ↓
GROUNDEDNESS CHECKING
    (Verify claims against sources)
```

---

## PHASE 2 EXPECTED OUTCOMES

After implementing Phase 2:

✅ **Multi-Adapter Retrieval:**
- 3 independent retrieval strategies
- Hybrid weighting (semantic + keyword + structure)
- Better support for different query types (how-to, definitions, comparisons, etc)

✅ **LLM Reranking:**
- Intelligent result ordering using LLM judgment
- Better alignment with user intent
- Reduced false positives in final results

✅ **Combined Impact:**
- 35-50% improvement in retrieval precision vs Phase 1
- Better answers for conceptual vs specific queries
- More grounded and accurate response generation

---

---

# PHASE 3: Production Operations
## Monitoring, Versioning, and Observability
**Estimated Duration:** Week 4+ | **Impact:** Operational reliability + data quality

---

## PHASE 3 COMPONENTS

### Component 7: Ingestion Pipeline Monitoring

**File:** `app/XNAi_rag_app/ingestion_monitor.py`

```python
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class IngestionMonitor:
    """Monitor ingestion pipeline health and performance"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def log_ingestion_event(self, file_path: str, status: str, metadata: dict = None):
        """Log ingestion event for monitoring and troubleshooting"""
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'file': file_path,
            'status': status,  # 'started', 'completed', 'failed'
            'metadata': metadata or {},
        }
        
        self.redis.rpush("ingestion:events", json.dumps(event))
        logger.info(f"[Ingestion] {file_path}: {status}")
    
    def get_ingestion_stats(self, hours: int = 24) -> dict:
        """Get ingestion statistics for last N hours"""
        
        events = []
        for event_str in self.redis.lrange("ingestion:events", 0, -1):
            try:
                event = json.loads(event_str)
                events.append(event)
            except json.JSONDecodeError:
                continue
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [
            e for e in events 
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]
        
        completed = len([e for e in recent_events if e['status'] == 'completed'])
        failed = len([e for e in recent_events if e['status'] == 'failed'])
        total = completed + failed
        
        return {
            'total_ingested': completed,
            'total_failed': failed,
            'success_rate': completed / total if total > 0 else 0.0,
            'time_period_hours': hours,
        }
```

### Component 8: Knowledge Base Versioning

**File:** `scripts/version_knowledge_base.py`

```python
from pathlib import Path
import tarfile
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseVersioning:
    """Track versions of knowledge bases for rollback and history"""
    
    def __init__(self, knowledge_base_path: Path):
        self.kb_path = knowledge_base_path
        self.versions_path = Path(f"{knowledge_base_path.parent}/versions")
        self.versions_path.mkdir(parents=True, exist_ok=True)
    
    def create_version(self, version_tag: str, domain: str = None):
        """Create snapshot of knowledge base at specific version"""
        
        domain_path = self.kb_path if not domain else self.kb_path / domain
        
        if not domain_path.exists():
            logger.warning(f"Knowledge base path does not exist: {domain_path}")
            return None
        
        version_path = self.versions_path / version_tag
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Archive knowledge base
        archive_path = version_path / "snapshot.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(domain_path, arcname=domain or domain_path.name)
        
        # Store metadata
        version_meta = {
            'timestamp': datetime.utcnow().isoformat(),
            'version_tag': version_tag,
            'domain': domain,
            'file_count': len(list(domain_path.glob('**/*'))),
            'archive_size_mb': archive_path.stat().st_size / (1024 * 1024),
        }
        
        meta_path = version_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(version_meta, f, indent=2)
        
        logger.info(f"✅ Created version {version_tag}: {archive_path}")
        return version_meta
    
    def list_versions(self) -> list:
        """List all available versions"""
        
        versions = []
        for version_dir in sorted(self.versions_path.iterdir()):
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    versions.append(meta)
        
        return versions
    
    def rollback_to_version(self, version_tag: str):
        """Rollback to a previous version"""
        
        version_path = self.versions_path / version_tag
        archive_path = version_path / "snapshot.tar.gz"
        
        if not archive_path.exists():
            logger.error(f"Version not found: {version_tag}")
            return False
        
        # Backup current version first
        backup_tag = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.create_version(backup_tag)
        
        # Extract and restore
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.kb_path.parent)
        
        logger.info(f"✅ Rolled back to version {version_tag}")
        return True
```

### Component 9: Data Quality Metrics (Prometheus)

**File:** `app/XNAi_rag_app/quality_metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

# Define metrics
ingestion_success = Counter(
    'ingestion_success_total',
    'Total successful ingestions',
    ['domain']
)

ingestion_failed = Counter(
    'ingestion_failed_total',
    'Total failed ingestions',
    ['domain']
)

ingestion_duration = Histogram(
    'ingestion_duration_seconds',
    'Time to ingest document',
    ['domain']
)

chunk_quality_score = Gauge(
    'chunk_quality_score',
    'Average quality score of chunks',
    ['domain']
)

retrieval_latency = Histogram(
    'retrieval_latency_seconds',
    'Time to retrieve results',
    ['strategy'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

groundedness_score = Histogram(
    'groundedness_score',
    'Groundedness score of responses',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

knowledge_base_staleness = Gauge(
    'knowledge_base_staleness_hours',
    'Hours since last update to knowledge base',
    ['domain']
)

# Usage:
def record_ingestion_metric(domain: str, success: bool, duration_seconds: float):
    """Record ingestion metrics"""
    if success:
        ingestion_success.labels(domain=domain).inc()
    else:
        ingestion_failed.labels(domain=domain).inc()
    
    ingestion_duration.labels(domain=domain).observe(duration_seconds)

def record_retrieval_metric(strategy: str, latency_seconds: float):
    """Record retrieval metrics"""
    retrieval_latency.labels(strategy=strategy).observe(latency_seconds)

def record_groundedness_metric(score: float):
    """Record groundedness score"""
    groundedness_score.observe(score)
```

---

## PHASE 3 PRODUCTION CHECKLIST

### Monitoring Setup
- [ ] Configure Prometheus scraping of metrics endpoints
- [ ] Set up alerts for ingestion failures (>5% failure rate)
- [ ] Set up alerts for high retrieval latency (>2 seconds)
- [ ] Set up alerts for low groundedness (<0.7 average)
- [ ] Create Grafana dashboards for pipeline health
- [ ] Set up email/Slack notifications for critical alerts

### Versioning Setup
- [ ] Implement daily automatic knowledge base snapshots
- [ ] Document rollback procedures
- [ ] Test rollback process weekly
- [ ] Archive old versions monthly
- [ ] Create disaster recovery plan

### Data Quality
- [ ] Track metadata completeness per domain
- [ ] Monitor chunk quality scores
- [ ] Track knowledge base staleness
- [ ] Generate weekly quality reports
- [ ] Establish SLOs for quality metrics

---

## PHASE 3 EXPECTED OUTCOMES

After implementing Phase 3:

✅ **Production Monitoring:**
- Real-time visibility into pipeline health
- Automated alerting for failures
- Performance metrics tracked continuously
- Historical data for trend analysis

✅ **Versioning & Rollback:**
- Daily snapshots of knowledge bases
- Ability to rollback to any version in <5 minutes
- Audit trail of all changes
- Disaster recovery capability

✅ **Data Quality:**
- Metrics tracking data quality continuously
- Alerts for degradation in real-time
- Weekly quality reports generated automatically
- Trending data for optimization

---

## IMPLEMENTATION TIMELINE

| Phase | Duration | Components | Expected Impact |
|-------|----------|-----------|-----------------|
| **Phase 1** | Week 1 | Metadata, Chunking, Delta Detection, Groundedness | 25-40% precision ↑ |
| **Phase 2** | Week 2-3 | Multi-Adapter Retrieval, LLM Reranking | 35-50% precision ↑ |
| **Phase 3** | Week 4+ | Monitoring, Versioning, Data Quality | Operational reliability ↑ |

---

## SUCCESS METRICS (TARGET END STATE)

| Metric | Target | Phase 1 | Phase 2 | Phase 3 |
|--------|--------|---------|---------|---------|
| Retrieval Precision@10 | 80%+ | 50% | 70% | 80%+ |
| Groundedness Score | 0.85+ | 0.70 | 0.80 | 0.85+ |
| Ingestion Success Rate | 98%+ | 90% | 95% | 98%+ |
| Avg Response Latency | <2s | 3s | 2.5s | <2s |
| KB Staleness | <24h | 7+ days | 2 days | <24h |

---

## QUICK START: PHASE 2 INTEGRATION

```bash
# 1. Create multi-adapter retriever
cp app/XNAi_rag_app/multi_adapter_retriever.py
cp app/XNAi_rag_app/llm_reranker.py

# 2. Test in Python
python -c "
from app.XNAi_rag_app.multi_adapter_retriever import MultiAdapterRetriever, RetrievalAdapter
# Test each adapter independently
retriever.retrieve('quantum entanglement', strategy=RetrievalAdapter.SEMANTIC)
retriever.retrieve('quantum entanglement', strategy=RetrievalAdapter.KEYWORD)
retriever.retrieve('quantum entanglement', strategy=RetrievalAdapter.HYBRID)
"

# 3. Integrate with RAG pipeline
# Update your RAG class to use MultiAdapterRetriever

# 4. Test end-to-end
python scripts/test_phase2_integration.py
```

## QUICK START: PHASE 3 SETUP

```bash
# 1. Create monitoring components
cp app/XNAi_rag_app/ingestion_monitor.py
cp app/XNAi_rag_app/quality_metrics.py

# 2. Create versioning script
cp scripts/version_knowledge_base.py

# 3. Set up Prometheus
# Add to docker-compose.yml or prometheus.yml:
# - job_name: 'xoe-novai'
#   static_configs:
#   - targets: ['localhost:8000']

# 4. Test monitoring
python -c "
from app.XNAi_rag_app.ingestion_monitor import IngestionMonitor
monitor = IngestionMonitor(redis_client)
monitor.log_ingestion_event('test.md', 'completed')
print(monitor.get_ingestion_stats(hours=1))
"

# 5. Test versioning
python scripts/version_knowledge_base.py --create v1.0 --domain science
```

---

**Next: Implement Phase 2 components, then integrate with existing RAG pipeline. Phase 3 can run in parallel with Phase 2 setup.**
