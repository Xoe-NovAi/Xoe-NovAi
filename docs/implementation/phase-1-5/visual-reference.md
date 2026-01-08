# Phase 1.5 Visual Reference Guide
## Quick Lookup for Architecture & Integration Points

---

## ARCHITECTURE OVERVIEW

### Current Phase 1 (Weeks 1-5)
```
User Query
    ↓
[Metadata Enrichment]
    ↓
[Semantic Chunking]
    ↓
[FAISS Vector Search] ←── Embedding (all-MiniLM-L12-v2)
    ↓
Top-5 Documents
    ↓
[Groundedness Scoring]
    ↓
[LLM Chain]
    ↓
Final Answer
```

### Enhanced Phase 1.5 (Weeks 6-15)
```
User Query
    ↓
[Query Router] ←── Domain Detection (code/science/data)
    ├─→ Code? ────→ [Code Retriever] (grep + AST)
    ├─→ Science? ──→ [Science Retriever] (citations)
    ├─→ Data? ─────→ [Data Retriever] (metadata)
    └─→ General ───→ [FAISS Vector Search]
    ↓
Retrieved Documents
    ↓
[Quality Scorer] ←── Retrieval History, User Feedback, Citations
    ↓
[Rerank by Quality] (70% vector + 30% quality)
    ↓
Top-5 Documents (improved quality)
    ↓
[Groundedness Scoring]
    ↓
[LLM Chain]
    ↓
Final Answer
```

---

## COMPONENT INTERACTION DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    QueryRouter                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input: "def calculate_revenue function"                 │   │
│  │ Detection: [code=0.8, science=0.1, data=0.1]           │   │
│  │ Route: CodeRetriever (confidence=0.8 > threshold=0.6)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           ├──→ CodeRetriever (if confidence ≥ 0.6)
           │    ├─ Extract symbols: ["calculate_revenue"]
           │    ├─ Grep for definitions: find_files(grep "def calculate_revenue")
           │    └─ Return: [(filepath, score)]
           │
           ├──→ ScienceRetriever (if confidence ≥ 0.5)
           │    ├─ Vector search seed papers (k=3)
           │    ├─ Follow citations (backward + forward)
           │    ├─ Weight by h-index
           │    └─ Return: [(paper_id, score)]
           │
           ├──→ DataRetriever (if confidence ≥ 0.6)
           │    ├─ Parse filters (date, author, columns)
           │    ├─ Metadata search
           │    ├─ Combine with vector search
           │    └─ Return: [(dataset_id, score)]
           │
           └──→ Vector Search (fallback)
                ├─ Embedding: query → vector
                ├─ FAISS search
                └─ Return: [(doc_id, similarity)]

           ↓
┌─────────────────────────────────────────────────────────────────┐
│              MetadataQualityScorer                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input: [doc1, doc2, doc3, doc4, doc5] (from retrievers) │   │
│  │                                                          │   │
│  │ For each doc:                                           │   │
│  │   quality_score = 0.2×retrieval                        │   │
│  │                 + 0.3×feedback                         │   │
│  │                 + 0.2×citations                        │   │
│  │                 + 0.15×freshness                       │   │
│  │                 + 0.15×expert                          │   │
│  │                                                          │   │
│  │ Hybrid Score = 0.7×vector_score + 0.3×quality_score   │   │
│  │                                                          │   │
│  │ Output: [doc1, doc2, doc3, doc4, doc5] (reranked)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           ↓
       [LLM Chain]
           ↓
       Final Answer
```

---

## DATA FLOW: QUALITY SCORING

```
Retrieval Event
    ├─→ [Track Retrieval] ──→ Redis: retrieval_count++
    │   track_retrieval(doc_id)
    │
User Feedback (Optional)
    ├─→ [Log Feedback] ──→ Redis: user_thumbs_up list
    │   log_user_feedback(doc_id, thumbs_up=True/False)
    │
Citation Discovery (External)
    ├─→ [Update Citation Count] ──→ Redis: citation_count++
    │   update_quality_score(doc_id, citation_count=N)
    │
Time Passage (Background)
    ├─→ [Compute Freshness] ──→ freshness = exp(-0.0013 * age_days)
    │
Expert Review (Manual)
    ├─→ [Set Expert Score] ──→ Redis: expert_review = 0.8
    │
Quality Recomputation (Every Retrieval or Daily)
    ├─→ [Update Quality Score] ──→ Redis:
    │   quality_score = 0.2×retrieval + 0.3×feedback + 0.2×citations + 0.15×freshness + 0.15×expert
    │
Result: Reranked Documents
    └─→ Documents sorted by hybrid_score = 0.7×vector + 0.3×quality
```

---

## INTEGRATION CHECKLIST

### Week 6-7: Quality Scoring Integration

**File: `app/XNAi_rag_app/rag_pipeline.py`**

```python
# Before (Phase 1)
def retrieve(self, query: str, k: int = 5):
    docs = self.vectorstore.similarity_search(query, k=k)
    return docs

# After (Phase 1.5)
def retrieve(self, query: str, k: int = 5):
    docs = self.vectorstore.similarity_search(query, k=k)
    
    # NEW: Track retrievals
    for doc in docs:
        self.quality_scorer.track_retrieval(doc.id)
    
    # NEW: Rerank by quality
    docs = self.quality_scorer.rerank_by_quality(docs)
    
    return docs

# NEW: Log feedback endpoint
def log_feedback(self, doc_id: str, thumbs_up: bool):
    self.quality_scorer.log_user_feedback(doc_id, thumbs_up)
```

### Week 8-9: Specialized Retrievers Integration

**File: `app/XNAi_rag_app/rag_pipeline.py`**

```python
# Before (Phase 1)
def retrieve(self, query: str, k: int = 5):
    docs = self.vectorstore.similarity_search(query, k=k)
    ...

# After (Phase 1.5)
def retrieve(self, query: str, k: int = 5):
    # NEW: Route to specialized retriever
    domain, retriever = self.query_router.route_query(query)
    
    if domain == 'code':
        # CodeRetriever returns (filepath, score) tuples
        results = retriever.retrieve(query, k=k)
        # Convert to Document objects
        docs = [self._load_document(filepath) for filepath, _ in results]
    elif domain == 'science':
        results = retriever.retrieve(query, k=k)
        docs = [self._load_document(paper_id) for paper_id, _ in results]
    elif domain == 'data':
        results = retriever.retrieve(query, k=k)
        docs = [self._load_document(dataset_id) for dataset_id, _ in results]
    else:
        # Fallback: vector search
        docs = self.vectorstore.similarity_search(query, k=k)
    
    # Track + rerank (existing Phase 1.5 code)
    for doc in docs:
        self.quality_scorer.track_retrieval(doc.id)
    docs = self.quality_scorer.rerank_by_quality(docs)
    
    return docs
```

---

## PERFORMANCE IMPROVEMENTS SUMMARY

### By Component

```
┌─────────────────────┬────────────┬──────────────────────┐
│ Component           │ Effort     │ Expected Improvement │
├─────────────────────┼────────────┼──────────────────────┤
│ Quality Scoring     │ 8 hours    │ +8-12% precision     │
│ Code Retriever      │ 4 hours    │ +25-30% on code Q&A  │
│ Science Retriever   │ 6 hours    │ +15-20% on papers    │
│ Data Retriever      │ 4 hours    │ +20% on datasets     │
│ Query Router        │ 3 hours    │ +5-10% routing acc.  │
│ Hypergraph (opt.)   │ 4 hours    │ +5-10% relational    │
├─────────────────────┼────────────┼──────────────────────┤
│ TOTAL               │ 37 hours   │ +20-25% cumulative   │
└─────────────────────┴────────────┴──────────────────────┘

Phase 1 Result: 60% precision
Phase 1.5 Result: 70-75% precision
Cumulative gain: +10-15 percentage points
```

### By Domain

```
Code Queries:
  Before: 60% recall (generic vector search)
  After:  85% recall (code-specific grep + AST)
  Improvement: +25 percentage points

Science Queries:
  Before: 55% precision (vector similarity)
  After:  70% precision (citation-aware)
  Improvement: +15 percentage points

Data Queries:
  Before: 50% precision (generic vector)
  After:  70% precision (metadata filtering)
  Improvement: +20 percentage points

General Queries:
  Before: 60% precision (vector search)
  After:  65% precision (quality reranking)
  Improvement: +5 percentage points
```

---

## REDIS KEY STRUCTURE

```
Quality Scoring Persistence:
┌──────────────────────────────────────────────┐
│ Key: doc_quality:{doc_id}                    │
├──────────────────────────────────────────────┤
│ Fields:                                      │
│   score         → 0.75 (current quality)    │
│   timestamp     → "2026-01-03T14:30:00Z"    │
│   factors       → JSON {retrieval,feedback} │
│   retrieval_count → 15                      │
│   user_thumbs_up → 0.85                     │
│   citation_count → 3                        │
│   expert_review → 0.9                       │
│   last_retrieved → "2026-01-03T14:35:00Z"   │
│                                              │
│ TTL: 365 days                               │
└──────────────────────────────────────────────┘

Key: doc_quality:{doc_id}:feedback
├─ List of feedback values (0.0 or 1.0)
│  └─ Used to compute average user_thumbs_up
│
Key: doc_quality:history
├─ Sorted set of quality scores over time
│  └─ member: doc_id, score: timestamp
```

---

## DOMAIN DETECTION REFERENCE

```
Keyword-Based Router:

┌─────────────────────────────────────────────────────┐
│ QUERY: "How to define a Python function?"          │
├─────────────────────────────────────────────────────┤
│ Code Keywords Found:     [def, function] (2 matches)│
│ Science Keywords Found:  [] (0 matches)            │
│ Data Keywords Found:     [] (0 matches)            │
├─────────────────────────────────────────────────────┤
│ Code Score:     2/15 = 0.133                       │
│ Science Score:  0/16 = 0.000                       │
│ Data Score:     0/14 = 0.000                       │
├─────────────────────────────────────────────────────┤
│ Best Domain:    code                               │
│ Confidence:     0.133 ≥ 0.6 threshold?  NO         │
│ → Route to: GENERAL (fallback)                     │
│                                                     │
│ * Improvement: Use word weighting to boost        │
│   "def" and "function" keywords                    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ QUERY: "def calculate_revenue(revenue, costs)..."  │
├─────────────────────────────────────────────────────┤
│ Code Keywords Found: [def, function, revenue] (3)  │
│ Science Keywords: [] (0)                           │
│ Data Keywords: [] (0)                              │
├─────────────────────────────────────────────────────┤
│ Code Score:     3/15 = 0.200                       │
│ Science Score:  0/16 = 0.000                       │
│ Data Score:     0/14 = 0.000                       │
├─────────────────────────────────────────────────────┤
│ Best Domain:    code                               │
│ Confidence:     0.200 ≥ 0.6?  NO (still low)      │
│ → Route to: GENERAL                                │
│                                                     │
│ * Better: Use exact symbol extraction (grep first) │
│   "def calculate_revenue" appears literally        │
└─────────────────────────────────────────────────────┘
```

---

## ROLLOUT TIMELINE

```
Week 6        Week 7        Week 8        Week 9        Week 10
│             │             │             │             │
├─ Quality    └─ Quality    ├─ Code       └─ Code      ├─ Hyper-
│  Scorer       Scorer        Retriever    Retriever    │  graph
│  Dev          Testing       Dev          Testing      │  (opt)
│              & Deploy                   & Deploy      │
│              to 10%         ├─ Science                └─ Base
│              traffic        │ Retriever                  (redis)
│                            │ Dev
│                            ├─ Data
│                            │ Retriever
│                            │ Dev
│                            ├─ Query
│                            │ Router
│                            │ Dev
│                            └─ Router
│                              Testing

Week 11       Week 12        Week 13       Week 14       Week 15
│             │             │             │             │
├─ Design     └─ Design     ├─ Code       └─ Spec.     └─ Phase
│  Adaptive     Complete      Retriever     Retrievers    1.5
│  Embeddings             │  Deploy        Deploy         Complete
│                         ├─ Science       (100%)
│                         │  Retriever
│                         │  Deploy
│                         └─ Data
│                            Retriever
│                            Deploy
```

---

## COMMON INTEGRATION ISSUES & SOLUTIONS

### Issue 1: Quality Scorer Slowing Down Retrieval
**Problem:** Every retrieval calls Redis, adding latency
**Solution:** Cache quality scores in memory (LRU)
```python
from functools import lru_cache

class MetadataQualityScorer:
    @lru_cache(maxsize=1000)
    def get_quality_score(self, doc_id: str):
        # Still fetch from Redis, but cache locally
        ...
```

### Issue 2: Domain Detection Too Aggressive
**Problem:** Router sends too many queries to specialized retrievers
**Solution:** Increase confidence thresholds
```python
config.code_threshold = 0.7  # Was 0.6
config.science_threshold = 0.6  # Was 0.5
config.data_threshold = 0.7  # Was 0.6

# Or: weight by keyword importance
code_keywords_high_weight = {'def': 2.0, 'class': 2.0, 'function': 1.5}
```

### Issue 3: CodeRetriever Finding Too Many Files
**Problem:** "calculate" matches both calculate.py and calculate_revenue.py
**Solution:** Use AST parsing to verify function signatures
```python
# Grep for "def calculate", then parse AST to get actual signature
import ast

def verify_function(filepath, symbol):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            return True
    return False
```

### Issue 4: Redis Connection Failures
**Problem:** Redis down → quality scorer throws exception
**Solution:** Graceful degradation (fallback to pure vector search)
```python
def rerank_by_quality(self, docs):
    try:
        # Try quality reranking
        return self.quality_scorer.rerank_by_quality(docs)
    except Exception as e:
        logger.error(f"Quality scoring failed: {e}, returning documents as-is")
        return docs  # Fallback: return unranked
```

---

## SUCCESS METRICS DASHBOARD

```
Real-time Monitoring (Prometheus):

doc_retrieval_count:        Total retrievals (counter)
                            ├─ code_retrieval: 1523
                            ├─ science_retrieval: 890
                            ├─ data_retrieval: 456
                            └─ general_retrieval: 2341

doc_quality_score:          Distribution of quality scores (histogram)
                            ├─ p50: 0.62
                            ├─ p95: 0.85
                            └─ p99: 0.92

user_satisfaction:          Positive feedback rate (gauge)
                            ├─ Overall: 72%
                            ├─ Code queries: 85%
                            ├─ Science queries: 70%
                            └─ Data queries: 65%

retrieval_latency:          Time per retrieval (histogram)
                            ├─ p50: 45ms
                            ├─ p95: 120ms
                            └─ p99: 250ms

query_router_accuracy:      % of queries routed correctly (gauge)
                            ├─ Overall: 78%
                            ├─ Code detection: 88%
                            ├─ Science detection: 72%
                            └─ Data detection: 68%

precision_score:            Measured precision (gauge)
                            ├─ Overall: 70% (target: 70%)
                            ├─ Code queries: 85% (target: 85%)
                            ├─ Science queries: 70% (target: 70%)
                            └─ Data queries: 70% (target: 70%)

hallucination_rate:         % of generated content not grounded (gauge)
                            └─ Overall: 15% (down from 22%)
```

---

**This visual guide is your quick reference. Print it!**

Document prepared by: Implementation Planning  
Date: January 3, 2026  
Version: 1.0
