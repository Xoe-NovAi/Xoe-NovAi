# Phase 1.5 Implementation Checklist
## Quick Reference for Quality Scoring + Specialized Retrievers
**Target Duration:** 10 weeks (Weeks 6-15 of project)  
**Difficulty:** Intermediate (extends Phase 1, not replacement)

---

## CHECKLIST FORMAT

Use this format for tracking:
```
[✓] = Completed
[~] = In Progress  
[ ] = Not Started
[!] = Blocked
```

---

## WEEK 6-7: QUALITY SCORING FRAMEWORK

### Component 1: MetadataQualityScorer Class
- [ ] Create `app/XNAi_rag_app/quality_scorer.py`
- [ ] Implement `update_quality_score()` method
  - [ ] Retrieval count tracking
  - [ ] User feedback integration (thumbs up/down)
  - [ ] Citation count aggregation
  - [ ] Freshness score calculation (exponential decay)
- [ ] Implement `compute_freshness()` with configurable lambda
- [ ] Add Redis persistence for quality history
- [ ] Unit tests (3-5 test cases)

**Acceptance Criteria:**
- Quality score computed from 5 factors
- Scores persist across sessions
- Freshness decays over time
- All tests pass

### Component 2: Quality Score Integration Points
- [ ] Retrieval tracking hook in RAG pipeline
  - [ ] Call `track_retrieval(doc_id)` after every retrieval
  - [ ] Location: `app/XNAi_rag_app/rag_pipeline.py`
- [ ] User feedback integration
  - [ ] Add feedback UI endpoints (or CLI flags for testing)
  - [ ] Call `log_user_feedback(doc_id, thumbs_up)` on feedback
- [ ] Reranking in retrieval
  - [ ] Replace pure vector similarity with hybrid score
  - [ ] Formula: `0.7 * vector_sim + 0.3 * quality_score`
  - [ ] Add `rerank_by_quality()` to retrieval chain

**Acceptance Criteria:**
- Every retrieval is tracked automatically
- Quality scores update in real-time
- Reranking improves precision on retrievals
- No performance degradation

### Component 3: Quality Metrics Dashboard
- [ ] Add quality metrics to Prometheus
  - [ ] `doc_quality_score` (histogram)
  - [ ] `retrieval_frequency` (counter)
  - [ ] `average_user_satisfaction` (gauge)
- [ ] Dashboard in Grafana (optional, nice-to-have)
- [ ] Logging for quality trends

**Acceptance Criteria:**
- Metrics exportable to Prometheus
- Can view quality distribution
- Track improvements over time

---

## WEEK 8-9: SPECIALIZED RETRIEVERS

### Component 4: Code Retriever (AST-Aware)
- [ ] Create `app/XNAi_rag_app/code_retriever.py`
- [ ] Implement `CodeRetriever` class
  - [ ] Symbol extraction from query (regex: `[\w_]+`)
  - [ ] Grep-based search for function/class definitions
  - [ ] AST parsing for verification (optional, Phase 2)
  - [ ] Confidence scoring
- [ ] Index all `.py` files in codebase
  - [ ] Function/class definitions with locations
  - [ ] Docstring extraction
  - [ ] Import statement tracking
- [ ] Unit tests
  - [ ] Retrieve function by name
  - [ ] Retrieve class by name
  - [ ] Handle ambiguous symbols

**Acceptance Criteria:**
- Returns exact function/class matches for code queries
- Faster than vector search for code (100x)
- Handles Python syntax correctly
- Tests pass (100%)

### Component 5: Science Retriever (Citation-Aware)
- [ ] Create `app/XNAi_rag_app/science_retriever.py`
- [ ] Implement `ScienceRetriever` class
  - [ ] Vector search for seed papers (k=3)
  - [ ] Citation graph traversal (backward + forward)
  - [ ] Author H-index weighting (if available)
  - [ ] Relevance combination
- [ ] Citation index (optional, Phase 1.5)
  - [ ] Manual curation of key papers + citations
  - [ ] Or: Parse from academic metadata if available
- [ ] Fallback to vector search if no citations

**Acceptance Criteria:**
- Prioritizes highly-cited papers
- Follows citation chains correctly
- Doesn't break if citation data incomplete
- Tests pass

### Component 6: Data Retriever (Metadata-Aware)
- [ ] Create `app/XNAi_rag_app/data_retriever.py`
- [ ] Implement `DataRetriever` class
  - [ ] Structured query parsing (date, author, columns)
  - [ ] Metadata index search
  - [ ] Combination with vector search
- [ ] Metadata index
  - [ ] Extract date ranges
  - [ ] Author information
  - [ ] Column/field names

**Acceptance Criteria:**
- Filters by date range
- Filters by author
- Filters by data columns
- Combines with vector scores

### Component 7: Query Router
- [ ] Create `app/XNAi_rag_app/query_router.py`
- [ ] Implement `QueryRouter` class
  - [ ] Domain detection (keyword-based heuristic)
  - [ ] Route to appropriate retriever
  - [ ] Fallback to general vector search
- [ ] Domain keywords mapping
  - [ ] Code: `def`, `class`, `function`, `algorithm`
  - [ ] Science: `equation`, `experiment`, `theory`, `paper`
  - [ ] Data: `dataset`, `csv`, `column`, `table`
- [ ] Unit tests for each domain

**Acceptance Criteria:**
- Correctly detects code queries
- Correctly detects science queries
- Correctly detects data queries
- Gracefully falls back for ambiguous queries

---

## WEEK 10: HYPERGRAPH FOUNDATION (OPTIONAL)

**Note:** This is optional for Phase 1.5. Start only if Phase 1 on track.

### Component 8: HypergraphKnowledgeBase Class
- [ ] Create `app/XNAi_rag_app/hypergraph_kb.py`
- [ ] Implement `HypergraphKnowledgeBase` class
  - [ ] Local networkx graph storage
  - [ ] Redis persistence for hyperedges
  - [ ] `add_hyperedge()` method
  - [ ] `multi_hop_query()` method
  - [ ] `hybrid_retrieve()` combining vector + graph
- [ ] Edge types
  - [ ] ENABLES, CITES, REFUTES, EXTENDS, SIMILAR
- [ ] Testing
  - [ ] Add hyperedges
  - [ ] Query multi-hop
  - [ ] Verify reranking

**Acceptance Criteria:**
- Hyperedges persist across restarts
- Multi-hop queries work correctly
- Hybrid retrieval combines scores properly
- Tests pass

### Component 9: Manual Hyperedge Curation
- [ ] Identify key relationships in knowledge base
- [ ] Add 10-20 high-value hyperedges manually
  - [ ] Example: "Quantum Mechanics" ENABLES "Quantum Cryptography"
- [ ] Test multi-hop queries on curated edges

**Acceptance Criteria:**
- Can traverse 2-3 hops
- Multi-hop results are relevant
- No circular references

---

## WEEK 11-12: ADAPTIVE SEMANTIC SPACES (RESEARCH PHASE)

**Note:** Phase 1.5 is research phase only. Implementation in Phase 2.

### Component 10: AdaptiveEmbeddingRouter (Design Only)
- [ ] Create design document: `docs/adaptive_embedding_design.md`
- [ ] Evaluate domain-specific models
  - [ ] all-MiniLM-L12-v2 (current) vs. all-mpnet-base-v2
  - [ ] aspire-distilbert for science
  - [ ] codebert-base for code
- [ ] Performance comparison (vector space coverage)
- [ ] Cost analysis (memory, inference time)
- [ ] Decision: Which models to implement in Phase 2

**Acceptance Criteria:**
- Clear performance trade-offs documented
- Cost analysis complete
- Implementation plan for Phase 2

---

## TESTING STRATEGY

### Unit Tests (Per Component)
```python
# tests/test_quality_scorer.py
def test_quality_score_calculation():
    """Verify 5-factor weighting."""
    
def test_freshness_decay():
    """Verify exponential decay curve."""
    
def test_quality_persistence():
    """Verify Redis storage/retrieval."""

# tests/test_code_retriever.py
def test_retrieve_function():
    """Retrieve function by name."""
    
def test_retrieve_class():
    """Retrieve class by name."""
    
def test_symbol_extraction():
    """Parse symbols from query."""

# tests/test_science_retriever.py
def test_citation_graph_traversal():
    """Follow citation edges."""
    
def test_h_index_weighting():
    """Weight by author H-index."""

# tests/test_query_router.py
def test_detect_code_query():
    """Route code query correctly."""
    
def test_detect_science_query():
    """Route science query correctly."""
    
def test_fallback_to_general():
    """Fallback for ambiguous query."""
```

### Integration Tests
```python
# tests/test_phase_1_5_integration.py
def test_quality_score_improves_precision():
    """End-to-end: quality scoring improves retrieval quality."""
    
def test_specialized_retrievers_improve_domain_recall():
    """End-to-end: code retriever better for code queries."""
    
def test_hybrid_retrieval_combines_scores():
    """End-to-end: vector + quality + graph scoring."""
```

### Performance Benchmarks
- Code retriever latency: <50ms (vs. vector search >100ms)
- Quality reranking overhead: <10ms
- Multi-hop query (2 hops): <100ms

---

## PERFORMANCE TARGETS

| Metric | Baseline (Phase 1) | Target (Phase 1.5) | Improvement |
|--------|-------------------|-------------------|-------------|
| Overall Precision | 60% | 70% | +10% |
| Code Query Recall | 60% | 85% | +25% |
| Science Query Precision | 55% | 70% | +15% |
| Avg. Retrieval Latency | 150ms | 140ms | -10ms (faster code retrieval) |
| Quality Score Accuracy | N/A | 75% | - |

---

## ROLLOUT STRATEGY

### Gradual Rollout (Minimize Risk)
1. **Week 10:** Deploy quality scoring to 10% of traffic
   - Monitor precision metrics
   - Verify no regression
   
2. **Week 11:** Deploy to 50% of traffic
   - Compare precision before/after
   - Gather feedback
   
3. **Week 12:** Full deployment
   - All traffic using quality scoring
   - Keep vector-only search as fallback

4. **Week 13-15:** Deploy specialized retrievers
   - Code retriever: Enable for code-related queries
   - Science/Data: Enable for respective domains
   - Monitor hit rate (how often each is used)

### Rollback Plan
- Keep vector-only search as fallback
- If precision drops >5%: Rollback to Phase 1 configuration
- Adjust weighting and retry

---

## DEPENDENCIES & COMPATIBILITY

### New Python Dependencies
```
networkx>=2.6           # Hypergraph operations
sentence-transformers   # For Phase 2 adaptive spaces (optional now)
```

### No New External Services Required
- Redis (already in use)
- FAISS (already in use)

### Backward Compatibility
- Phase 1.5 is purely additive
- All Phase 1 components unchanged
- Quality scoring optional (can disable)

---

## DOCUMENTATION UPDATES

- [ ] Add `MetadataQualityScorer` to API docs
- [ ] Add specialized retrievers to API docs
- [ ] Update README with domain routing examples
- [ ] Create "Quality Scoring" user guide
- [ ] Add "Multi-Agent RAG" architecture diagram

---

## SUCCESS CRITERIA

Phase 1.5 is successful when:

1. ✅ Quality scoring deployed and tracking metrics
2. ✅ Code retriever returns exact matches for code queries
3. ✅ Science retriever follows citation chains
4. ✅ Query router correctly detects domains
5. ✅ Hybrid retrieval combines vector + quality scores
6. ✅ Overall precision improves to 70% (from 60%)
7. ✅ All tests pass (unit + integration)
8. ✅ Documentation complete
9. ✅ No regressions (latency unchanged)
10. ✅ Hypergraph foundation ready for Phase 2

---

## ESTIMATED EFFORT

| Component | Effort | Owner |
|-----------|--------|-------|
| Quality Scoring | 8 hours | Team |
| Code Retriever | 4 hours | Team |
| Science Retriever | 6 hours | Team |
| Data Retriever | 4 hours | Team |
| Query Router | 3 hours | Team |
| Testing | 8 hours | Team |
| Hypergraph Foundation | 4 hours | Team |
| **Total** | **37 hours** | **~1.5 weeks** |

---

## PHASE 2 PREVIEW: QDRANT MIGRATION (Weeks 16-18)

**Note:** Phase 2 starts after Phase 1.5 complete (Week 15). Below is early planning info.

### Key Phase 2 Components (Qdrant-Centric)

**Component 1: Qdrant Vector DB Setup (Week 16)**
- [ ] Provision Qdrant service (local or cloud)
- [ ] Performance validation (100K vectors test)
- [ ] Data migration plan (FAISS → Qdrant)

**Component 2: Migration Window (Week 16-18)**
- [ ] Week 16: Dual-write mode (write to both FAISS + Qdrant)
- [ ] Week 17: Read validation (compare results FAISS vs Qdrant)
- [ ] Week 18: Cutover (switch reads to Qdrant, deprecate FAISS)

**Component 3: Qdrant Feature Enablement**
- [ ] Built-in metadata filtering (replaces post-processing)
- [ ] Server-side reranking (improves latency)
- [ ] Incremental indexing (dynamic vector additions)

**Expected Improvements:**
- Query latency: 150ms (FAISS) → 85-100ms (Qdrant)
- Indexing speed: Full rebuild → Incremental
- Operational flexibility: Single machine → Cluster-ready

**For Detailed Phase 2 Planning:**
→ See ADVANCED_RAG_REFINEMENTS_2026.md section 4.2 (Vector Database Strategy)
→ See PHASE_2_3_ADVANCED_IMPLEMENTATION.md (full Phase 2 roadmap)

---

## NEXT STEPS

1. **Immediately:** Create files for Phase 1.5 components
2. **Week 6:** Start with quality scoring
3. **Week 8:** Add specialized retrievers in parallel
4. **Week 10:** Optional hypergraph foundation
5. **Week 15:** Complete Phase 1.5, plan Phase 2
6. **Week 16-18:** Qdrant migration (Phase 2 start)

---

**Document prepared by:** Implementation Planning  
**Date:** January 3, 2026  
**Version:** 1.0  
**Status:** Ready for Execution
