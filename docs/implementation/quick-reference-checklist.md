# Quick Reference: Phase Implementation Checklist
## Xoe-NovAi Implementation Guide - TL;DR Version

**Last Updated:** January 2, 2026 | **Version:** 2.0

---

## 📋 PHASE 1: WEEK 1 - FOUNDATION (4 Components)

### Component 1.1: Metadata Enricher
**Where:** `app/XNAi_rag_app/metadata_enricher.py`  
**Time:** 2 hours  
**Extracts:** author, date, topic, confidence, keywords, entities, quality score

**To Implement:**
1. Copy code from `PHASE_1_IMPLEMENTATION_GUIDE.md` → Component 1
2. Test on 5 sample documents
3. Verify metadata accuracy (expect >85%)
4. Check confidence scores make sense (0.0-1.0)

**Success Criteria:**
- [ ] Author extracted for ≥90% of docs
- [ ] Confidence scores calibrated (0.5-0.95 range)
- [ ] Keywords relevant (spot-check 10 docs)

---

### Component 1.2: Semantic Chunker
**Where:** `app/XNAi_rag_app/semantic_chunker.py`  
**Time:** 2 hours  
**Features:** Structure preservation, code block integrity, context awareness

**To Implement:**
1. Copy code from `PHASE_1_IMPLEMENTATION_GUIDE.md` → Component 2
2. Test on markdown files with:
   - Headings (verify preserved)
   - Code blocks (verify not split)
   - Equations/formulas (verify context intact)
3. Check average chunk size (reasonable: 300-800 tokens)

**Success Criteria:**
- [ ] Headings preserved in chunk metadata
- [ ] Code blocks kept as units
- [ ] Chunks have semantic coherence
- [ ] Bidirectional context present

---

### Component 1.3: Delta Detector
**Where:** `app/XNAi_rag_app/delta_detector.py`  
**Time:** 1.5 hours  
**Features:** Hash-based change detection, audit trail

**To Implement:**
1. Copy code from `PHASE_1_IMPLEMENTATION_GUIDE.md` → Component 3
2. Test change detection:
   - Ingest file → hash stored ✓
   - No changes → skipped ✓
   - File modified → detected ✓
3. Verify audit trail logging

**Success Criteria:**
- [ ] Hash computed correctly (SHA256)
- [ ] Changes detected with >99% accuracy
- [ ] Audit trail logged to Redis
- [ ] Ingestion time reduced 40%+

---

### Component 1.4: Groundedness Scorer
**Where:** `app/XNAi_rag_app/groundedness_scorer.py`  
**Time:** 1.5 hours  
**Features:** Hallucination detection, claim verification

**To Implement:**
1. Copy code from `PHASE_1_IMPLEMENTATION_GUIDE.md` → Component 4
2. Integrate with your LLM response generation
3. Test on 20 sample responses:
   - Ground ones (expect 0.8+)
   - Ones with hallucinations (expect 0.3-0.6)
4. Calibrate scoring thresholds

**Success Criteria:**
- [ ] Groundedness scores correlate with actual accuracy
- [ ] Hallucinations detected automatically
- [ ] All responses scored (0.0-1.0 scale)
- [ ] Logging of ungrounded claims

---

### Phase 1 End-to-End Integration Test
**Time:** 2 hours

```bash
# 1. Set up test data
mkdir -p /test_data
cp sample_documents/*.md /test_data/

# 2. Run integration test
python scripts/ingest_with_metadata.py

# Expected output:
# ✅ 10 documents ingested
# ✅ Metadata extracted
# ✅ Chunks created (semantic + structure preserved)
# ✅ Deltas detected (only new/modified)
# ✅ Groundedness scores calculated

# 3. Verify results
# Check:
# - /knowledge/ contains FAISS indices
# - /data/ contains metadata
# - Audit logs in Redis
# - Groundedness scores in responses
```

**Phase 1 Success:** All 4 components working + 25-40% precision improvement

---

## 📋 PHASE 2: WEEKS 2-3 - RETRIEVAL (2 Components)

### Component 2.1: Multi-Adapter Retriever
**Where:** `app/XNAi_rag_app/multi_adapter_retriever.py`  
**Time:** 4 hours  
**Strategies:** Semantic (50%) + Keyword (30%) + Structure (20%)

**To Implement:**
1. Copy code from `PHASE_2_3_ADVANCED_IMPLEMENTATION.md` → Component 5
2. Integrate with your RAG pipeline's retrieve() method
3. Test each adapter independently:
   ```python
   # Semantic retrieval
   results = retriever.retrieve(query, strategy=RetrievalAdapter.SEMANTIC)
   
   # Keyword retrieval
   results = retriever.retrieve(query, strategy=RetrievalAdapter.KEYWORD)
   
   # Structure-aware retrieval
   results = retriever.retrieve(query, strategy=RetrievalAdapter.STRUCTURE)
   
   # Hybrid (combined)
   results = retriever.retrieve(query, strategy=RetrievalAdapter.HYBRID)
   ```
4. Run 50-query evaluation against Phase 1 baseline

**Success Criteria:**
- [ ] All 3 adapters working independently
- [ ] Hybrid always scores higher than single adapters
- [ ] Query-type routing working (how/what/why/compare)
- [ ] Precision improved by 15-20%

---

### Component 2.2: LLM-based Reranker
**Where:** `app/XNAi_rag_app/llm_reranker.py`  
**Time:** 2 hours  
**Feature:** Rerank top-20 results using LLM judgment

**To Implement:**
1. Copy code from `PHASE_2_3_ADVANCED_IMPLEMENTATION.md` → Component 6
2. Instantiate with your LLM:
   ```python
   from langchain.llms import OpenAI
   llm = OpenAI(model_name="gpt-3.5-turbo")
   reranker = LLMReranker(llm)
   ```
3. Test on 20 queries - measure latency
4. Decide: Enable by default or optional?

**Success Criteria:**
- [ ] Reranking adds <1s latency
- [ ] Precision improved by 10-15%
- [ ] Reranked results better than hybrid baseline
- [ ] LLM cost acceptable

---

### Phase 2 Integration
**Time:** 4 hours

```bash
# 1. Update RAG pipeline's retrieve() method
# Replace single FAISS query with:
from app.XNAi_rag_app.multi_adapter_retriever import MultiAdapterRetriever, RetrievalAdapter

retriever = MultiAdapterRetriever(faiss, bm25, redis, llm_reranker=reranker)
results = retriever.retrieve(query, strategy=RetrievalAdapter.HYBRID)

# 2. Update generate_response() to use top-10 results
for result in results[:10]:
    context += result.content

# 3. Run evaluation
python scripts/evaluate_phase2.py
```

**Phase 2 Success:** Retrieval precision 75%+ + cumulative 35-50% improvement

---

## 📋 PHASE 3: WEEK 4+ - OPERATIONS (3 Components)

### Component 3.1: Ingestion Monitor
**Where:** `app/XNAi_rag_app/ingestion_monitor.py`  
**Time:** 2 hours  
**Tracks:** Success/failure rates, processing time, audit trail

**To Implement:**
1. Copy code from `PHASE_2_3_ADVANCED_IMPLEMENTATION.md` → Component 7
2. Add logging to ingestion pipeline:
   ```python
   monitor.log_ingestion_event(file_path, 'started')
   # ... process file ...
   monitor.log_ingestion_event(file_path, 'completed', {'chunks': 50})
   ```
3. Verify metrics collection in Redis
4. Create basic dashboard query:
   ```python
   stats = monitor.get_ingestion_stats(hours=24)
   print(f"Success rate: {stats['success_rate']:.1%}")
   ```

**Success Criteria:**
- [ ] All ingestion events logged
- [ ] Stats queryable from Redis
- [ ] <1% monitoring overhead

---

### Component 3.2: Knowledge Base Versioning
**Where:** `scripts/version_knowledge_base.py`  
**Time:** 2 hours  
**Features:** Daily snapshots, rollback, archive

**To Implement:**
1. Copy code from `PHASE_2_3_ADVANCED_IMPLEMENTATION.md` → Component 8
2. Create daily cron job:
   ```bash
   # Add to crontab
   0 2 * * * /path/to/version_knowledge_base.py --create v$(date +\%Y\%m\%d)
   ```
3. Test rollback:
   ```bash
   # Create version
   python scripts/version_knowledge_base.py --create v20260102
   
   # Test rollback
   python scripts/version_knowledge_base.py --rollback v20260102
   ```
4. Document rollback procedure

**Success Criteria:**
- [ ] Daily snapshots created automatically
- [ ] Rollback completes in <5 minutes
- [ ] Snapshots verified (no data loss)
- [ ] Runbook documented

---

### Component 3.3: Prometheus Metrics
**Where:** `app/XNAi_rag_app/quality_metrics.py`  
**Time:** 3 hours  
**Metrics:** Success rate, latency, groundedness, KB staleness

**To Implement:**
1. Copy code from `PHASE_2_3_ADVANCED_IMPLEMENTATION.md` → Component 9
2. Add to your main application:
   ```python
   from app.XNAi_rag_app.quality_metrics import (
       record_ingestion_metric,
       record_retrieval_metric,
       record_groundedness_metric
   )
   
   # In ingestion loop:
   record_ingestion_metric('science', success=True, duration=2.5)
   
   # In retrieval:
   record_retrieval_metric('hybrid', latency=0.35)
   
   # In response generation:
   record_groundedness_metric(score=0.85)
   ```
3. Add Prometheus scraper to docker-compose or prometheus.yml
4. Create Grafana dashboards (see template below)

**Success Criteria:**
- [ ] All metrics exported to Prometheus
- [ ] Metrics scraped successfully
- [ ] Dashboards created and readable
- [ ] Alerts configured

---

### Phase 3 Prometheus Setup
**File:** `prometheus.yml` (add to docker-compose or standalone)

```yaml
scrape_configs:
  - job_name: 'xoe-novai'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Phase 3 Alert Rules
**File:** `alerts.yml`

```yaml
groups:
  - name: xoe_alerts
    rules:
      - alert: HighIngestionFailureRate
        expr: ingestion_failed_total > 5
        for: 5m
        annotations:
          summary: "High ingestion failures"
      
      - alert: HighRetrievalLatency
        expr: retrieval_latency_seconds{quantile="0.95"} > 2.0
        for: 5m
      
      - alert: LowGroundedness
        expr: groundedness_score < 0.7
        for: 10m
      
      - alert: KBStale
        expr: knowledge_base_staleness_hours > 48
        for: 1h
```

**Phase 3 Success:** 98%+ success rate + daily snapshots + real-time monitoring

---

## 🎯 SUCCESS METRICS TRACKER

| Metric | Target | Phase 1 | Phase 2 | Phase 3 |
|--------|--------|---------|---------|---------|
| Precision@10 | 80%+ | 60% ✓ | 75% ✓ | 80%+ ✓ |
| Groundedness | 0.85+ | 0.78 ✓ | 0.83 ✓ | 0.85+ ✓ |
| Ingestion Success | 98%+ | 90% ✓ | 95% ✓ | 98%+ ✓ |
| Query Latency | <2s | 3s ✓ | 2.5s ✓ | <2s ✓ |
| KB Staleness | <24h | 7d ✓ | 2d ✓ | <24h ✓ |

---

## 🛠️ COMMON ISSUES & FIXES

### Phase 1
| Issue | Solution |
|-------|----------|
| Metadata extraction inaccurate | Improve regex patterns; use LLM fallback |
| Chunks too large | Reduce chunk size; tighten boundaries |
| Delta detector missing changes | Check hash function; verify file reading |
| Groundedness false positives | Re-calibrate LLM prompt; use better verifier |

### Phase 2
| Issue | Solution |
|-------|----------|
| Hybrid worse than semantic | Adjust weights (try 0.6 semantic, 0.2 keyword) |
| Reranker too slow | Use cheaper LLM; disable for latency-critical paths |
| Precision regressed | Check chunk quality; verify metadata |

### Phase 3
| Issue | Solution |
|-------|----------|
| Metrics not exported | Verify Prometheus config; check /metrics endpoint |
| Rollback fails | Check disk space; verify tar command |
| Alerts too noisy | Raise thresholds; add for: conditions |

---

## 📚 DOCUMENT REFERENCE

| Phase | Main Document | Implementation | Test Strategy |
|-------|---------------|-----------------|---------------|
| **Phase 1** | PHASE_1_IMPLEMENTATION_GUIDE.md | Components 1-4 | Unit + integration |
| **Phase 2** | PHASE_2_3_ADVANCED_IMPLEMENTATION.md | Components 5-6 | Evaluation on 50 queries |
| **Phase 3** | PHASE_2_3_ADVANCED_IMPLEMENTATION.md | Components 7-9 | Load + chaos testing |

Full strategy: `COMPLETE_IMPLEMENTATION_ROADMAP.md`  
Configuration: `Enterprise_Configuration_Ingestion_Strategy.md`

---

## ⏱️ TIME ESTIMATE SUMMARY

```
Phase 1 (Week 1):      40-50 hours
  ├─ Component 1.1 (Metadata):    2h
  ├─ Component 1.2 (Chunker):     2h
  ├─ Component 1.3 (Delta):       1.5h
  ├─ Component 1.4 (Groundedness):1.5h
  └─ Integration + Testing:       6-8h

Phase 2 (Weeks 2-3):   60-70 hours
  ├─ Component 2.1 (Retriever):   4h
  ├─ Component 2.2 (Reranker):    2h
  ├─ RAG Integration:             4h
  └─ Evaluation + Tuning:         8h

Phase 3 (Week 4+):     80-100 hours
  ├─ Component 3.1 (Monitor):     2h
  ├─ Component 3.2 (Versioning):  2h
  ├─ Component 3.3 (Metrics):     3h
  ├─ Prometheus + Grafana:        6-8h
  └─ Alerting + Runbooks:         4h

TOTAL: ~200 hours (~5 engineer-weeks)
```

---

## 🚀 START NOW

**Today (Day 1):**
1. ✅ Read this quick reference
2. ✅ Read PHASE_1_IMPLEMENTATION_GUIDE.md (Components 1-4)
3. ✅ Clone code into your dev environment

**Week 1 (Phase 1):**
- Implement Component 1.1-1.4
- Run integration test
- Verify metrics improved

**Weeks 2-3 (Phase 2):**
- Implement Component 2.1-2.2
- Integrate with RAG
- Run evaluation

**Week 4+ (Phase 3):**
- Implement monitoring
- Set up versioning
- Create dashboards

---

**Status:** Ready to implement | **Version:** 2.0 | **Last Updated:** January 2, 2026

🎯 **Goal:** Production-grade knowledge management system with 80%+ precision, <2s latency, 98%+ reliability
