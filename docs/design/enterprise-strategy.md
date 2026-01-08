# Enterprise-Grade Configuration, Ingestion & Knowledge Management Strategy
## Xoe-NovAi v0.1.4-stable

**Report Generated:** January 2, 2026  
**Status:** ✅ HEALTH CHECKS PASS | ✅ TELEMETRY AUDIT PASS | 📋 RECOMMENDATIONS PROVIDED

---

## EXECUTIVE SUMMARY

Your Xoe-NovAi stack has **excellent foundational architecture** with proper config loading (Pydantic validation, TOML-based, environment overrides). To achieve **true enterprise-grade maturity**, three key improvements are recommended:

1. **Configuration Management:** Add environment layering (dev/staging/prod), hot-reload capability, and audit trails
2. **Ingestion Pipeline:** Implement multi-source ingestion with file watching, automatic routing, and versioning  
3. **Knowledge Organization:** Design domain-expert routing with library→knowledge curation flow and metadata tagging

---

## PART 1: CURRENT STATE ANALYSIS

### ✅ Health Check Results
```
Status: healthy
Version: v0.1.4-stable
Memory: 4.19GB / 5.95GB available
Components: ✅ embeddings, ✅ vectorstore, ✅ llm, ✅ health_memory, ✅ health_redis, ✅ health_ryzen
```

### ✅ Telemetry Audit Results
All 8 privacy disables verified:
- ✅ CHAINLIT_NO_TELEMETRY=true
- ✅ CRAWL4AI_TELEMETRY=0
- ✅ LANGCHAIN_TRACING_V2=false
- ✅ SCARF_NO_ANALYTICS=true
- ✅ DO_NOT_TRACK=1
- ✅ PYTHONDONTWRITEBYTECODE=1
- ✅ project.telemetry_enabled=false
- ✅ chainlit.no_telemetry=true

### Current Config Structure Strengths
```
✅ Pydantic schema validation (7 model classes)
✅ TOML-based configuration
✅ Dot-notation access (config.project.name)
✅ Environment variable fallbacks
✅ Cached loading (@lru_cache)
✅ 23 config sections
✅ Multi-path fallback candidates
```

### Configuration Gaps (Non-Critical, Enterprise-Level)
```
⚠️  No environment layering (dev/staging/prod configs)
⚠️  No hot-reload capability (requires app restart)
⚠️  No config change audit trail
⚠️  No validation tiers (strict vs. permissive modes)
⚠️  No secrets management integration (Vault, AWS Secrets Manager)
⚠️  No config versioning/rollback
```

---

## PART 2: CONFIGURATION MANAGEMENT RECOMMENDATIONS

### Current Architecture
```
config.toml (23 sections)
    ↓
config_loader.py (Pydantic validation)
    ↓
App initialization
    ↓
No hot-reload, no audit trail
```

### Enterprise-Grade Architecture (Recommended)

#### 2.1 Environment Layering Pattern

**Implement 3-tier configuration override system:**

```python
# Priority order (highest to lowest):
1. Environment variables     (runtime overrides)
2. config.{ENV}.toml        (environment-specific: dev/staging/prod)
3. config.base.toml         (shared defaults)
4. Hardcoded defaults       (fallback)

# File structure:
/Xoe-NovAi/
  config/
    config.base.toml         # Shared defaults
    config.dev.toml          # Development overrides
    config.staging.toml      # Staging overrides
    config.prod.toml         # Production overrides
    secrets.example.toml     # Template for secrets
```

**Benefit:** Different memory/throughput targets per environment, secure separation of credentials

#### 2.2 Configuration Validation Tiers

```python
class ValidationMode(Enum):
    STRICT = 1        # Production: fail on ANY unknown field
    PERMISSIVE = 2    # Development: warn on unknown fields
    LENIENT = 3       # Testing: ignore unknown fields

# Usage:
config = load_config(mode=ValidationMode.STRICT if ENV=="prod" else ValidationMode.PERMISSIVE)
```

#### 2.3 Hot-Reload with File Watching

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.toml'):
            logger.info(f"Config changed: {event.src_path}")
            # Reload config without restarting app
            CONFIG_CACHE.clear()  # Clear LRU cache
            new_config = load_config()
            # Notify subscribers of config change
            CONFIG_CHANGED_EVENT.notify(new_config)

# Start in background thread
observer = Observer()
observer.schedule(ConfigReloadHandler(), path="/app/config", recursive=False)
observer.start()
```

**Benefits:**
- Change logging/monitoring/orchestration URLs without restarting
- Update memory limits, rate limits, timeouts at runtime
- A/B test different configurations

#### 2.4 Configuration Audit Trail

```python
class ConfigAuditLog:
    """Track all config changes with timestamp and source"""
    
    def log_change(self, section: str, key: str, old_value: Any, new_value: Any, source: str):
        """Log config change with audit trail"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "section": section,
            "key": key,
            "old_value": str(old_value),  # Hide sensitive data
            "new_value": "[REDACTED]" if self._is_sensitive(key) else str(new_value),
            "source": source,  # "env_var" | "config_file" | "api"
            "user": os.getenv("USER", "system"),
        }
        # Store to Redis or log file
        self.store_audit_entry(entry)

# Usage:
audit_log.log_change("server", "port", 8000, 8001, source="manual_api_change")
```

---

## PART 3: INGESTION PIPELINE RECOMMENDATIONS

### Current Ingestion Architecture
```python
library/ingest.py
    ↓ load_markdown_files()
    ↓ Hardcoded glob("**/*.md")
    ↓ Split & chunk
    ↓ FAISS vectorization
    ↓ Save with fsync
    
Result: ❌ Manual only, no file watching, no metadata routing
        ❌ Static indices (no incremental re-indexing)
        ❌ One-shot pipelines (staleness problems)
        ❌ No data lineage or versioning
```

### Enterprise-Grade Ingestion: The Declarative Paradigm

**KEY INSIGHT from Pixeltable Framework:**
Traditional ingestion treats documents as write-once archives. **Declarative ingestion** treats chunks, embeddings, and metadata as "computed columns" that automatically update when source data changes.

**Before (Traditional):**
```
Update source.md
    ↓ Manual: trigger re-index
    ↓ Re-process entire collection
    ↓ Expensive, error-prone
    ↓ Metadata gets out of sync
```

**After (Declarative):**
```
Update source.md
    ↓ System detects delta automatically
    ↓ Only affected chunks recomputed
    ↓ Metadata propagates automatically
    ↓ Lineage/relationships maintained
    ↓ NO manual intervention needed
```

### Proposed Multi-Source Ingestion Architecture

#### 3.1 Advanced Three-Tier Ingestion Flow (Declarative Pattern)

```
INGESTION SOURCES:
├── MANUAL DROP (user drops files into /library/incoming/)
├── AUTO-DETECT (watches /library/ for changes - including UPDATES)
└── CURATION (expert validation with LLM agents)
    
        ↓
        
CHANGE DETECTION (Delta-Based, NOT Full Re-indexing)
├── File hash computation (MD5/SHA256)
├── Modification timestamp tracking
├── Incremental change detection (only modified sections)
└── Source versioning with git-like diffs
    
        ↓
        
METADATA ENRICHMENT (40% of development time goes here)
├── Semantic metadata: domain, topics, confidence_score
├── Technical metadata: chunk_size, embedding_model, date
├── PII removal & data masking
├── Normalization: stemming, boilerplate removal (+25-40% precision)
└── Quality scoring (based on source authority, freshness, completeness)
    
        ↓
        
DOMAIN-SPECIFIC PROCESSING (Specialized Handlers per Expert Domain)
├── Science: equation extraction (LaTeX), definitions, citations
├── Code: AST parsing, symbol caching, syntax extraction
├── Esoteric: symbolic notation normalization, archaic language preservation
└── Librarian: cross-references, ontology linking (controlled vocabulary)
    
        ↓
        
SEMANTIC CHUNKING (Context-Aware, NOT Simple Token Split)
├── Respect paragraph boundaries
├── Preserve complete thoughts
├── Extract heading hierarchies
└── Maintain code block integrity
    
        ↓
        
INCREMENTAL EMBEDDING & INDEXING (Declarative Recomputation)
├── Compute embeddings only for NEW/MODIFIED chunks
├── Update FAISS indices surgically (not full rebuild)
├── Maintain chunk relationships automatically
└── Version embeddings with source version hash
    
        ↓
        
STORAGE WITH LINEAGE TRACKING
├── Library (raw sources): /library/{domain}/ [git-managed]
├── Knowledge (curated indices): /knowledge/{domain}/ [versioned]
├── Metadata DB: source_hash → {embeddings, chunk_refs, quality_score}
└── Audit log: all ingestion operations with source:destination tracing
```

**Key Differences from Naive RAG:**
| Aspect | Naive RAG | Declarative RAG |
|--------|-----------|-----------------|
| File Update | Manual re-index entire collection | Automatic delta detection |
| Chunking | Simple token split | Semantic boundaries respected |
| Metadata | Minimal (filename only) | Rich (40% effort spent here) |
| Index Updates | Rebuild entire indices | Surgical, incremental updates |
| Freshness | TTL-based staleness | Continuous change tracking |
| Lineage | None | Full source→embedding traceability |

#### 3.2 File Watcher Implementation

```python
# scripts/ingestion_watcher.py

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import hashlib
import json

class IngestionHandler(FileSystemEventHandler):
    """Watch /library/ for new files and route to appropriate handlers"""
    
    DOMAIN_PATTERNS = {
        'science': ['physics', 'chemistry', 'biology', 'research'],
        'coding': ['code', 'python', 'rust', 'javascript', 'algorithm'],
        'esoteric': ['occult', 'mystical', 'tarot', 'hermetic'],
        'classic': ['literature', 'philosophy', 'history'],
    }
    
    def on_created(self, event):
        """New file appeared in /library/"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Skip hidden files, temp files
        if file_path.name.startswith('.'):
            return
        
        logger.info(f"📥 Detected new file: {file_path}")
        
        # Route to appropriate handler
        domain = self._detect_domain(file_path)
        file_type = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        file_hash = self._compute_hash(file_path)
        
        # Create metadata record
        metadata = {
            "filename": file_path.name,
            "domain": domain,
            "file_type": file_type,
            "size_bytes": file_size,
            "hash": file_hash,
            "detected_at": datetime.utcnow().isoformat(),
            "status": "pending_ingestion",
            "version": "1.0",
        }
        
        # Queue for ingestion
        self._enqueue_ingestion(file_path, metadata)
    
    def _detect_domain(self, file_path: Path) -> str:
        """Detect expert domain from filename and path"""
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            if any(pattern in name or pattern in parent for pattern in patterns):
                return domain
        
        return "uncategorized"  # Default fallback
    
    def _enqueue_ingestion(self, file_path: Path, metadata: dict):
        """Add to Redis queue for processing"""
        # Store metadata
        redis_client.hset(
            f"ingestion:files:{metadata['hash']}",
            mapping=metadata
        )
        # Queue for worker
        redis_client.rpush("ingestion:queue", metadata['hash'])
        logger.info(f"✅ Queued: {file_path.name} as {metadata['domain']}")

# Usage:
observer = Observer()
observer.schedule(IngestionHandler(), path="/library", recursive=True)
observer.start()
observer.join()
```

#### 3.3 Ingestion Worker with Domain Routing

```python
# scripts/curation_worker.py (enhanced)

class IngestionWorker:
    """Process queued documents with domain-specific handlers"""
    
    DOMAIN_HANDLERS = {
        'science': ScienceDocumentHandler(),      # Extract equations, data
        'coding': CodeDocumentHandler(),          # Extract code blocks, syntax
        'esoteric': EsotericDocumentHandler(),    # Handle symbols, notation
        'classic': ClassicLiteratureHandler(),    # Extract passages, themes
        'uncategorized': GenericDocumentHandler(),
    }
    
    def process_queue(self):
        """Process ingestion queue continuously"""
        while True:
            # Get from Redis queue
            file_hash = redis_client.blpop("ingestion:queue", timeout=5)
            if not file_hash:
                continue
            
            # Get metadata
            metadata = redis_client.hgetall(f"ingestion:files:{file_hash[1]}")
            file_path = Path(metadata['filepath'])
            domain = metadata['domain']
            
            try:
                logger.info(f"🔄 Processing: {file_path.name} ({domain})")
                
                # Load document
                documents = self._load_documents(file_path)
                
                # Domain-specific processing
                handler = self.DOMAIN_HANDLERS[domain]
                processed_docs = handler.process(documents, metadata)
                
                # Chunk and embed
                chunks = self._chunk_documents(processed_docs)
                embeddings = get_embeddings()
                vectors = embeddings.embed_documents([c.page_content for c in chunks])
                
                # Store in FAISS
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Save to domain-specific knowledge folder
                knowledge_path = f"/knowledge/{domain}/{metadata['hash'][:8]}"
                vectorstore.save_local(knowledge_path)
                
                # Fsync for durability
                self._fsync_directory(knowledge_path)
                
                # Update metadata
                metadata['status'] = 'completed'
                metadata['knowledge_path'] = knowledge_path
                metadata['vector_count'] = len(chunks)
                metadata['completed_at'] = datetime.utcnow().isoformat()
                
                redis_client.hset(f"ingestion:files:{file_hash[1]}", mapping=metadata)
                logger.info(f"✅ Completed: {file_path.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"❌ Failed: {file_path.name}: {e}")
                metadata['status'] = 'failed'
                metadata['error'] = str(e)
                redis_client.hset(f"ingestion:files:{file_hash[1]}", mapping=metadata)
```

#### 3.4 Document Type Handlers (Template)

```python
from abc import ABC, abstractmethod
from typing import List

class DomainDocumentHandler(ABC):
    """Base class for domain-specific document processing"""
    
    @abstractmethod
    def process(self, documents: List[Document], metadata: dict) -> List[Document]:
        """Process documents with domain-specific extraction/enrichment"""
        pass

class CodeDocumentHandler(DomainDocumentHandler):
    """Extract code blocks, syntax highlighting, examples"""
    
    def process(self, documents: List[Document], metadata: dict) -> List[Document]:
        processed = []
        
        for doc in documents:
            # Extract code blocks using regex or AST
            code_blocks = self._extract_code_blocks(doc.page_content)
            
            for block in code_blocks:
                # Create separate doc for each code block
                code_doc = Document(
                    page_content=block['code'],
                    metadata={
                        **doc.metadata,
                        **metadata,
                        'type': 'code',
                        'language': block['language'],
                        'expert': 'coding',
                    }
                )
                processed.append(code_doc)
        
        return processed
    
    def _extract_code_blocks(self, content: str) -> List[dict]:
        """Extract code blocks from markdown/text"""
        import re
        blocks = []
        
        # Match ```language code ``` patterns
        pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(pattern, content, re.DOTALL):
            blocks.append({
                'language': match.group(1) or 'unknown',
                'code': match.group(2),
            })
        
        return blocks

class ScienceDocumentHandler(DomainDocumentHandler):
    """Extract equations, definitions, scientific notation"""
    
    def process(self, documents: List[Document], metadata: dict) -> List[Document]:
        # Extract LaTeX equations
        # Extract definitions
        # Extract citations
        # Extract data/figures
        pass

class EsotericDocumentHandler(DomainDocumentHandler):
    """Handle esoteric notation, symbols, systems"""
    
    def process(self, documents: List[Document], metadata: dict) -> List[Document]:
        # Extract symbolic notation
        # Map to standardized meanings
        # Preserve archaic language
        pass
```

#### 3.5 Data Quality Parameters (Production Baseline)

Research shows **40% of development time should focus on metadata strategy** for retrieval precision.

```python
class DataQualityConfig:
    """Parameters for high-precision ingestion"""
    
    # CHUNKING STRATEGY
    SEMANTIC_CHUNKING = True          # Respect boundaries, not tokens
    PRESERVE_STRUCTURE = True         # Keep headings, code blocks, equations
    CHUNK_SIZE_MIN = 100              # Minimum characters
    CHUNK_SIZE_MAX = 800              # Maximum characters (semantic units)
    
    # METADATA ENRICHMENT (Critical!)
    EXTRACT_METADATA = {
        'date': True,                 # When was this written?
        'author': True,               # Who wrote this?
        'topic': True,                # What is the topic?
        'confidence_score': True,     # Domain expert confidence (0.0-1.0)
        'source_quality': True,       # Academic? Blog? Official docs?
        'keywords': True,             # Extract key terms
        'entities': True,             # Named entities (companies, people, etc.)
    }
    
    # DATA CLEANING
    NORMALIZATION = {
        'stemming': True,             # Reduces variants (jump/jumping/jumped)
        'boilerplate_removal': True,  # Remove footers, headers, ads
        'lowercasing': True,          # Standardize case
        'format_cleaning': True,      # Remove extra whitespace, line breaks
    }
    
    # SECURITY & COMPLIANCE
    PII_REMOVAL = True                # Mask personal data
    COMPLIANCE_FILTERS = {
        'remove_emails': True,
        'remove_phone_numbers': True,
        'remove_credit_cards': True,
        'apply_gdpr_retention': True, # Respect data residency
    }
    
    # QUALITY TIERS
    QUALITY_SCORE_WEIGHTS = {
        'source_authority': 0.4,      # Academic > Blog > Social media
        'freshness': 0.2,             # Older content scores lower
        'completeness': 0.2,          # Is the content full or fragmented?
        'expert_review': 0.2,         # Has a domain expert validated?
    }
    
    # RETRIEVAL PRECISION IMPROVEMENT
    EXPECTED_PRECISION_GAIN = 0.25    # 25-40% improvement from metadata
    EXPECTED_RECALL_GAIN = 0.15       # Better filtering, fewer false positives

# Usage in ingestion:
quality = DataQualityConfig()
if quality.NORMALIZATION['stemming']:
    text = stem_text(text)  # jump/jumping/jumped → jump
```

**Impact:** Production systems with rich metadata see **25-40% precision improvement** over baseline vector-only RAG.

---

## PART 4: KNOWLEDGE ORGANIZATION & ADVANCED RETRIEVAL

### Proposed Directory Structure

```
/Xoe-NovAi/
├── library/               # Raw digital library (all source materials)
│   ├── incoming/         # Drop zone for new files
│   ├── science/          # Science documents
│   │   ├── physics/
│   │   ├── biology/
│   │   └── chemistry/
│   ├── coding/           # Code and programming
│   │   ├── python/
│   │   ├── rust/
│   │   └── algorithms/
│   ├── esoteric/         # Esoteric and mystical
│   │   ├── tarot/
│   │   ├── hermetic/
│   │   └── occult/
│   ├── classic/          # Classical literature
│   │   ├── philosophy/
│   │   ├── literature/
│   │   └── history/
│   └── archives/         # Processed/indexed sources
│
├── knowledge/            # Curated expert knowledge bases
│   ├── science/          # Science expert's curated KB
│   │   ├── vectors/      # FAISS index
│   │   ├── metadata.json # Quality metrics
│   │   └── sources.json  # Source lineage
│   ├── coding/           # Coding expert's curated KB
│   ├── esoteric/         # Esoteric expert's curated KB
│   ├── classic/          # Librarian's curated KB
│   └── librarian/        # Cross-domain curation
│
└── data/
    └── ingestion/
        ├── queue.json         # Processing queue
        ├── audit.log          # Ingestion audit trail
        └── metadata/          # File metadata cache
```

### 4.1 Hybrid Retrieval Architecture (Beyond Vector-Only RAG)

**Challenge:** Pure vector RAG fails for:
- Complex multi-hop reasoning ("What is the impact of X on Y on Z?")
- Relationship-aware queries (legal precedents, causal chains)
- Hallucination reduction (LLM makes up false connections)

**Solution: LightRAG - Parallel Vector + Graph Pipelines**

```
USER QUERY: "How does quantum entanglement affect entanglement-based cryptography?"
    ↓
    ├─── VECTOR PIPELINE (Low-Level Semantic Search)
    │    ├─ Embed query
    │    ├─ Search FAISS for semantically similar chunks
    │    └─ Return: Top-20 semantically related documents
    │
    └─── GRAPH PIPELINE (High-Level Relationship Reasoning)
         ├─ Extract entities from query
         ├─ Link to knowledge graph (quantum → entanglement → cryptography)
         ├─ Perform multi-hop path reasoning
         ├─ Rank paths by relevance (e.g., causal impact)
         └─ Return: Top-5 relationship chains

        ↓
        
FUSION LAYER
├─ Combine vector results + graph results
├─ Deduplicate
├─ Re-rank using combined score
└─ Provide to LLM with BOTH semantic AND relational context

        ↓

FINAL RESPONSE
✅ Grounded in retrieved documents (reduced hallucination)
✅ Semantically coherent (from vectors)
✅ Relationship-aware (from graph)
✅ Explainable (can trace reasoning path)
```

**Key Advantage for Multi-Expert System:**
- Science expert: "Show me the relationship chain from X to Y"
- Coding expert: "Show me all code that calls this function"
- Librarian: "Show me all related documents across domains"

### 4.2 Knowledge Graph Construction (Controlled Vocabulary)

To prevent "semiotic turbulence" (shifting meanings as LLMs evolve), use **domain ontologies**:

```python
class DomainOntology:
    """Structured knowledge schema for each expert domain"""
    
    SCIENCE_ONTOLOGY = {
        'concepts': {
            'quantum_entanglement': {
                'definition': 'A quantum phenomenon where particles share state',
                'aliases': ['entanglement', 'EPR pairs'],
                'broader': ['quantum_mechanics'],
                'related': ['Bell inequality', 'superposition'],
                'formal_definition': 'ρ(A,B) ≠ ρ(A) ⊗ ρ(B)',  # Not separable
            },
            'quantum_cryptography': {
                'definition': 'Using quantum mechanics for secure communication',
                'uses': ['quantum_entanglement', 'photon polarization'],
                'protocols': ['BB84', 'E91'],
            },
        },
        'relationships': [
            {
                'from': 'quantum_entanglement',
                'to': 'quantum_cryptography',
                'type': 'ENABLES',
                'strength': 0.9,  # Confidence score
                'temporal_scope': 'established_since_1984',
            },
        ],
    }
    
    CODING_ONTOLOGY = {
        'concepts': {
            'function': {
                'attributes': ['name', 'parameters', 'return_type', 'language'],
                'broader': ['code_element'],
            },
        },
        'relationships': [
            {'type': 'CALLS', 'from': 'function_a', 'to': 'function_b'},
            {'type': 'IMPORTS', 'from': 'module_a', 'to': 'library_b'},
        ],
    }

# Usage: when LLM says "entanglement", normalize to canonical form
canonical_term = ontology.get_canonical('entanglement')  # → quantum_entanglement
related_concepts = ontology.get_related('quantum_entanglement')
```

**Benefit:** LLM can't hallucinate non-existent relationships; must ground in ontology.

### 4.3 Knowledge Graph Traversal Patterns

For multi-expert queries, support **depth vs. breadth** search:

```python
class KnowledgeGraphTraversal:
    """Navigate knowledge graph for expert-specific reasoning"""
    
    def depth_search(self, start_node: str, max_depth: int = 5) -> List[Path]:
        """Vertical traversal: deep analysis in one direction
        
        Use case: Legal expert finding specific precedent chain
        Example: Find the evolution of contract interpretation law
        Query → Contract Law → Covenant Interpretation → Estoppel → Historical precedent
        """
        paths = []
        self._dfs(start_node, [], max_depth, paths)
        return paths
    
    def breadth_search(self, start_node: str, max_distance: int = 2) -> Dict[str, List[Path]]:
        """Horizontal traversal: broad perspective exploring adjacent relationships
        
        Use case: Market researcher understanding competitive landscape
        Example: Company → Competitors → Suppliers → Regulatory bodies
        """
        adjacent = {}
        self._bfs(start_node, max_distance, adjacent)
        return adjacent
    
    def neighbor_extraction(self, node: str, relation_type: str = None) -> List[Node]:
        """Direct neighbors for specific relation type
        
        Use case: Customer support agent finding related products/FAQs
        Example: Product A → Related Products → Accessories → Tutorials
        """
        return self.graph.get_neighbors(node, relation_type)
    
    def path_ranking(self, path: Path, context: str) -> float:
        """Rank reasoning paths by relevance
        
        Use case: Risk assessment expert evaluating failure chains
        Scores paths based on causal strength and recency
        """
        score = 0.0
        for edge in path.edges:
            score += edge.strength * self._temporal_decay(edge.timestamp)
        return score / len(path.edges)  # Average normalized score
```

**Expert-Specific Routing:**

| Expert | Search Type | Example Query |
|--------|------------|---------------|
| Legal/Compliance Officer | Depth | "Find the legal precedent chain for contract X" |
| Market Research Analyst | Breadth | "Map the competitive ecosystem around Product Y" |
| Customer Support Agent | Neighbor | "Show all products related to this one" |
| Risk Assessment Expert | Path Ranking | "What's the causal chain leading to failure Z?" |
| Scientist | Depth | "Trace the theoretical foundations of this paper" |
| Librarian | All | Cross-domain synthesis |

---

### 4.4 Semiotic Stability: Controlled Vocabulary Framework

**Problem:** LLMs synthesize from evolving textual databases, causing "semantic drift"

**Solution:** Encode domain-expert definitions explicitly

```python
class ControlledVocabulary:
    """Ensure stable meaning as knowledge base evolves"""
    
    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology
        self.canonical_terms = {}
        self.semantic_relationships = {}
    
    def normalize_entity(self, raw_text: str, domain: str) -> Tuple[str, float]:
        """Map user text to canonical entity with confidence score
        
        Examples:
        - "quantum entanglement" → canonical: quantum_entanglement (confidence: 0.99)
        - "spooky action at a distance" → quantum_entanglement (confidence: 0.85)
        - "EPR pairs" → quantum_entanglement (confidence: 0.95)
        """
        candidates = self.ontology.find_matches(raw_text, domain)
        
        # Score matches based on:
        # 1. Lexical similarity
        # 2. Semantic similarity
        # 3. Alias matching
        # 4. Domain context
        
        best_match = max(candidates, key=lambda c: c['confidence'])
        return best_match['canonical'], best_match['confidence']
    
    def get_definition(self, entity: str, domain: str) -> str:
        """Retrieve expert-curated definition (not LLM-generated)"""
        return self.ontology.get_definition(entity, domain)
    
    def validate_relationship(self, entity_a: str, relation: str, entity_b: str) -> bool:
        """Check if relationship exists in ontology
        
        Returns False for hallucinated relationships:
        - quantum_entanglement CAUSES (weather)  → False ✗
        - quantum_entanglement ENABLES quantum_cryptography  → True ✓
        """
        return self.ontology.has_edge(entity_a, relation, entity_b)
```

---

## PART 5: OPERATIONAL EXCELLENCE & STALENESS MITIGATION

```python
# app/XNAi_rag_app/expert_router.py

class ExpertRouter:
    """Route documents to appropriate domain experts"""
    
    EXPERTS = {
        'science': {
            'domain': 'science',
            'experts': ['physicist', 'chemist', 'biologist'],
            'knowledge_path': '/knowledge/science',
            'quality_threshold': 0.8,
        },
        'coding': {
            'domain': 'coding',
            'experts': ['software_engineer', 'devops_engineer', 'data_scientist'],
            'knowledge_path': '/knowledge/coding',
            'quality_threshold': 0.7,
        },
        'esoteric': {
            'domain': 'esoteric',
            'experts': ['occultist', 'scholar'],
            'knowledge_path': '/knowledge/esoteric',
            'quality_threshold': 0.75,
        },
        'classic': {
            'domain': 'classic',
            'experts': ['librarian', 'scholar', 'philologist'],
            'knowledge_path': '/knowledge/classic',
            'quality_threshold': 0.85,
        },
    }
    
    def route_query(self, query: str, user_domain: str = None) -> dict:
        """
        Route query to most appropriate expert knowledge base
        
        1. Detect domain from query (using LLM or keyword matching)
        2. If user_domain specified, prefer that
        3. Return expert info and knowledge_path
        """
        
        # Detect domain from query
        detected_domain = self._detect_domain(query)
        
        # Override with user preference if provided
        domain = user_domain or detected_domain
        
        if domain not in self.EXPERTS:
            domain = 'librarian'  # Fallback to cross-domain
        
        expert_info = self.EXPERTS[domain]
        
        return {
            'domain': domain,
            'experts': expert_info['experts'],
            'knowledge_path': expert_info['knowledge_path'],
            'quality_threshold': expert_info['quality_threshold'],
        }
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text using keyword matching"""
        keywords = {
            'science': ['equation', 'law', 'physics', 'molecule', 'atom', 'experiment'],
            'coding': ['code', 'function', 'algorithm', 'debug', 'compiler', 'api'],
            'esoteric': ['tarot', 'ritual', 'energy', 'chakra', 'aura', 'sigil'],
            'classic': ['novel', 'poetry', 'philosophy', 'ancient', 'manuscript'],
        }
        
        text_lower = text.lower()
        scores = {}
        
        for domain, terms in keywords.items():
            scores[domain] = sum(1 for term in terms if term in text_lower)
        
        # Return domain with highest score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'uncategorized'
```

### 4.2 Symlink vs Copy Strategy

```python
# Symlink strategy (recommended for most cases)
class SourceManagement:
    """Manage source documents: copy, symlink, or reference"""
    
    def ingest_from_library(self, source_file: Path, domain: str, method: str = 'symlink'):
        """
        Ingest source from /library/ to /knowledge/
        
        Methods:
        - 'symlink': Create symbolic link (save space, track original)
        - 'copy': Copy file (isolation, durability)
        - 'reference': Just store reference (minimal space)
        """
        
        if method == 'symlink':
            # Create symlink: /knowledge/domain/ -> /library/source
            knowledge_path = Path(f"/knowledge/{domain}/{source_file.stem}_symlink")
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create relative symlink for portability
            relative_source = source_file.relative_to(knowledge_path.parent)
            knowledge_path.symlink_to(relative_source)
            
            logger.info(f"🔗 Symlinked: {source_file} -> {knowledge_path}")
            
        elif method == 'copy':
            # Copy file for isolation
            knowledge_path = Path(f"/knowledge/{domain}/{source_file.name}")
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(source_file, knowledge_path)  # Preserve metadata
            
            logger.info(f"📋 Copied: {source_file} -> {knowledge_path}")
        
        elif method == 'reference':
            # Just store reference
            metadata = {
                'source_path': str(source_file),
                'domain': domain,
                'imported_at': datetime.utcnow().isoformat(),
            }
            
            redis_client.hset(
                f"source_reference:{source_file.stem}",
                mapping=metadata
            )
            
            logger.info(f"📌 Referenced: {source_file}")
```

**When to use each:**
- **Symlink** (Recommended): Save space, maintain single source of truth, track lineage
- **Copy**: Need isolation, worried about original deletion, archival purposes
- **Reference**: Extremely large files, external URIs, streaming sources

### 4.3 Version Control & Lineage

```python
# Data lineage tracking (similar to MLOps)

class SourceLineage:
    """Track source document versions and lineage"""
    
    def track_ingestion(self, source_file: Path, output_vectors: str, metadata: dict):
        """Create lineage record for data provenance"""
        
        lineage_record = {
            'source': {
                'path': str(source_file),
                'hash': self._hash_file(source_file),
                'size': source_file.stat().st_size,
                'timestamp': source_file.stat().st_mtime,
            },
            'processing': {
                'ingestion_time': datetime.utcnow().isoformat(),
                'chunks_created': metadata['vector_count'],
                'embedding_model': metadata['embedding_model'],
                'chunking_strategy': metadata['chunk_size'],
            },
            'output': {
                'vector_path': output_vectors,
                'vector_count': metadata['vector_count'],
                'faiss_hash': self._hash_directory(output_vectors),
            },
            'metadata': metadata,
        }
        
        # Store to Redis for queryable lineage
        redis_client.hset(
            f"lineage:{source_file.stem}",
            mapping=json.dumps(lineage_record, default=str)
        )
        
        # Also log to audit trail
        logger.info(f"📊 Lineage tracked: {source_file} -> {output_vectors}")
        
        return lineage_record
    
    def get_sources_for_vector(self, vector_id: str) -> dict:
        """Reverse lookup: which sources contributed to this vector?"""
        # Query Redis for lineage
        pass
```

### 5.1 Mitigating Knowledge Staleness

**The Primary Threat:** Knowledge base information becomes outdated, causing hallucinations or incorrect recommendations

**Multi-Layered Mitigation Strategy:**

```python
class StalenessControl:
    """Four-pronged approach to keep knowledge fresh"""
    
    # STRATEGY 1: TTL Policies (Time-To-Live)
    TTL_POLICIES = {
        'science': 365 * 24,           # Research: 1 year before re-review
        'coding': 90 * 24,             # Code: 90 days (APIs change rapidly)
        'esoteric': 730 * 24,          # Classical: 2 years (slower evolution)
        'finance': 1 * 24,             # Finance: 1 day (stock prices change hourly!)
    }
    
    # STRATEGY 2: Relationship Decay (Time-Based Edge Weights)
    def relationship_strength_at_time(self, edge, now: datetime) -> float:
        """Model temporal decay of relationship strength
        
        Example: "Python 2.7 is the standard" (2010)
                 → Strength 1.0 in 2010
                 → Strength 0.01 in 2024 (End-of-life 2020)
        
        Formula: strength_t = strength_0 * exp(-λ * time_elapsed)
        """
        time_elapsed = (now - edge.created_at).days
        decay_rate = 0.01  # Adjust per domain
        return edge.strength * math.exp(-decay_rate * time_elapsed)
    
    # STRATEGY 3: Continuous Refresh (Automatic Detection)
    def auto_refresh_on_source_change(self, source_file: Path):
        """Detect source file changes and trigger re-indexing
        
        Not just file timestamp—track actual content changes (deltas)
        """
        current_hash = self._compute_file_hash(source_file)
        previous_hash = redis.get(f"source_hash:{source_file}")
        
        if current_hash != previous_hash:
            logger.info(f"🔄 Source changed: {source_file}")
            # Trigger declarative re-indexing (only affected chunks)
            self.ingestion_queue.enqueue_incremental_reindex(source_file)
            redis.set(f"source_hash:{source_file}", current_hash)
    
    # STRATEGY 4: Near-Real-Time Updates
    async def stream_updates_to_knowledge_base(self, source_stream):
        """Detect and apply updates with minimal delay
        
        For rapidly evolving domains (finance, news, code):
        - Pub/Sub pattern for real-time updates
        - Cache invalidation as soon as source changes
        - No waiting for batch jobs
        """
        async for event in source_stream:
            if event.type == 'FILE_MODIFIED':
                await self.invalidate_cache(event.file_path)
                await self.queue_incremental_reindex(event.file_path)
            
            # Publish to subscribers (e.g., "finance_quotes_channel")
            await redis_pubsub.publish(f"{event.domain}_updates", event)
```

**Per-Domain Staleness Strategy:**

| Domain | TTL | Decay Rate | Refresh Trigger |
|--------|-----|-----------|-----------------|
| **Finance** | 1 hour | Fast (0.05) | Real-time market data feed |
| **Coding** | 3 days | Medium (0.01) | GitHub webhooks on code changes |
| **Science** | 6 months | Slow (0.001) | Periodic literature review |
| **Esoteric** | 2 years | Very slow (0.0001) | Manual expert curation |
| **News** | 24 hours | Medium (0.02) | Crawler + pub/sub |

### 5.2 Data Versioning & Lineage Traceability (Git-Like Strategy)

**Problem:** When a response is generated, can you trace it back to:
- Which specific document version was used?
- Who modified the knowledge base?
- When did the change happen?
- Can we rollback to previous knowledge state?

**Solution: Use lakeFS for Document Store Versioning**

```python
class DataLineageTracking:
    """MLOps-style versioning for knowledge base"""
    
    def ingest_with_versioning(self, 
                               source_file: Path, 
                               domain: str,
                               git_commit_hash: str = None,
                               docker_image_hash: str = None):
        """Record full ingestion lineage"""
        
        # Compute hash of source material
        source_hash = self._sha256(source_file)
        
        # Ingest and create vectors
        vectors = self._create_embeddings(source_file)
        vector_set_hash = self._hash_vectors(vectors)
        
        # Create immutable lineage record
        lineage = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': {
                'file': source_file.name,
                'path': str(source_file),
                'hash': source_hash,
                'size': source_file.stat().st_size,
            },
            'processing': {
                'domain': domain,
                'chunking_strategy': 'semantic',
                'embedding_model': 'all-MiniLM-L12-v2',
                'chunk_count': len(vectors),
            },
            'output': {
                'vector_set_hash': vector_set_hash,
                'faiss_index_path': f'/knowledge/{domain}/{vector_set_hash[:8]}',
                'vector_count': len(vectors),
            },
            'environment': {
                'git_commit': git_commit_hash,  # Reproducibility
                'docker_image': docker_image_hash,
                'ingestion_script_hash': self._hash_ingestion_code(),
            },
        }
        
        # Store lineage in version control (lakeFS or git-lfs)
        self._store_versioned_lineage(domain, lineage)
        
        # Enable queries like:
        # "Show me all embeddings created from this source version"
        # "Rollback to knowledge base state from 2 weeks ago"
        # "Audit: who changed this document?"
        
        return lineage
    
    def rollback_to_timestamp(self, domain: str, timestamp: str) -> dict:
        """Rollback knowledge base to previous state
        
        Use case: A bad ingestion corrupted results → rollback to last good version
        """
        previous_version = self.version_store.get_version_at(domain, timestamp)
        self.knowledge_store.restore_version(previous_version)
        logger.info(f"✅ Rolled back {domain} to {timestamp}")
        return previous_version
    
    def trace_response_to_sources(self, response_id: str) -> List[dict]:
        """Reverse lookup: show which documents contributed to this response
        
        Use case: Compliance audit - prove response is grounded in specific sources
        """
        # Query: which vector chunks were used in this response?
        chunks = self.response_cache.get_chunks_for_response(response_id)
        
        # For each chunk, get its lineage
        sources = []
        for chunk in chunks:
            lineage = self.lineage_store.get_lineage(chunk.id)
            sources.append({
                'document': lineage['source']['file'],
                'document_hash': lineage['source']['hash'],
                'chunk_id': chunk.id,
                'ingested_at': lineage['timestamp'],
                'embedding_model': lineage['processing']['embedding_model'],
            })
        
        return sources
```

**Git-like Versioning Structure:**

```
/knowledge/{domain}/.versions/
├── v1.0_2024-01-01T10:00:00Z/
│   ├── faiss_index/
│   ├── metadata.json
│   └── lineage.json
├── v1.1_2024-01-08T15:30:00Z/
│   ├── faiss_index/
│   ├── metadata.json
│   └── lineage.json
└── HEAD → v1.1_2024-01-08T15:30:00Z  # Current version
```

### 5.3 Observability & AI-Specific Metrics

**Beyond Infrastructure:** Monitor AI-specific signals for data drift and performance degradation

```python
class RAGObservability:
    """Production-grade monitoring for evolving knowledge base"""
    
    CRITICAL_METRICS = {
        # INGESTION PERFORMANCE
        'ingestion_latency_ms': 'How long to process new documents?',
        'ingestion_throughput': 'Documents/hour?',
        'chunk_count_per_doc': 'Is chunking strategy stable?',
        
        # RETRIEVAL QUALITY
        'context_retrieval_relevance': 'Are retrieved chunks actually relevant? (0-1)',
        'retrieval_latency_ms': 'How fast are queries answered?',
        'chunk_reranking_score': 'Confidence in chunk ordering?',
        'vector_similarity_distribution': 'Are scores concentrated or spread?',
        
        # GENERATION QUALITY (AI-Specific)
        'groundedness_score': 'Is response grounded in retrieved docs? (0-1)',
        'hallucination_rate': 'What % of responses invent facts?',
        'response_coherence': 'Is response well-formed?',
        'expert_domain_accuracy': 'Science expert: chemistry accuracy %?',
        
        # STALENESS MONITORING
        'source_freshness_age_days': 'Oldest source in active KB',
        'document_modification_rate': 'How often sources change?',
        'ttl_expiration_rate': 'How many docs exceed TTL?',
        
        # USER FEEDBACK
        'user_satisfaction_thumbs_up': 'What % of responses are marked good?',
        'expert_review_rate': 'What % of responses reviewed by experts?',
        'correction_rate': 'How often do experts correct suggestions?',
    }
    
    def instrument_response_generation(self, response: str, retrieved_chunks: List[str]):
        """Measure groundedness and hallucination in real-time"""
        
        # Groundedness: Does each claim in response have supporting chunk?
        claims = self._extract_claims(response)
        groundedness_scores = []
        
        for claim in claims:
            max_similarity = max(
                self._semantic_similarity(claim, chunk) 
                for chunk in retrieved_chunks
            )
            groundedness_scores.append(max_similarity)
        
        avg_groundedness = sum(groundedness_scores) / len(claims)
        
        # Log to observability backend
        prometheus_metrics.hist(
            'rag_response_groundedness',
            value=avg_groundedness,
            labels={'domain': self.domain}
        )
        
        # Alert if falling below threshold
        if avg_groundedness < 0.6:
            logger.warning(f"⚠️  Low groundedness: {avg_groundedness:.2f}")
            alert_ops_team("Potential hallucination spike")
        
        return avg_groundedness
    
    def track_embedding_drift(self):
        """Monitor if embedding model producing different vectors over time
        
        Signal of: model version changes, tokenizer bugs, data quality issues
        """
        # Periodically re-embed same chunks
        # Compare new embeddings to stored embeddings
        # If cosine similarity < threshold, model drift detected
        
        pass
    
    def setup_user_feedback_loop(self):
        """Capture expert reviews to continuously improve system
        
        "This response was: [Excellent] [Good] [Needs Work] [Wrong]"
        Then correlate feedback with:
        - Which chunks were retrieved?
        - What domain was the query?
        - Which embedding model?
        - What generation model?
        """
        pass
```

**Monitoring Dashboard Queries:**

```sql
-- Alert: Groundedness dropping (potential hallucinations)
SELECT 
    domain,
    AVG(groundedness_score) as avg_groundedness,
    STDDEV(groundedness_score) as variability
FROM rag_metrics
WHERE timestamp > NOW() - INTERVAL 24 HOUR
GROUP BY domain
HAVING avg_groundedness < 0.65

-- Trend: Which domains need re-indexing?
SELECT 
    domain,
    MAX(document_age_days) as oldest_source,
    COUNT(*) as doc_count,
    AVG(modification_frequency) as churn_rate
FROM knowledge_base_health
GROUP BY domain
ORDER BY oldest_source DESC

-- Alert: Embedding model drift
SELECT 
    chunk_id,
    old_embedding_hash,
    new_embedding_hash,
    cosine_similarity,
    CASE WHEN cosine_similarity < 0.95 THEN 'DRIFT_DETECTED' ELSE 'OK' END as status
FROM embedding_drift_check
WHERE timestamp > NOW() - INTERVAL 1 HOUR
```

---

## PART 6: MULTI-AGENT ORCHESTRATION

### Phase 1 (Immediate - Week 1-2)
```
Priority: CRITICAL
Effort: Low
Impact: High

✓ Add environment layering (config.dev.toml, config.prod.toml)
✓ Implement file watcher for /library/incoming/
✓ Add domain detection (simple keyword-based)
✓ Create domain subdirectories in /knowledge/
✓ Add metadata JSON tracking for ingested files
```

### Phase 2 (Short-term - Week 3-4)
```
Priority: HIGH
Effort: Medium
Impact: High

✓ Implement hot-reload for config changes
✓ Add config audit trail logging
✓ Create domain-specific handlers (CodeHandler, ScienceHandler, etc.)
✓ Implement ExpertRouter for query routing
✓ Add version control to ingested documents
```

### Phase 3 (Medium-term - Month 2)
```
Priority: MEDIUM
Effort: High
Impact: Medium

✓ Secrets management integration (Vault, AWS Secrets Manager)
✓ Advanced LLM-based domain classification
✓ Knowledge base quality metrics
✓ Cross-domain curation interface
✓ Ingestion performance optimization
```

### Phase 4 (Long-term - Month 3+)
```
Priority: LOW
Effort: High
Impact: Medium

✓ Multi-version knowledge management
✓ Advanced lineage tracking (MLOps-style)
✓ A/B testing framework for ingestion strategies
✓ Integration with external knowledge sources (APIs)
```

---

## PART 6: KEY DECISIONS & TRADE-OFFS

### Question 1: Should files go to /library/ first or directly to /knowledge/?

**Answer:** ✅ **Use /library/ as intake** → then route to /knowledge/

**Reasoning:**
```
/library/
├── Single source of truth for raw documents
├── Allows version history (git/git-lfs compatible)
├── Enables re-processing with new handlers
├── Audit trail preserved (original unchanged)
└── Supports parallel expert curation

/knowledge/
├── Domain-specific curated collections
├── FAISS indices (ephemeral, can regenerate)
├── Can be symlinked back to /library/ for linkage
└── Expert annotations and reviews
```

**Flow:** User drops file → /library/ → Auto-detected as domain X → Queued → Processed by domain handler → Indexed in /knowledge/X/ → Expert reviews & annotates

---

### Question 2: Symlinks vs Copies?

**Answer:** ✅ **Prefer symlinks with copy fallback**

```python
# Default strategy:
if source_is_large:  # > 100MB
    use_symlink()  # Save space
elif source_is_critical:
    copy_file()    # Ensure durability
else:
    use_symlink()  # Default
```

**Trade-offs:**
```
SYMLINKS                          COPIES
✅ Save 95%+ space               ✅ Guarantee durability
✅ Single source of truth        ✅ Isolation from changes
✅ Track lineage easily          ✅ No broken link issues
❌ Fragile to moves              ❌ Duplicate storage
❌ Require cleanup               ❌ Sync challenges
```

---

### Question 3: One main FAISS index or per-domain indices?

**Answer:** ✅ **One main + per-domain indices**

```
ARCHITECTURE:
/data/faiss_index/              # Main unified index (all domains)
├── Comprehensive search
├── Cross-domain queries
└── Used by general RAG

/knowledge/{domain}/            # Domain-specific indices
├── science/faiss_index         # Physics expert's KB
├── coding/faiss_index          # Coding expert's KB
├── esoteric/faiss_index        # Esoteric expert's KB
└── classic/faiss_index         # Literature expert's KB

QUERY FLOW:
User asks: "What is quantum entanglement?"
    ↓
detect domain: "science"
    ↓
Route to: /knowledge/science/faiss_index
    ↓
Return: Top 5 science-curated results
```

**Benefits:**
- Main index for fuzzy/cross-domain queries
- Domain indices for expert-specific queries
- Reduces false positives (science queries don't match code)
- Easier to update individual domains

---

### Question 4: How to handle curation expert workflow?

**Answer:** ✅ **Quality tiering with expert review gates**

```
INGESTION → QUALITY TIERS → EXPERT REVIEW → PUBLICATION

1. AUTO-TIER (quality_score < 0.5):
   - Metadata extracted
   - Document indexed
   - Requires expert review before use

2. CURATED-TIER (0.5 <= quality_score < 0.8):
   - Indexed and available
   - Expert can annotate/improve
   - Versioning tracked

3. VERIFIED-TIER (quality_score >= 0.8):
   - Published immediately
   - Still open to expert improvements
   - High confidence for RAG

EXPERT REVIEW INTERFACE:
GET /api/expert/review-queue/{domain}
  → List documents needing review
  
PUT /api/expert/review/{doc_id}
  → {"quality_score": 0.85, "annotation": "...", "action": "publish"}
  
GET /api/knowledge/{domain}?quality_min=0.7
  → Only retrieve tier-2 and tier-3 documents
```

---

## PART 7: SECURITY & COMPLIANCE CONSIDERATIONS

### Configuration Security

```python
class SecureConfigManager:
    """Secure config handling with secrets protection"""
    
    SENSITIVE_KEYS = {
        'redis_password',
        'api_key',
        'secret_key',
        'aws_access_key',
        'vault_token',
    }
    
    def redact_sensitive_data(self, config: dict) -> dict:
        """Return safe copy for logging/viewing"""
        safe_config = json.loads(json.dumps(config))  # Deep copy
        
        for key in self.SENSITIVE_KEYS:
            if key in safe_config:
                safe_config[key] = "[REDACTED]"
        
        return safe_config
    
    def validate_secrets_not_in_logs(self):
        """Verify sensitive data never reaches logs"""
        # Check for AWS keys, API tokens, etc. in logfiles
        pass
```

### Knowledge Base Security

```python
class KnowledgeAccessControl:
    """Control access to curated knowledge by domain"""
    
    ACL = {
        'science': ['science_expert', 'librarian', 'admin'],
        'coding': ['coding_expert', 'devops_engineer', 'admin'],
        'esoteric': ['esoteric_expert', 'librarian'],  # More restricted
        'classic': ['anyone'],  # Public domain
    }
    
    def can_access(self, user_role: str, domain: str) -> bool:
        allowed_roles = self.ACL.get(domain, [])
        return user_role in allowed_roles
```

---

## PART 8: ANSWERS TO YOUR SPECIFIC QUESTIONS

### Q1: Enterprise-Grade Config Improvements?

✅ **Add 4 key features:**
1. Environment layering (dev/staging/prod with override hierarchy)
2. Hot-reload via file watching (config changes without restart)
3. Audit trail (who changed what, when, why)
4. Secrets management (AWS Secrets Manager or HashiCorp Vault)

**Quick Win:** Create `/config/config.prod.toml` override for production memory/rate-limit settings

---

### Q2: Robust Ingestion for Manual + Auto + Curation?

✅ **Three separate flows with central queue:**

1. **Manual:** User drops files in `/library/incoming/` → file watcher detects → queues
2. **Auto:** Scheduled crawler finds new files → auto-detects domain → queues  
3. **Curation:** Curation worker processes queue → applies domain handlers → publishes to `/knowledge/`

**Queue system:** Use Redis FIFO with metadata tracking for visibility

---

### Q3: Can you categorize sources into expert subfolders?

✅ **YES - recommended architecture:**

```
/library/
├── science/          # Science expert's raw library
├── coding/           # Coding expert's raw library
├── esoteric/         # Esoteric expert's raw library
├── classic/          # Librarian's raw library
└── incoming/         # Auto-sort from here

/knowledge/
├── science/          # Science expert's CURATED KB (FAISS)
├── coding/           # Coding expert's CURATED KB (FAISS)
├── esoteric/         # Esoteric expert's CURATED KB (FAISS)
└── classic/          # Librarian's CURATED KB (FAISS)
```

**Flow:** `/library/{domain}/*.md` → processing → `/knowledge/{domain}/faiss_index/`

---

### Q4: Should all sources go to main directory first?

✅ **Two options (pick one):**

**Option A (RECOMMENDED): Inbox pattern**
```
/library/incoming/          ← All files drop here
    ↓ Auto-detect domain
/library/{domain}/          ← Auto-sort by type
    ↓ Expert review
/knowledge/{domain}/        ← Published curated KB
```

**Option B: Direct drop**
```
/library/{domain}/          ← Users drop in domain subfolder
    ↓ Auto-detect if domain unclear
/knowledge/{domain}/        ← Published curated KB
```

**Recommendation:** Use Option A for centralized intake, prevents miscategorization by users

---

### Q5: Can domain experts drop files directly into domain subfolder?

✅ **YES - use watchdog with recursive monitoring:**

```python
observer.schedule(
    IngestionHandler(),
    path="/library",
    recursive=True  # ← Watch all subdirectories
)

# Detects both:
# /library/science/new_paper.pdf
# /library/incoming/new_paper.pdf
```

**Benefits:**
- Science expert can organize their own sources
- Automatic re-detection provides safety net
- Audit trail shows who dropped what where

---

## PART 9: IMPLEMENTATION STARTER CODE

### File 1: Environment Layering

Create `/config/` directory:
```bash
mkdir -p /config
cp config.toml /config/config.base.toml
touch /config/config.dev.toml
touch /config/config.prod.toml
```

### File 2: Env-aware Config Loader

```python
# app/XNAi_rag_app/config_loader_v2.py

import os
from pathlib import Path
import toml
from typing import Dict, Any

class EnvAwareConfigLoader:
    """Load config with environment-based overrides"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('ENV', 'dev')
    
    def load(self) -> Dict[str, Any]:
        """Load config: base + env-specific + env vars"""
        
        config_dir = Path('/config')
        
        # 1. Load base config
        base_config = toml.load(config_dir / 'config.base.toml')
        
        # 2. Load environment-specific config
        env_config_path = config_dir / f'config.{self.env}.toml'
        if env_config_path.exists():
            env_config = toml.load(env_config_path)
            # Deep merge
            base_config = self._deep_merge(base_config, env_config)
        
        # 3. Apply environment variables
        base_config = self._apply_env_overrides(base_config)
        
        return base_config
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge override into base"""
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self, config: dict) -> dict:
        """Override config values from environment variables"""
        # ENV_SECTION_KEY=value becomes config[section][key]=value
        for env_key, env_val in os.environ.items():
            if env_key.startswith('CONFIG_'):
                parts = env_key.split('_')[1:]  # Remove CONFIG_
                if len(parts) >= 2:
                    section = '_'.join(parts[:-1]).lower()
                    key = parts[-1].lower()
                    
                    if section not in config:
                        config[section] = {}
                    
                    config[section][key] = self._parse_value(env_val)
        
        return config
    
    @staticmethod
    def _parse_value(val: str):
        """Parse environment variable to appropriate type"""
        if val.lower() in ('true', 'false'):
            return val.lower() == 'true'
        if val.isdigit():
            return int(val)
        if val.replace('.', '', 1).isdigit():
            return float(val)
        return val

# Usage:
loader = EnvAwareConfigLoader(env='prod')
config = loader.load()
```

---

## PART 6: MULTI-AGENT ORCHESTRATION

### 6.1 Gated Mixture-of-Experts (MoXpert) Architecture

**For multi-domain system, route query to appropriate expert agent:**

```python
class MixtureOfExperts:
    """Gated expert selection based on query characteristics"""
    
    EXPERT_MODULES = {
        'science': {
            'experts': ['physicist', 'chemist', 'biologist'],
            'retrieval_type': 'depth_search',
            'knowledge_path': '/knowledge/science',
        },
        'coding': {
            'experts': ['software_engineer', 'devops_engineer'],
            'retrieval_type': 'grep_first',
            'knowledge_path': '/knowledge/coding',
        },
        'esoteric': {
            'experts': ['occultist', 'scholar'],
            'retrieval_type': 'breadth_search',
            'knowledge_path': '/knowledge/esoteric',
        },
    }
    
    def route_query(self, query: str) -> dict:
        """Intelligent routing: which expert(s) should handle this?"""
        domain_scores = self._score_domain_match(query)
        top_domains = [d for d, s in domain_scores.items() if s > 0.3]
        
        if len(top_domains) == 1:
            expert_config = self.EXPERT_MODULES[top_domains[0]]
            return {
                'primary_expert': top_domains[0],
                'multi_expert': False,
            }
        else:
            return {
                'primary_expert': None,
                'contributing_experts': top_domains,
                'multi_expert': True,
                'synthesis_required': True,
            }
```

---

## PART 7: DOMAIN-SPECIFIC INGESTION STRATEGIES

### 7.1 Code RAG: Beyond Vector Search

**Problem:** Code changes rapidly; re-indexing entire codebase is inefficient. Pure vector RAG fails on structured code.

**Solution: Grep-First + AST Parsing (from research)**

```python
class CodeRAGHandler:
    """Specialized ingestion for code repositories"""
    
    def ingest_code_smart(self, repo_path: Path):
        """
        Three-layer retrieval instead of naive vector search:
        
        Layer 1: Grep-first (fast symbol lookup using ripgrep)
        Layer 2: AST parsing (extract semantic structure)
        Layer 3: Vector reranking (semantic similarity)
        """
        
        # LAYER 1: Build symbol cache (symbols, not full vectors)
        symbols = self._extract_symbols_with_tree_sitter(repo_path)
        self._build_symbol_index(symbols)  # Store in Redis
        
        # Example: looking for "authenticate" function
        # grep: "def authenticate" → fast list of files
        # AST: Parse those files → Extract function signatures, imports, types
        # Vector: Rerank by semantic relevance
        
        # LAYER 2: Build dependency graph (function calls, imports)
        self._build_call_graph(repo_path)
        self._build_import_graph(repo_path)
        
        # LAYER 3: Create vectors ONLY for complex entities (whole functions)
        # NOT for individual tokens
        
        return {
            'symbol_index': symbols,
            'call_graph': self.call_graph,
            'import_graph': self.import_graph,
            'vectors': self.vectors,
        }
    
    def retrieve_code_smart(self, query: str):
        """Fast code retrieval: grep → AST → vector rerank"""
        
        # Step 1: Grep for potential matches (FAST)
        potential_files = self._grep_search(query)
        
        if len(potential_files) == 0:
            return []
        
        # Step 2: Parse matched files with tree-sitter (extract structure)
        candidates = []
        for file in potential_files:
            tree = tree_sitter.parse(file)
            # Extract functions/classes/definitions near matches
            candidates.extend(self._extract_relevant_nodes(tree, query))
        
        # Step 3: Rerank using semantic similarity (OPTIONAL if too many)
        if len(candidates) > 20:
            candidates = self._vector_rerank(candidates, query, top_k=10)
        
        return candidates
```

**Key Advantage:** 100x faster than pure vector RAG on code; maintains structural context

### 7.2 Scientific Document Ingestion

**Specialized extraction for research papers:**

```python
class ScientificDocumentHandler:
    """Extract equations, citations, figures from papers"""
    
    def ingest_paper(self, paper_path: Path):
        """Extract structured elements from scientific paper"""
        
        # 1. LaTeX equation extraction
        equations = self._extract_equations(paper_path)
        # Store with metadata: equation hash, figure number, page
        
        # 2. Named entity extraction (proteins, chemicals, genes)
        entities = self._extract_entities_specialized(paper_path)
        # NER model trained on biomedical texts (vs generic NER)
        
        # 3. Citation network building
        citations = self._extract_citations(paper_path)
        # Create edges: this_paper → cited_papers
        
        # 4. Figure and table extraction (not just text)
        figures = self._extract_figures(paper_path)
        figures_ocr = self._ocr_figures(figures)
        
        # 5. Chunk respecting scientific structure
        chunks = self._semantic_chunk_paper(paper_path)
        # Chunks = abstract + introduction + methodology + results + conclusion
        # NOT arbitrary token splits
        
        return {
            'equations': equations,
            'entities': entities,
            'citations': citations,
            'figures': figures_ocr,
            'chunks': chunks,
            'quality_score': self._compute_paper_quality(paper_path),
        }
    
    def _extract_entities_specialized(self, paper: Path) -> List[dict]:
        """Use biomedical NER model (90%+ accuracy on medical texts)"""
        from transformers import pipeline
        
        # Specialized model: bert-base-biomedical
        ner = pipeline(
            "ner",
            model="allenai/scibert_scivocab_cased",
            aggregation_strategy="simple"
        )
        
        text = self._extract_text(paper)
        entities = ner(text)
        
        return [
            {
                'text': e['word'],
                'type': e['entity_group'],
                'confidence': e['score'],
            }
            for e in entities
            if e['score'] > 0.8
        ]
```

### 7.3 Esoteric Document Ingestion

**Handle archaic language, symbolic notation:**

```python
class EsotericDocumentHandler:
    """Process esoteric/occult texts with symbolic preservation"""
    
    def ingest_esoteric_text(self, text_path: Path):
        """Extract symbolic notation while preserving archaic language"""
        
        # 1. Symbol normalization (but preserve original)
        symbols = self._extract_symbols(text_path)
        normalized_symbols = self._normalize_symbols(symbols)
        
        # Map: ♀ (female symbol) → canonical "female_principle"
        # But also preserve: original text includes ♀
        
        # 2. Preserve archaic language (don't modernize)
        text = self._read_text(text_path)
        # Store as-is, don't autocorrect "hath" → "has"
        
        # 3. Create ontology mappings
        # "the Great Work" → links to hermetic_alchemy_tradition
        # "the Tree of Life" → links to kabbalah_system
        
        # 4. Extract correspondences (table-based metadata)
        correspondences = self._extract_correspondence_tables(text_path)
        # Store as graph edges for expert traversal
        
        # 5. Semantic chunking respecting scholarly structure
        chunks = self._chunk_esoteric_text(text_path)
        # Respect: chapter, section, commentary boundaries
        
        return {
            'original_text': text,
            'symbols': {
                'original': symbols,
                'normalized': normalized_symbols,
            },
            'ontology_links': self._create_ontology_mappings(text),
            'correspondences': correspondences,
            'chunks': chunks,
        }
    
    def _normalize_symbols(self, symbols: List[str]) -> Dict[str, str]:
        """Map symbols to standardized forms
        
        ♀ → female, ♂ → male
        ☿ → mercury, ☉ → sun
        But track original form for reference
        """
        symbol_map = {
            '♀': 'female',
            '♂': 'male',
            '☿': 'mercury',
            '☉': 'sun',
            '☾': 'moon',
            '♃': 'jupiter',
            '♄': 'saturn',
            '♅': 'uranus',
            '♆': 'neptune',
            '♇': 'pluto',
        }
        
        return {sym: symbol_map.get(sym, sym) for sym in symbols}
```

---

## PART 8: IMPLEMENTATION ROADMAP (REFINED)

### Phase 1 (Immediate - Week 1-2)
```
Priority: CRITICAL
Effort: Low
Impact: Immediate

✓ Add declarative change detection (hash-based delta tracking)
✓ Implement semantic chunking (respect boundaries)
✓ Metadata enrichment (40% of effort - author, date, topic, confidence)
✓ File watcher for /library/incoming/
✓ Domain detection (keyword + optional LLM)
✓ Create domain subdirectories in /knowledge/
```

### Phase 2 (Short-term - Week 3-4)
```
Priority: HIGH
Effort: Medium
Impact: High

✓ Implement incremental re-indexing (don't rebuild entire FAISS)
✓ Add staleness control (TTL policies per domain)
✓ Create domain-specific handlers (Code → AST, Science → equations, etc.)
✓ Implement ExpertRouter (MoXpert gating)
✓ Add observability metrics (groundedness, hallucination rate)
✓ Implement lakeFS-style versioning for knowledge
```

### Phase 3 (Medium-term - Month 2)
```
Priority: MEDIUM
Effort: High
Impact: High

✓ Hybrid retrieval: LightRAG (vectors + knowledge graph)
✓ Multi-agent orchestration (Science + Coding + Esoteric agents)
✓ Controlled vocabulary / ontology framework
✓ User feedback loops (expert reviews)
✓ Advanced graph traversal (depth vs breadth search)
```

### Phase 4 (Long-term - Month 3+)
```
Priority: MEDIUM
Effort: High
Impact: Ongoing

✓ Multi-version knowledge management
✓ Advanced hallucination detection via grounding metrics
✓ Integration with external knowledge sources (APIs)
✓ Quantum-agentic orchestration (when applicable)
```

---

## CONCLUSION

Your stack is **production-ready** but has room for **enterprise-grade maturity**. This refined strategy incorporates advanced patterns from autonomous knowledge ecosystem research:

1. **Declarative Ingestion:** Automatic delta detection, not manual re-indexing
2. **Advanced Retrieval:** Hybrid vector + graph (LightRAG), not pure vector search
3. **Multi-Agent System:** Gated MoE routing to specialized experts, not monolithic
4. **Operational Excellence:** Staleness control, data versioning, AI-specific observability
5. **Domain-Specific Handling:** Code (grep-first), Science (equations), Esoteric (symbols)

**Recommend starting with Phase 1** - focus on declarative change detection and semantic chunking with rich metadata. This provides the foundation for all subsequent improvements.

**Key Insight from Research:** 40% of development time should focus on metadata strategy, as it's the primary lever for retrieval precision. Invest there first.

---

## APPENDIX: USEFUL LIBRARIES

| Feature | Library | Why |
|---------|---------|-----|
| File watching | `watchdog==6.0.0` | Cross-platform, production-tested |
| Config management | `python-dotenv` + `pydantic` | Already using both |
| Secrets | `python-vault` or `boto3` | Integration with Vault/AWS |
| Versioning | `git-lfs` | Binary file versioning |
| Task queue | `redis` + `rq` | Already have Redis |
| Data lineage | Custom tracking | MLflow too heavyweight |
| Metadata DB | `sqlite` or `redis` | Lightweight options |

---

**Report Complete.** Ready to implement. Questions? Clarifications needed?
