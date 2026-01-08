---
status: active
last_updated: 2026-01-09
category: reference
---

# Data Directories Reference

**Purpose:** Documentation of `library/` and `knowledge/` directory purposes and usage.  
**For:** Developers and AI assistants to understand data organization.

---

## Directory Structure

```
Xoe-NovAi/
├── library/          # Main local library directory
├── knowledge/        # Domain-specific expert knowledge bases
├── data/            # Runtime data (FAISS indices, Redis data)
├── backups/         # Backup storage
└── models/          # LLM models (GGUF files)
```

---

## library/ Directory

### Purpose
**Main local library directory** for storing all books, technical documents, and curated content.

### Contents
- **Manually Added Content:**
  - Books (PDF, EPUB, TXT, etc.)
  - Technical documents
  - Research papers
  - Personal notes and documents

- **Curated Content:**
  - Content added by the crawler
  - Content processed by the curation expert
  - Organized by domain/category (optional)

### Usage
- **Ingestion:** Content in `library/` is processed by `ingest_library.py`
- **FAISS Indexing:** Documents are embedded and added to FAISS vectorstore
- **RAG Retrieval:** Content is searchable via RAG queries
- **Organization:** Can be organized in subdirectories by domain/topic

### Integration
- **Crawler:** CrawlModule can save curated content to `library/`
- **Curation Worker:** Processes and enriches content before adding to `library/`
- **Ingestion Script:** `scripts/ingest_library.py` processes `library/` for FAISS

### Example Structure
```
library/
├── books/
│   ├── technical/
│   │   ├── python-programming.pdf
│   │   └── machine-learning-basics.pdf
│   └── fiction/
│       └── novel.epub
├── papers/
│   ├── arxiv/
│   │   └── quantum-computing.pdf
│   └── research/
│       └── ai-ethics.pdf
└── documents/
    └── personal-notes.md
```

### Configuration
- **Path:** `./library` (bind mount in docker-compose.yml)
- **Permissions:** Owned by appuser:1001
- **Backup:** Included in backup system (if enabled)

---

## knowledge/ Directory

### Purpose
**Domain-specific knowledge bases** for various AI experts, organized in appropriate sub-folders.

### Contents
- **Expert Knowledge Bases:**
  - Domain-specific curated content
  - Expert-specific training data
  - Specialized knowledge organized by expert/domain

- **Organization:**
  - Subdirectories for each expert/domain
  - Structured knowledge for dynamic model loading (Phase 5)
  - Expert personality and response style data

### Usage
- **Expert Loading:** Knowledge bases loaded when expert is activated
- **Domain Specialization:** Each expert has its own knowledge subdirectory
- **Dynamic Loading:** Supports Phase 5 dynamic model/expert loading
- **Voice Integration:** Expert-specific knowledge for voice-to-voice system

### Integration
- **Phase 5 (Future):** Dynamic expert loading will use `knowledge/` subdirectories
- **Voice System:** Expert-specific knowledge for voice responses
- **RAG System:** Can be indexed separately for expert-specific queries

### Example Structure
```
knowledge/
├── research-expert/
│   ├── academic-papers/
│   ├── citations/
│   └── research-methodology/
├── code-expert/
│   ├── programming-languages/
│   ├── frameworks/
│   └── best-practices/
├── writing-expert/
│   ├── style-guides/
│   ├── templates/
│   └── examples/
└── analysis-expert/
    ├── data-analysis/
    ├── visualization/
    └── statistical-methods/
```

### Configuration
- **Path:** `./knowledge` (bind mount in docker-compose.yml)
- **Permissions:** Owned by appuser:1001
- **Backup:** Included in backup system (if enabled)

---

## Key Differences

| Aspect | library/ | knowledge/ |
|--------|----------|------------|
| **Purpose** | Main library (all content) | Expert-specific knowledge bases |
| **Content** | Books, documents, papers | Domain-specific expert knowledge |
| **Organization** | Optional subdirectories | Required subdirectories by expert |
| **Usage** | General RAG retrieval | Expert-specific retrieval |
| **Ingestion** | Processed by ingest_library.py | Processed per expert (Phase 5) |
| **Future Use** | General knowledge base | Dynamic expert loading |

---

## Workflow

### Adding Content to library/
1. **Manual Addition:**
   ```bash
   cp document.pdf library/books/
   ```

2. **Via Crawler:**
   ```bash
   python3 crawl.py --curate gutenberg -c classics -q "Plato"
   # Content saved to library/ via curation worker
   ```

3. **Ingestion:**
   ```bash
   python3 scripts/ingest_library.py --library-path library/
   # Content indexed into FAISS
   ```

### Adding Content to knowledge/
1. **Manual Organization:**
   ```bash
   mkdir -p knowledge/research-expert/academic-papers
   cp paper.pdf knowledge/research-expert/academic-papers/
   ```

2. **Future (Phase 5):**
   - Expert-specific ingestion scripts
   - Dynamic loading based on expert activation
   - Automatic organization by expert domain

---

## Docker Configuration

### Volume Mounts
```yaml
volumes:
  - ./library:/library          # Main library directory
  - ./knowledge:/knowledge      # Expert knowledge bases
```

### Permissions
- Both directories owned by `appuser:1001`
- Created with correct permissions before Docker start:
  ```bash
  sudo mkdir -p library knowledge
  sudo chown -R 1001:1001 library knowledge
  ```

---

## Backup & Recovery

### Backup Configuration
- **library/:** Included in backup system (if enabled)
- **knowledge/:** Included in backup system (if enabled)
- **Backup Path:** `./backups/`
- **Retention:** Configurable in `config.toml`

### Recovery
- Restore from `backups/` directory
- Maintain directory structure
- Restore permissions (appuser:1001)

---

## Best Practices

### library/ Directory
1. **Organization:** Use subdirectories for better organization
2. **Naming:** Use descriptive filenames
3. **Formats:** Support PDF, EPUB, TXT, MD, etc.
4. **Size:** Monitor directory size (backup considerations)

### knowledge/ Directory
1. **Structure:** Organize by expert/domain
2. **Naming:** Use consistent naming conventions
3. **Content:** Curate high-quality, domain-specific content
4. **Maintenance:** Keep expert knowledge bases up-to-date

---

## Future Enhancements (Phase 5)

### Dynamic Expert Loading
- Experts load their knowledge base from `knowledge/{expert-name}/`
- Automatic indexing when expert is activated
- Lazy loading for memory efficiency

### Voice Integration
- Expert-specific voice profiles
- "Hey [Expert Name]" activation
- Expert-specific knowledge retrieval

### Multi-Expert Collaboration
- Multiple experts can access shared knowledge
- Expert-specific knowledge isolation
- Cross-expert knowledge sharing

---

**Last Updated:** 2026-01-09  
**Maintained By:** Project Team

