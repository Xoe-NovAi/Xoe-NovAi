# Xoe-NovAi Library API Integration Setup Guide
## Comprehensive Guide to Domain Categorization & Metadata Enrichment

---

## TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Supported APIs](#supported-apis)
3. [Domain Categories](#domain-categories)
4. [Dewey Decimal System](#dewey-decimal-system)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [API Key Setup](#api-key-setup)
8. [Troubleshooting](#troubleshooting)

---

## QUICK START

### No Configuration Required (Free APIs Only)
The system works out-of-the-box with NO configuration. The following free APIs are enabled by default:

```bash
# Test library integration locally
cd /home/arcana-novai/Documents/GitHub/Xoe-NovAi
python3 -c "from app.XNAi_rag_app.library_api_integrations import test_library_integration; test_library_integration()"
```

### With Docker
```bash
# System works in Docker with zero configuration
docker-compose up -d

# Test inside container
docker exec xnai_rag_api python3 -c "from app.XNAi_rag_app.library_api_integrations import test_library_integration; test_library_integration()"
```

---

## SUPPORTED APIS

### ✅ FREE APIs (No API Key Required)

#### 1. **Open Library API**
- **URL**: https://openlibrary.org/developers/api
- **What It Does**: Book search, ISBN lookup, author data, book covers
- **Rate Limit**: Unlimited (must include User-Agent header)
- **User-Agent Required**: Yes (automatically included)
- **Status**: ✅ Fully integrated

```python
from app.XNAi_rag_app.library_api_integrations import OpenLibraryClient

client = OpenLibraryClient(config)
results = client.search("Python Programming")
# Returns: List[LibraryMetadata] with title, authors, subjects
```

#### 2. **Internet Archive API**
- **URL**: https://archive.org/developers/
- **What It Does**: Full-text book search, metadata, items
- **Rate Limit**: Unlimited (reasonable rate limiting advised)
- **Status**: ✅ Fully integrated

```python
from app.XNAi_rag_app.library_api_integrations import InternetArchiveClient

client = InternetArchiveClient(config)
results = client.search("Scientific Papers")
# Returns: Books, papers, collections matching query
```

#### 3. **Library of Congress API**
- **URL**: https://www.loc.gov/apis/
- **What It Does**: Books, prints, photographs, manuscripts
- **Rate Limit**: Unlimited
- **Status**: ⚠️ Partially integrated (endpoint may have changed)

```python
from app.XNAi_rag_app.library_api_integrations import LibraryOfCongressClient

client = LibraryOfCongressClient(config)
results = client.search("American History")
```

#### 4. **Project Gutenberg (Gutendex)**
- **URL**: https://gutendex.com/
- **What It Does**: Public domain books
- **Rate Limit**: Unlimited
- **Status**: ⚠️ Partially integrated (endpoint may have changed)

```python
from app.XNAi_rag_app.library_api_integrations import ProjectGutenbergClient

client = ProjectGutenbergClient(config)
results = client.search("Classic Literature")
```

#### 5. **Free Music Archive API**
- **URL**: https://freemusicarchive.org/api
- **What It Does**: Free/open music, artist data
- **Rate Limit**: Reasonable rate limiting
- **Status**: ⚠️ Partially integrated (API availability varies)

```python
from app.XNAi_rag_app.library_api_integrations import FreeMusicArchiveClient

client = FreeMusicArchiveClient(config)
results = client.search("Electronic Music")
```

### 🔑 FREEMIUM APIs (Optional API Key)

#### 6. **Google Books API**
- **URL**: https://developers.google.com/books/docs/v1/using
- **Free Tier**: 100 queries/day
- **Requires**: API key (free)
- **Status**: ✅ Fully integrated
- **Setup**: See [API Key Setup](#api-key-setup)

```python
# With API key
client = GoogleBooksClient(config)
results = client.search("Technology Books")
```

#### 7. **ISBNdb API**
- **URL**: https://isbndb.com/api
- **Free Tier**: 100 requests/day
- **Requires**: API key (free)
- **Status**: ✅ Ready for integration
- **Setup**: See [API Key Setup](#api-key-setup)

#### 8. **New York Public Library API**
- **URL**: https://www.nypl.org/developers/apis
- **Free Tier**: Available
- **Requires**: API key (free)
- **Status**: ✅ Ready for integration
- **Setup**: See [API Key Setup](#api-key-setup)

### ❌ NOT YET INTEGRATED

- WorldCat Classify (requires OCLC setup)
- Cambridge University Library (limited API access)
- Bookworm Epub Reader (specialized use)

---

## DOMAIN CATEGORIES

### 12 Supported Domain Categories

The system automatically classifies content into these categories:

```python
class DomainCategory(str, Enum):
    CODE = "code"              # Programming, software, algorithms
    SCIENCE = "science"        # Physics, chemistry, biology, research
    DATA = "data"              # Analytics, statistics, datasets, ML
    GENERAL = "general"        # General knowledge, mixed topics
    BOOKS = "books"            # Literature, fiction, narrative
    MUSIC = "music"            # Audio, music, songs, albums
    ARCHIVES = "archives"      # Collections, archives, historical
    MANUSCRIPTS = "manuscripts"# Handwritten, original documents
    PHOTOGRAPHS = "photographs"# Images, visual media, pictures
    AUDIO = "audio"            # Audio files, podcasts, voice
    FICTION = "fiction"        # Novels, short stories, creative
    REFERENCE = "reference"    # Encyclopedia, dictionary, reference
```

### Automatic Classification Example

```python
from app.XNAi_rag_app.library_api_integrations import LibraryEnrichmentEngine

engine = LibraryEnrichmentEngine()

# Automatic classification
category, confidence = engine.domain_manager.classify(
    text="def fibonacci(n): return 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    title="Python Algorithms"
)

print(f"Category: {category.value}")      # Output: "code"
print(f"Confidence: {confidence:.2f}")    # Output: "0.85"
```

### Get All Available Categories

```python
categories = engine.domain_manager.get_all_categories()
# Output: ['code', 'science', 'data', 'general', 'books', 'music', 'archives', ...]
```

---

## DEWEY DECIMAL SYSTEM

### What is Dewey Decimal?

The Dewey Decimal Classification (DDC) is a library classification system used worldwide:
- **000-099**: Computer science, information
- **100-199**: Philosophy & psychology
- **200-299**: Religion
- **300-399**: Social sciences
- **400-499**: Language
- **500-599**: Science
- **600-699**: Technology
- **700-799**: Arts
- **800-899**: Literature
- **900-999**: History & geography

### Automatic Dewey Mapping

The system automatically maps domain categories to Dewey Decimal classifications:

```python
engine = LibraryEnrichmentEngine(config)

# Get primary Dewey classification
dewey = engine.domain_manager.domain_to_dewey(DomainCategory.CODE)
# Output: "000"

# Get Dewey suggestions
suggestions = engine.domain_manager.get_dewey_suggestion(DomainCategory.SCIENCE)
# Output: ['500', '540', '570']

# Map Dewey back to domain
category = engine.domain_manager.dewey_to_domain("500")
# Output: DomainCategory.SCIENCE
```

### Full Enrichment with Dewey

```python
enrichment = engine.classify_and_enrich(
    title="Quantum Mechanics Research",
    content="Wave functions and superposition principles...",
    author="Erwin Schrödinger"
)

# Result includes:
print(enrichment['domain_category'])      # "science"
print(enrichment['primary_dewey'])        # "500"
print(enrichment['dewey_suggestions'])    # ['500', '540', '570']
```

---

## CONFIGURATION

### Configuration Files

#### 1. `.env.library_apis` (Template)
Located in project root. Copy to `.env` and customize:

```bash
# Optional API keys (leave blank to disable)
GOOGLE_BOOKS_API_KEY=your_key_here
ISBNDB_API_KEY=your_key_here
NYPL_API_KEY=your_key_here

# Feature flags
LIBRARY_API_ENABLE_OPENLIBRARY=true
LIBRARY_API_ENABLE_GOOGLEBOOKS=true
LIBRARY_API_ENABLE_INTERNETARCHIVE=true
LIBRARY_API_ENABLE_DEWEY=true
LIBRARY_API_AUTO_CLASSIFY=true

# Performance
LIBRARY_API_RATE_LIMIT_CALLS=10
LIBRARY_API_RATE_LIMIT_PERIOD=60
LIBRARY_API_ENABLE_CACHE=true
LIBRARY_API_CACHE_TTL=3600
```

#### 2. LibraryAPIConfig (Programmatic)

```python
from app.XNAi_rag_app.library_api_integrations import LibraryAPIConfig

# Load from environment
config = LibraryAPIConfig()

# Or configure manually
config = LibraryAPIConfig(
    google_books_api_key="your_key",
    isbndb_api_key="your_key",
    enable_cache=True,
    enable_dewey_mapping=True,
    request_timeout=10,
    cache_ttl=3600
)
```

### Environment Variables

```bash
# API Keys
export GOOGLE_BOOKS_API_KEY="your_key"
export ISBNDB_API_KEY="your_key"
export NYPL_API_KEY="your_key"

# Features
export LIBRARY_API_ENABLE_DEWEY=true
export LIBRARY_API_AUTO_CLASSIFY=true

# Performance
export LIBRARY_API_REQUEST_TIMEOUT=10
export LIBRARY_API_CACHE_TTL=3600
```

---

## USAGE EXAMPLES

### Example 1: Simple Domain Classification

```python
from app.XNAi_rag_app.library_api_integrations import LibraryEnrichmentEngine

engine = LibraryEnrichmentEngine()

# Classify a document
category, confidence = engine.domain_manager.classify(
    text="Python programming guide with examples",
    title="Python for Beginners"
)

print(f"Category: {category.value}")
print(f"Confidence: {confidence:.2%}")
```

### Example 2: Complete Enrichment Workflow

```python
from app.XNAi_rag_app.library_api_integrations import LibraryEnrichmentEngine

engine = LibraryEnrichmentEngine()

# Full enrichment
result = engine.classify_and_enrich(
    title="The Pragmatic Programmer",
    content="A comprehensive guide to software development best practices...",
    author="David Thomas, Andrew Hunt"
)

# Access results
print(f"Title: {result['title']}")
print(f"Domain: {result['domain_category']}")
print(f"Confidence: {result['category_confidence']:.2%}")
print(f"Dewey: {result['primary_dewey']}")
print(f"Metadata matches: {len(result['metadata_results'])}")

if result.get('primary_metadata'):
    meta = result['primary_metadata']
    print(f"Authors: {', '.join(meta['authors'])}")
    print(f"Publisher: {meta.get('publisher')}")
    print(f"Subjects: {', '.join(meta['subjects'][:3])}")
```

### Example 3: Batch Enrichment

```python
from app.XNAi_rag_app.library_api_integrations import LibraryEnrichmentEngine

engine = LibraryEnrichmentEngine()

items = [
    {
        "title": "Python Best Practices",
        "content": "Software development guide...",
        "author": "Expert Developer"
    },
    {
        "title": "Quantum Physics",
        "content": "Wave function collapse...",
        "author": "Physics Professor"
    },
    {
        "title": "Great Novels",
        "content": "Literary analysis...",
        "author": "Literature Critic"
    }
]

# Batch process
results = engine.batch_enrich(items)

for result in results:
    print(f"{result['title']}: {result['domain_category']}")
```

### Example 4: Integrate with Crawled Documents

```python
from app.XNAi_rag_app.crawler_curation import CrawledDocument, enrich_with_library_metadata

# Create crawled document
doc = CrawledDocument(
    url="https://example.com/article",
    content="Article content here...",
    domain="general"
)

# Enrich with library metadata
enriched_doc = enrich_with_library_metadata(
    doc,
    title="Article Title",
    author="Article Author"
)

# Access enriched data
if hasattr(enriched_doc.metadata, 'enriched_library_data'):
    enrichment = enriched_doc.metadata.enriched_library_data
    print(f"Domain: {enrichment['domain_category']}")
    print(f"Dewey: {enrichment['primary_dewey']}")
```

### Example 5: Custom Domain Categories

```python
from app.XNAi_rag_app.library_api_integrations import DomainManager

manager = DomainManager()

# Add custom category
manager.add_custom_category(
    "blockchain",
    ["blockchain", "cryptocurrency", "smart contract", "ethereum", "bitcoin"]
)

# Use custom category
category, confidence = manager.classify(
    text="Blockchain technology and cryptocurrencies...",
    title="Blockchain Guide"
)

print(f"Category: {category.value}")
```

---

## API KEY SETUP

### Google Books API

1. **Create Project**:
   - Visit: https://console.cloud.google.com/
   - Create new project: "Xoe-NovAi"

2. **Enable API**:
   - Search for "Books API"
   - Click "Enable"

3. **Create API Key**:
   - Go to "Credentials" → "Create Credentials" → "API Key"
   - Copy the key

4. **Configure**:
   ```bash
   # Option 1: Environment variable
   export GOOGLE_BOOKS_API_KEY="AIzaSyD..."
   
   # Option 2: .env file
   echo "GOOGLE_BOOKS_API_KEY=AIzaSyD..." >> .env
   
   # Option 3: Docker
   # Add to docker-compose.yml:
   # environment:
   #   - GOOGLE_BOOKS_API_KEY=AIzaSyD...
   ```

5. **Test**:
   ```python
   from app.XNAi_rag_app.library_api_integrations import GoogleBooksClient, LibraryAPIConfig
   
   config = LibraryAPIConfig()
   client = GoogleBooksClient(config)
   results = client.search("Python Programming")
   print(f"Found {len(results)} results")
   ```

### ISBNdb API

1. **Sign Up**:
   - Visit: https://isbndb.com/api
   - Click "Sign Up" → Create account

2. **Get API Key**:
   - Login to dashboard
   - Copy API key

3. **Configure**:
   ```bash
   export ISBNDB_API_KEY="your_key..."
   # or
   echo "ISBNDB_API_KEY=your_key..." >> .env
   ```

4. **Test** (when integrated):
   ```python
   # Coming in Phase 1.5
   ```

### New York Public Library API

1. **Request Access**:
   - Visit: https://www.nypl.org/developers/apis
   - Fill out registration form

2. **Get API Key**:
   - Check email for API key

3. **Configure**:
   ```bash
   export NYPL_API_KEY="your_key..."
   ```

---

## TROUBLESHOOTING

### "API Key Not Configured"

**Problem**: Warning message about missing API key
**Solution**: This is normal! Free APIs work without keys. To enable optional APIs:

```bash
# Set API key
export GOOGLE_BOOKS_API_KEY="your_key"

# Or edit .env
GOOGLE_BOOKS_API_KEY=your_key
```

### "Connection Timeout"

**Problem**: API requests timing out
**Solution**:

```python
from app.XNAi_rag_app.library_api_integrations import LibraryAPIConfig

# Increase timeout
config = LibraryAPIConfig(request_timeout=20)  # 20 seconds
```

### "Rate Limit Exceeded"

**Problem**: Too many requests to API
**Solution**:

```python
from app.XNAi_rag_app.library_api_integrations import LibraryAPIConfig

# Adjust rate limiting
config = LibraryAPIConfig(
    rate_limit_calls=5,      # 5 calls
    rate_limit_period=60     # per 60 seconds
)
```

### "Domain Classification Low Confidence"

**Problem**: Category confidence below 0.5
**Solution**:

```python
# Provide more context
category, confidence = manager.classify(
    text="Very detailed content...",  # More text = better accuracy
    title="Clear, descriptive title",
    metadata=library_metadata        # Include library metadata if available
)

# Check confidence threshold
if confidence >= 0.6:
    use_classification()
else:
    ask_user_for_category()
```

### "Cache Not Working"

**Problem**: Cache disabled
**Solution**:

```python
from app.XNAi_rag_app.library_api_integrations import LibraryAPIConfig

config = LibraryAPIConfig(
    enable_cache=True,           # Enable cache
    cache_ttl=3600              # 1 hour
)
```

### "Dewey Decimal Not Applied"

**Problem**: Dewey classification missing
**Solution**:

```python
from app.XNAi_rag_app.library_api_integrations import LibraryAPIConfig

config = LibraryAPIConfig(
    enable_dewey_mapping=True    # Enable Dewey system
)
```

### "API Endpoint Returns 404"

**Problem**: API endpoint changed or deprecated
**Solution**: Use free APIs which are more stable:
- Open Library API (most stable)
- Internet Archive API (very stable)
- Project Gutenberg (stable)

Avoid:
- Library of Congress API (endpoints changed)
- Free Music Archive (may be unavailable)

---

## BEST PRACTICES

### 1. Always Use User-Agent Headers
```python
config = LibraryAPIConfig()
# User-Agent automatically set to:
# "Xoe-NovAi/0.1.4 (RAG System; +https://github.com/Xoe-NovAi/Xoe-NovAi)"
```

### 2. Enable Caching
```python
config = LibraryAPIConfig(enable_cache=True)
# Reduces API calls and improves performance
```

### 3. Handle Rate Limits Gracefully
```python
# Library integration handles automatically
# But adjust if needed
config = LibraryAPIConfig(
    rate_limit_calls=10,
    rate_limit_period=60
)
```

### 4. Respect API Rate Limits
- Don't make more than necessary calls
- Use caching to reduce requests
- Batch operations when possible

### 5. Provide Context for Classification
```python
# Good: Lots of context
category, conf = manager.classify(
    text="Full document content...",  # 1000+ chars
    title="Clear title",
    metadata=enriched_metadata
)

# Poor: Little context
category, conf = manager.classify(
    text="Hello",  # Too short
    title=""
)
```

---

## NEXT STEPS

1. **Local Testing**:
   ```bash
   python3 -c "from app.XNAi_rag_app.library_api_integrations import test_library_integration; test_library_integration()"
   ```

2. **Docker Testing**:
   ```bash
   docker-compose up -d
   ./test_docker_integration.sh
   ```

3. **Optional: Set API Keys**:
   - Google Books (free tier available)
   - ISBNdb (free tier available)

4. **Integration**:
   - Use `enrich_with_library_metadata()` in crawler
   - Classify documents automatically
   - Apply Dewey Decimal classifications

---

**Status**: ✅ Ready for Production Use

---

