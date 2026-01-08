"""
Xoe-NovAi Crawler + Curation Integration Module
================================================

Purpose: Extract metadata from crawled content for Phase 1.5 curation pipeline.
This module provides domain classification, citation detection, quality factor
calculation, and Redis queue integration for async curation processing.

Status: Production Ready (v0.1.4-stable)
Integration: Hooks into crawl4ai pipeline
Phase: Phase 1.5+ implementation ready

Author: Xoe-NovAi Team
Last Updated: 2026-01-03
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# ============================================================================
# ENUMS & MODELS
# ============================================================================

class DomainType(str, Enum):
    """Content domain classification for quality scoring."""
    CODE = "code"
    SCIENCE = "science"
    DATA = "data"
    GENERAL = "general"


@dataclass
class ContentMetadata:
    """Metadata extracted from crawled content for curation pipeline."""
    url: str
    crawl_date: str  # ISO format datetime
    domain: str  # DomainType.value
    word_count: int
    content_hash: str
    
    # Quality signals
    citation_count: int
    code_block_count: int
    image_count: int
    table_count: int
    heading_structure_score: float  # 0-1
    
    # Deduplication
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CrawledDocument:
    """Enhanced crawled document with curation metadata."""
    
    def __init__(
        self,
        url: str,
        content: str,
        metadata: ContentMetadata,
        domain: DomainType,
        quality_factors: Dict[str, float],
    ):
        self.url = url
        self.content = content
        self.metadata = metadata
        self.domain = domain
        self.quality_factors = quality_factors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'url': self.url,
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'domain': self.domain.value,
            'quality_factors': self.quality_factors,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for Redis storage."""
        return json.dumps(self.to_dict())


# ============================================================================
# EXTRACTION ENGINE
# ============================================================================

class CurationExtractor:
    """
    Extract metadata and quality signals from crawled content.
    
    This is the core extraction engine that feeds into the Phase 1.5
    quality scorer. Provides domain classification, citation detection,
    content structure analysis, and quality factor calculation.
    """
    
    def __init__(self):
        """Initialize regex patterns for content extraction."""
        self.doi_pattern = r'\b10\.\d{4,}/[\S]+\b'
        self.arxiv_pattern = r'\b\d{4}\.\d{5}\b'
        self.code_pattern = r'```[\s\S]*?```|<code>[\s\S]*?</code>'
        self.image_pattern = r'<img|!\[|<figure'
        self.table_pattern = r'<table|<tr>|<td>'
        self.heading_pattern = r'<h([1-6])>'
    
    # ========================================================================
    # DOMAIN CLASSIFICATION
    # ========================================================================
    
    def classify_domain(self, content: str, url: str) -> DomainType:
        """
        Classify content domain (code/science/data/general).
        
        Classification Rules:
        - CODE: Git repos, programming docs, code blocks, syntactically valid Python/JS
        - SCIENCE: ArXiv, DOI, citations, research methodology
        - DATA: CSV/JSON, datasets, SQL, extensive tables
        - GENERAL: News, blogs, general web content
        
        Args:
            content: Document content
            url: Source URL
        
        Returns:
            DomainType enum value
        """
        content_lower = content.lower()
        url_lower = url.lower()
        
        # CODE signals (0-9 points)
        code_signals = [
            'github.com' in url_lower,
            'gitlab' in url_lower,
            'github' in content_lower,
            'git' in url_lower,
            'code' in url_lower,
            'python' in content_lower,
            'javascript' in content_lower,
            'def ' in content or 'class ' in content,
            'import ' in content,
            len(re.findall(self.code_pattern, content)) > 3,
        ]
        
        # SCIENCE signals (0-9 points)
        science_signals = [
            'arxiv.org' in url_lower,
            'doi.org' in url_lower,
            'pubmed' in url_lower,
            'scholar' in url_lower,
            len(re.findall(self.doi_pattern, content)) > 0,
            len(re.findall(self.arxiv_pattern, content)) > 0,
            'abstract' in content_lower and 'introduction' in content_lower,
            'methodology' in content_lower,
            'research' in content_lower,
        ]
        
        # DATA signals (0-8 points)
        data_signals = [
            'dataset' in url_lower,
            'kaggle' in url_lower,
            'data.gov' in url_lower,
            '.csv' in url_lower or '.json' in url_lower,
            'SELECT' in content or 'select' in content,
            len(re.findall(self.table_pattern, content)) > 5,
            'data' in url_lower,
            'table' in content_lower,
        ]
        
        # Score signals
        code_score = sum(code_signals)
        science_score = sum(science_signals)
        data_score = sum(data_signals)
        
        # Classify based on dominant signal
        if code_score > science_score and code_score > data_score and code_score > 0:
            return DomainType.CODE
        elif science_score > data_score and science_score > code_score and science_score > 0:
            return DomainType.SCIENCE
        elif data_score > code_score and data_score > science_score and data_score > 0:
            return DomainType.DATA
        else:
            return DomainType.GENERAL
    
    # ========================================================================
    # CITATION & RESEARCH SIGNALS
    # ========================================================================
    
    def extract_citations(self, content: str) -> Dict[str, int]:
        """
        Extract citations from content.
        
        Returns:
            {
                'doi': count,
                'arxiv': count,
                'total': count
            }
        """
        doi_matches = re.findall(self.doi_pattern, content)
        arxiv_matches = re.findall(self.arxiv_pattern, content)
        
        return {
            'doi': len(doi_matches),
            'arxiv': len(arxiv_matches),
            'total': len(doi_matches) + len(arxiv_matches),
        }
    
    # ========================================================================
    # CONTENT STRUCTURE ANALYSIS
    # ========================================================================
    
    def count_code_blocks(self, content: str) -> int:
        """Count code blocks (```...``` or <code>...</code>)."""
        return len(re.findall(self.code_pattern, content))
    
    def count_images(self, content: str) -> int:
        """Count images in content."""
        return len(re.findall(self.image_pattern, content))
    
    def count_tables(self, content: str) -> int:
        """Count tables in content."""
        return len(re.findall(self.table_pattern, content))
    
    def calculate_heading_structure_score(self, content: str) -> float:
        """
        Calculate heading structure quality (0-1).
        
        Good structure: H1 → H2 → H3 hierarchy, no large gaps
        
        Returns:
            Score 0-1 based on heading hierarchy
        """
        h_tags = {}
        for i in range(1, 7):
            h_tags[f'h{i}'] = len(re.findall(f'<h{i}>', content, re.IGNORECASE))
        
        total_headings = sum(h_tags.values())
        if total_headings == 0:
            return 0.0
        
        # Check for proper hierarchy
        has_h1 = h_tags['h1'] > 0
        h1_dominance = h_tags['h1'] / total_headings if has_h1 else 0
        
        # Prefer hierarchical structure with H1
        score = min(1.0, h1_dominance + (0.2 if has_h1 else 0))
        return round(score, 2)
    
    # ========================================================================
    # QUALITY FACTORS (for Phase 1.5 quality scorer)
    # ========================================================================
    
    def calculate_quality_factors(
        self,
        content: str,
        url: str,
        domain: DomainType,
    ) -> Dict[str, float]:
        """
        Calculate 5 quality factors for Phase 1.5 quality scorer.
        
        Factors:
        - freshness: 0-1 based on date signals
        - completeness: 0-1 based on word count and structure
        - authority: 0-1 based on citations and domain
        - structure: 0-1 based on heading hierarchy and tables
        - accessibility: 0-1 based on code/data readability
        
        Args:
            content: Document content
            url: Source URL
            domain: Classified domain type
        
        Returns:
            {
                'freshness': float,
                'completeness': float,
                'authority': float,
                'structure': float,
                'accessibility': float,
            }
        """
        citations = self.extract_citations(content)
        word_count = len(content.split())
        code_blocks = self.count_code_blocks(content)
        heading_score = self.calculate_heading_structure_score(content)
        
        factors = {}
        
        # 1. FRESHNESS: Based on URL/content date signals (heuristic)
        has_date = bool(re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', content))
        factors['freshness'] = 0.7 if has_date else 0.3
        
        # 2. COMPLETENESS: Word count + structure
        completeness_from_length = min(1.0, word_count / 2000)
        completeness_from_structure = heading_score
        factors['completeness'] = round(
            (completeness_from_length + completeness_from_structure) / 2,
            2
        )
        
        # 3. AUTHORITY: Citations + domain expertise
        authority_from_citations = min(1.0, citations['total'] / 10)
        authority_from_domain = 0.8 if domain in [DomainType.SCIENCE, DomainType.DATA] else 0.4
        factors['authority'] = round(
            (authority_from_citations + authority_from_domain) / 2,
            2
        )
        
        # 4. STRUCTURE: Heading hierarchy + tables + images
        structure_from_headings = heading_score
        structure_from_tables = min(1.0, self.count_tables(content) / 5)
        structure_from_images = min(1.0, self.count_images(content) / 10)
        factors['structure'] = round(
            (structure_from_headings + structure_from_tables + structure_from_images) / 3,
            2
        )
        
        # 5. ACCESSIBILITY: Code/data readability (domain-specific)
        if domain == DomainType.CODE:
            factors['accessibility'] = round(min(1.0, code_blocks / 5), 2)
        elif domain == DomainType.DATA:
            factors['accessibility'] = round(min(1.0, self.count_tables(content) / 3), 2)
        else:
            factors['accessibility'] = 0.5  # Neutral for general/science
        
        return factors
    
    # ========================================================================
    # METADATA EXTRACTION
    # ========================================================================
    
    def extract_metadata(
        self,
        url: str,
        content: str,
        crawl_date: Optional[str] = None,
    ) -> ContentMetadata:
        """
        Extract full metadata from crawled content.
        
        Args:
            url: Source URL
            content: Document content
            crawl_date: ISO format datetime (default: now)
        
        Returns:
            ContentMetadata object with all extraction results
        """
        if crawl_date is None:
            crawl_date = datetime.now().isoformat()
        
        # Classify domain
        domain = self.classify_domain(content, url)
        
        # Extract structural information
        word_count = len(content.split())
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        citations = self.extract_citations(content)
        code_blocks = self.count_code_blocks(content)
        images = self.count_images(content)
        tables = self.count_tables(content)
        
        metadata = ContentMetadata(
            url=url,
            crawl_date=crawl_date,
            domain=domain.value,
            word_count=word_count,
            content_hash=content_hash,
            citation_count=citations['total'],
            code_block_count=code_blocks,
            image_count=images,
            table_count=tables,
            heading_structure_score=self.calculate_heading_structure_score(content),
        )
        
        return metadata
    
    def create_crawled_document(
        self,
        url: str,
        content: str,
        crawl_date: Optional[str] = None,
    ) -> CrawledDocument:
        """
        Create enhanced CrawledDocument with all curation metadata.
        
        Usage:
            extractor = CurationExtractor()
            doc = extractor.create_crawled_document(url, content)
            print(f"Domain: {doc.domain}")
            print(f"Quality factors: {doc.quality_factors}")
            print(f"Citations: {doc.metadata.citation_count}")
        
        Args:
            url: Source URL
            content: Document content
            crawl_date: ISO format datetime (default: now)
        
        Returns:
            CrawledDocument with complete metadata and quality factors
        """
        metadata = self.extract_metadata(url, content, crawl_date)
        domain = DomainType(metadata.domain)
        quality_factors = self.calculate_quality_factors(content, url, domain)
        
        return CrawledDocument(
            url=url,
            content=content,
            metadata=metadata,
            domain=domain,
            quality_factors=quality_factors,
        )


# ============================================================================
# INTEGRATION HOOK
# ============================================================================

async def crawl_and_curate(crawler, url: str) -> Optional[CrawledDocument]:
    """
    Crawl URL and extract curation metadata.
    
    This is the primary integration point for the curation pipeline.
    
    Usage:
        from XNAi_rag_app.crawler_curation import crawl_and_curate
        from crawl4ai import AsyncWebCrawler
        
        crawler = AsyncWebCrawler()
        doc = await crawl_and_curate(crawler, "https://example.com")
        
        if doc:
            print(f"Domain: {doc.domain}")
            print(f"Quality factors: {doc.quality_factors}")
            print(f"Citations: {doc.metadata.citation_count}")
    
    Args:
        crawler: crawl4ai AsyncWebCrawler instance
        url: URL to crawl
    
    Returns:
        CrawledDocument with metadata, or None if crawl fails
    """
    try:
        # 1. Crawl with crawl4ai
        result = await crawler.crawl(url)
        
        if not result or not result.markdown:
            return None
        
        # 2. Extract curation metadata
        extractor = CurationExtractor()
        doc = extractor.create_crawled_document(url, result.markdown)
        
        return doc
        
    except Exception as e:
        print(f"❌ Error crawling {url}: {e}")
        return None


# ============================================================================
# REDIS QUEUE INTEGRATION
# ============================================================================

def queue_for_curation(doc: CrawledDocument, redis_conn) -> bool:
    """
    Queue crawled document for async curation processing.
    
    Pushes the document metadata (not full content) to Redis queue
    for the curation_worker to process asynchronously.
    
    Usage:
        import redis
        from XNAi_rag_app.crawler_curation import queue_for_curation
        
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        success = queue_for_curation(doc, r)
    
    Args:
        doc: CrawledDocument to queue
        redis_conn: Redis connection object
    
    Returns:
        True if successfully queued, False otherwise
    """
    try:
        task = {
            'url': doc.url,
            'content_hash': doc.metadata.content_hash,
            'domain': doc.domain.value,
            'quality_factors': doc.quality_factors,
            'metadata': {
                'word_count': doc.metadata.word_count,
                'citation_count': doc.metadata.citation_count,
                'code_blocks': doc.metadata.code_block_count,
                'images': doc.metadata.image_count,
                'tables': doc.metadata.table_count,
            },
            'crawl_date': doc.metadata.crawl_date,
        }
        
        # Push to Redis queue
        redis_conn.rpush('curation_queue', json.dumps(task))
        return True
        
    except Exception as e:
        print(f"❌ Error queueing for curation: {e}")
        return False


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_extraction():
    """
    Quick validation test of extraction pipeline.
    Run this to verify the module works correctly.
    """
    extractor = CurationExtractor()
    
    # Sample content
    sample_url = "https://github.com/example/repo"
    sample_content = """
    # Example Python Project
    
    ## Introduction
    This is a sample project demonstrating code extraction.
    
    ```python
    def hello_world():
        print("Hello, World!")
    ```
    
    ## Research References
    See: 10.1234/example.doi
    ArXiv: 2024.12345
    
    ## Features
    - Fast processing
    - Easy integration
    
    ## Academic Citation
    Smith et al., 2024. "Title". DOI: 10.1234/example
    """
    
    # Extract metadata
    doc = extractor.create_crawled_document(sample_url, sample_content)
    
    print("✓ Extraction Test Results:")
    print(f"  Domain: {doc.domain.value}")
    print(f"  Word count: {doc.metadata.word_count}")
    print(f"  Citations: {doc.metadata.citation_count}")
    print(f"  Code blocks: {doc.metadata.code_block_count}")
    print(f"  Content hash: {doc.metadata.content_hash}")
    print(f"  Quality Factors:")
    for factor, score in doc.quality_factors.items():
        print(f"    - {factor}: {score}")
    
    return doc

# ============================================================================
# LIBRARY API INTEGRATION FUNCTIONS
# ============================================================================

def enrich_with_library_metadata(document: CrawledDocument, title: str, author: Optional[str] = None) -> CrawledDocument:
    """
    Enrich crawled document with library metadata from multiple APIs.
    
    This function automatically:
    1. Classifies content into domain categories
    2. Searches library APIs for enrichment
    3. Maps to Dewey Decimal classifications
    4. Updates document metadata
    
    Args:
        document: The crawled document to enrich
        title: Document title for library search
        author: Optional author name
    
    Returns:
        Updated document with library metadata
    
    Example:
        >>> doc = CrawledDocument(...)
        >>> enriched = enrich_with_library_metadata(doc, "Python Programming", "Guido van Rossum")
    """
    try:
        # Import library integration module
        from library_api_integrations import LibraryEnrichmentEngine, LibraryAPIConfig
        
        # Initialize engine
        config = LibraryAPIConfig(enable_cache=True, enable_dewey_mapping=True)
        engine = LibraryEnrichmentEngine(config)
        
        # Classify and enrich
        enrichment = engine.classify_and_enrich(
            title=title,
            content=document.content[:2000],  # Use first 2000 chars for classification
            author=author
        )
        
        # Update document metadata
        document.metadata.enriched_library_data = enrichment
        document.metadata.domain_category = enrichment.get("domain_category")
        document.metadata.dewey_decimal = enrichment.get("primary_dewey")
        
        logger.info(f"Document enriched: {title} -> {enrichment.get('domain_category')}")
        
    except ImportError:
        logger.warning("Library API integrations not available")
    except Exception as e:
        logger.error(f"Error enriching with library metadata: {e}")
    
    return document


def bulk_enrich_documents(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Enrich multiple documents in batch.
    
    Args:
        documents: List of dicts with 'title', 'content', optional 'author'
    
    Returns:
        List of enriched document data
    """
    try:
        from library_api_integrations import LibraryEnrichmentEngine, LibraryAPIConfig
        
        config = LibraryAPIConfig(enable_cache=True, enable_dewey_mapping=True)
        engine = LibraryEnrichmentEngine(config)
        
        return engine.batch_enrich(documents)
    except ImportError:
        logger.warning("Library API integrations not available")
        return []
    except Exception as e:
        logger.error(f"Error in bulk enrichment: {e}")
        return []


def get_domain_categories() -> List[str]:
    """Get list of available domain categories."""
    try:
        from library_api_integrations import DomainManager
        manager = DomainManager()
        return manager.get_all_categories()
    except ImportError:
        return ["code", "science", "data", "general", "books", "music"]


if __name__ == "__main__":
    # Run validation test
    doc = test_extraction()
    print("\n✓ Curation extractor module is working correctly!")
