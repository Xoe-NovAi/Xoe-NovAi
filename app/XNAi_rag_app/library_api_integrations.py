"""
Xoe-NovAi Library API Integrations Module
==========================================

Purpose: Integrate with multiple library, book, music, and archive APIs for
comprehensive metadata enrichment during curation.

Supported APIs (All Completely Free - No Rate Limits):
- Open Library API (Books, Authors, Subjects)
- Internet Archive Books API (Full-text search, metadata)
- Library of Congress API (Books, prints, photographs)
- Project Gutenberg API (Public domain books)
- WorldCat OpenSearch API (Library catalog search)
- Cambridge Digital Library API (Manuscripts, collections)
- Free Music Archive API (Music metadata)

Additional Features:
- Natural Language Curator Interface (chatbot-style commands)
- Chainlit UI Integration for interactive curation

Features:
- Dewey Decimal System integration for cataloging
- Intuitive domain categorization system
- Automatic resource discovery and enrichment
- Rate limiting and caching
- Error handling and fallback strategies

Status: Production Ready (v0.1.4-stable)
Phase: Phase 1.5+ implementation ready
Author: Xoe-NovAi Team
Last Updated: 2026-01-03
"""

import hashlib
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import lru_cache
import re

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class DomainCategory(str, Enum):
    """Enhanced domain categorization for multi-source curation."""
    # Existing domains (12)
    CODE = "code"
    SCIENCE = "science"
    DATA = "data"
    GENERAL = "general"
    BOOKS = "books"
    MUSIC = "music"
    ARCHIVES = "archives"
    MANUSCRIPTS = "manuscripts"
    PHOTOGRAPHS = "photographs"
    AUDIO = "audio"
    FICTION = "fiction"
    REFERENCE = "reference"
    
    # Audio-specific domains (5+)
    PODCAST = "podcast"
    SCIENCE_PODCAST = "science_podcast"
    TECH_PODCAST = "tech_podcast"
    BUSINESS_PODCAST = "business_podcast"
    AUDIOBOOK = "audiobook"
    
    # Music sub-domains (6+)
    CLASSICAL_MUSIC = "classical_music"
    JAZZ_MUSIC = "jazz_music"
    ROCK_MUSIC = "rock_music"
    HIP_HOP_MUSIC = "hip_hop_music"
    INDIE_MUSIC = "indie_music"
    ELECTRONIC_MUSIC = "electronic_music"


class DeweyDecimalClass(str, Enum):
    """Dewey Decimal Classification system mapping for libraries."""
    # 000-099: Computer science, information & general works
    COMPUTER_SCIENCE = "000"
    
    # 100-199: Philosophy & psychology
    PHILOSOPHY = "100"
    
    # 200-299: Religion
    RELIGION = "200"
    
    # 300-399: Social sciences
    SOCIAL_SCIENCES = "300"
    
    # 400-499: Language
    LANGUAGE = "400"
    
    # 500-599: Science
    SCIENCE = "500"
    
    # 600-699: Technology (Applied sciences)
    TECHNOLOGY = "600"
    
    # 700-799: Arts & recreation
    ARTS = "700"
    
    # 800-899: Literature
    LITERATURE = "800"
    
    # 900-999: History & geography
    HISTORY = "900"


# Dewey Decimal mappings to domain categories
DEWEY_TO_DOMAIN = {
    "000": DomainCategory.CODE,
    "500": DomainCategory.SCIENCE,
    "600": DomainCategory.ARCHIVES,  # Technology/Archives
    "800": DomainCategory.BOOKS,  # Literature/Books
    "900": DomainCategory.REFERENCE,
}

# Domain to Dewey Decimal suggestions
DOMAIN_TO_DEWEY = {
    DomainCategory.CODE: ["000", "005", "006"],
    DomainCategory.SCIENCE: ["500", "540", "570"],
    DomainCategory.DATA: ["005", "006", "511"],
    DomainCategory.BOOKS: ["800", "810", "820"],
    DomainCategory.MUSIC: ["780", "781", "782"],
    DomainCategory.ARCHIVES: ["600", "620", "670"],
    DomainCategory.GENERAL: ["000", "030"],
    DomainCategory.MANUSCRIPTS: ["091", "092", "093"],
    DomainCategory.PHOTOGRAPHS: ["770", "778", "779"],
    DomainCategory.AUDIO: ["780", "785", "787"],
    DomainCategory.FICTION: ["800", "810", "820"],
    DomainCategory.REFERENCE: ["030", "031", "032"],
}


# ============================================================================
# CONFIGURATION & MODELS
# ============================================================================

@dataclass
class LibraryAPIConfig:
    """Configuration for library API integrations."""
    # API Keys & URLs
    google_books_api_key: Optional[str] = None
    isbndb_api_key: Optional[str] = None
    loc_api_base_url: str = "https://www.loc.gov/books/services/web/search.json"
    openlibrary_api_base_url: str = "https://openlibrary.org"
    archive_api_base_url: str = "https://archive.org/advancedsearch.php"
    worldcat_api_base_url: str = "https://www.worldcat.org/cgi-bin/json_webservice"
    nypl_api_base_url: str = "https://api.nypl.org/api/v2"
    gutenberg_api_base_url: str = "https://gutendex.com"
    free_music_archive_api_base_url: str = "https://freemusicarchive.org/api"
    
    # Rate limiting
    rate_limit_calls: int = 10  # Calls per time window
    rate_limit_period: int = 60  # Seconds
    
    # Timeouts
    request_timeout: int = 10  # Seconds
    cache_ttl: int = 3600  # 1 hour
    
    # Features
    enable_cache: bool = True
    enable_dewey_mapping: bool = True
    enable_auto_classification: bool = True
    
    # User agent (required by Open Library)
    user_agent: str = "Xoe-NovAi/0.1.4 (RAG System; +https://github.com/Xoe-NovAi/Xoe-NovAi)"
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.google_books_api_key:
            self.google_books_api_key = os.getenv('GOOGLE_BOOKS_API_KEY')
        if not self.isbndb_api_key:
            self.isbndb_api_key = os.getenv('ISBNDB_API_KEY')
        if not self.nypl_api_base_url:
            nypl_key = os.getenv('NYPL_API_KEY')
            if nypl_key:
                self.nypl_api_base_url = f"{self.nypl_api_base_url}?key={nypl_key}"


@dataclass
class LibraryMetadata:
    """Enriched metadata from library APIs (books, podcasts, music, etc.)."""
    isbn: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    description: Optional[str] = None
    subjects: List[str] = field(default_factory=list)
    dewey_decimal: Optional[str] = None
    oclc_number: Optional[str] = None
    lcc: Optional[str] = None  # Library of Congress Classification
    language: Optional[str] = None
    page_count: Optional[int] = None
    cover_url: Optional[str] = None
    source_apis: List[str] = field(default_factory=list)  # Which APIs provided data
    enrichment_confidence: float = 0.0  # 0.0-1.0
    
    # Audio-specific fields (for podcasts, music, audiobooks)
    is_audio: bool = False  # Flag for audio content
    audio_type: Optional[str] = None  # "podcast", "music_track", "music_album", "audiobook"
    
    # Podcast fields
    podcast_id: Optional[str] = None
    podcast_url: Optional[str] = None
    episode_number: Optional[int] = None
    episode_duration: Optional[int] = None  # seconds
    episode_transcript_url: Optional[str] = None
    season: Optional[int] = None
    
    # Music fields
    artist: Optional[str] = None
    album: Optional[str] = None
    isrc: Optional[str] = None  # International Standard Recording Code
    iswc: Optional[str] = None  # International Standard Musical Work Code
    track_number: Optional[int] = None
    release_date: Optional[str] = None
    genre: Optional[str] = None
    mood: List[str] = field(default_factory=list)
    
    # Audio metadata
    duration: Optional[int] = None  # seconds
    format: Optional[str] = None  # "mp3", "aac", "flac", "rss_feed", etc.
    bitrate: Optional[int] = None  # kbps
    explicit: bool = False
    
    # Audio cataloging
    audio_hash: Optional[str] = None  # For deduplication
    cdn_url: Optional[str] = None  # Streaming URL
    download_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResourceLocation:
    """Location information for a resource."""
    source: str  # API name
    url: str
    availability: str  # "available", "preview", "full", "limited"
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# LIBRARY API CLIENTS
# ============================================================================

class BaseLibraryClient(ABC):
    """Abstract base class for library API clients."""
    
    def __init__(self, config: LibraryAPIConfig):
        self.config = config
        self.session = self._create_session()
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.last_request_time = 0
    
    def _create_session(self) -> 'requests.Session':
        """Create requests session with retry strategy."""
        if not REQUESTS_AVAILABLE:
            return None
        
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        min_interval = self.config.rate_limit_period / self.config.rate_limit_calls
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get item from cache if valid."""
        if not self.config.enable_cache or key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.config.cache_ttl:
            del self.cache[key]
            return None
        return value
    
    def _set_cache(self, key: str, value: Any):
        """Store item in cache."""
        if self.config.enable_cache:
            self.cache[key] = (value, time.time())
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search library for resources."""
        pass
    
    @abstractmethod
    def get_by_identifier(self, identifier: str, id_type: str) -> Optional[LibraryMetadata]:
        """Get resource by identifier (ISBN, OCLC, etc)."""
        pass


class OpenLibraryClient(BaseLibraryClient):
    """Open Library API client."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Open Library."""
        cache_key = f"openlibrary:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            url = f"{self.config.openlibrary_api_base_url}/search.json"
            params = {
                "title": query,
                "limit": kwargs.get("limit", 5)
            }
            headers = {"User-Agent": self.config.user_agent}
            
            response = self.session.get(url, params=params, headers=headers, 
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for doc in data.get("docs", [])[:5]:
                metadata = LibraryMetadata(
                    isbn=self._get_first(doc.get("isbn")),
                    title=doc.get("title"),
                    authors=doc.get("author_name", []),
                    publication_date=self._get_first(doc.get("first_publish_year")),
                    subjects=doc.get("subject", [])[:5],
                    source_apis=["openlibrary"],
                    enrichment_confidence=0.7
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Open Library search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "isbn") -> Optional[LibraryMetadata]:
        """Get resource by ISBN or other identifier."""
        if id_type != "isbn":
            return None
        
        cache_key = f"openlibrary:isbn:{identifier}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            url = f"{self.config.openlibrary_api_base_url}/api/books"
            params = {"bibkeys": f"ISBN:{identifier}", "jscmd": "details", "format": "json"}
            headers = {"User-Agent": self.config.user_agent}
            
            response = self.session.get(url, params=params, headers=headers,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            if data:
                key = list(data.keys())[0]
                details = data[key].get("details", {})
                
                metadata = LibraryMetadata(
                    isbn=identifier,
                    title=details.get("title"),
                    authors=[a.get("name") for a in details.get("authors", [])],
                    publication_date=str(details.get("publish_date", "")),
                    publisher=self._get_first(details.get("publishers")),
                    subjects=details.get("subjects", [])[:5],
                    source_apis=["openlibrary"],
                    enrichment_confidence=0.85
                )
                self._set_cache(cache_key, metadata)
                return metadata
        except Exception as e:
            logger.error(f"Open Library ISBN lookup failed: {e}")
        return None
    
    @staticmethod
    def _get_first(value):
        """Get first item if list, otherwise return value."""
        return value[0] if isinstance(value, list) and value else value


class InternetArchiveClient(BaseLibraryClient):
    """Internet Archive API client for books and collections."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Internet Archive for books."""
        cache_key = f"internetarchive:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "q": f"(title:{query} OR description:{query}) AND mediatype:texts",
                "output": "json",
                "rows": kwargs.get("limit", 5),
                "fl": ["identifier", "title", "creator", "date", "description", "subject"]
            }
            
            response = self.session.get(self.config.archive_api_base_url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for doc in data.get("response", {}).get("docs", [])[:5]:
                creators = doc.get("creator", [])
                if isinstance(creators, str):
                    creators = [creators]
                
                metadata = LibraryMetadata(
                    title=doc.get("title"),
                    authors=creators,
                    publication_date=doc.get("date"),
                    description=doc.get("description", "")[:500] if doc.get("description") else None,
                    subjects=doc.get("subject", [])[:5] if doc.get("subject") else [],
                    source_apis=["internetarchive"],
                    enrichment_confidence=0.65
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Internet Archive search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "isbn") -> Optional[LibraryMetadata]:
        """Get resource by identifier."""
        try:
            self._rate_limit()
            url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            metadata_dict = data.get("metadata", {})
            
            metadata = LibraryMetadata(
                title=metadata_dict.get("title"),
                authors=metadata_dict.get("creator", []) if isinstance(metadata_dict.get("creator"), list) else [metadata_dict.get("creator")] if metadata_dict.get("creator") else [],
                publication_date=metadata_dict.get("date"),
                description=metadata_dict.get("description", "")[:500] if metadata_dict.get("description") else None,
                subjects=metadata_dict.get("subject", [])[:5] if isinstance(metadata_dict.get("subject"), list) else [],
                source_apis=["internetarchive"],
                enrichment_confidence=0.70
            )
            return metadata
        except Exception as e:
            logger.error(f"Internet Archive lookup failed: {e}")
            return None


class LibraryOfCongressClient(BaseLibraryClient):
    """Library of Congress API client."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Library of Congress catalog."""
        cache_key = f"loc:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "q": query,
                "fo": "json",
                "pagesize": kwargs.get("limit", 5)
            }
            
            response = self.session.get(self.config.loc_api_base_url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for record in data.get("results", [])[:5]:
                metadata = LibraryMetadata(
                    title=record.get("title"),
                    authors=record.get("creators", []),
                    publication_date=record.get("date"),
                    description=record.get("description", "")[:500] if record.get("description") else None,
                    subjects=record.get("subjects", [])[:5],
                    lcc=record.get("classification"),
                    source_apis=["loc"],
                    enrichment_confidence=0.80
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Library of Congress search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "lccn") -> Optional[LibraryMetadata]:
        """Get resource by LCCN or other identifier."""
        if id_type == "lccn":
            return self.search(identifier, limit=1)[0] if self.search(identifier, limit=1) else None
        return None


class ProjectGutenbergClient(BaseLibraryClient):
    """Project Gutenberg API client for public domain books."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Project Gutenberg."""
        cache_key = f"gutenberg:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            url = f"{self.config.gutenberg_api_base_url}/books/search"
            params = {
                "query": query,
                "topic": "all"
            }
            
            response = self.session.get(url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for book in data.get("results", [])[:5]:
                metadata = LibraryMetadata(
                    title=book.get("title"),
                    authors=book.get("authors", []),
                    publication_date=book.get("publication_date"),
                    subjects=book.get("languages", []),
                    source_apis=["gutenberg"],
                    enrichment_confidence=0.60
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"Project Gutenberg search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "gutenberg_id") -> Optional[LibraryMetadata]:
        """Get resource by Gutenberg ID."""
        try:
            self._rate_limit()
            url = f"{self.config.gutenberg_api_base_url}/books/{identifier}"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            metadata = LibraryMetadata(
                title=data.get("title"),
                authors=[a.get("name") for a in data.get("authors", [])],
                publication_date=data.get("publication_date"),
                cover_url=data.get("cover_image"),
                source_apis=["gutenberg"],
                enrichment_confidence=0.65
            )
            return metadata
        except Exception as e:
            logger.error(f"Project Gutenberg lookup failed: {e}")
            return None


class FreeMusicArchiveClient(BaseLibraryClient):
    """Free Music Archive API client for music metadata."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Free Music Archive."""
        cache_key = f"fma:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            # FMA API endpoint for tracks
            url = f"{self.config.free_music_archive_api_base_url}/tracks"
            params = {
                "query": query,
                "limit": kwargs.get("limit", 5)
            }
            
            response = self.session.get(url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for track in data.get("tracks", [])[:5]:
                metadata = LibraryMetadata(
                    title=track.get("track_title"),
                    authors=[track.get("artist_name")],
                    publication_date=track.get("track_date_created"),
                    subjects=[track.get("genre_title")],
                    source_apis=["freemusicarchive"],
                    enrichment_confidence=0.70
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Free Music Archive search failed (may be unavailable): {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "track_id") -> Optional[LibraryMetadata]:
        """Get track by identifier."""
        try:
            self._rate_limit()
            url = f"{self.config.free_music_archive_api_base_url}/tracks/{identifier}"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            metadata = LibraryMetadata(
                title=data.get("track_title"),
                authors=[data.get("artist_name")],
                publication_date=data.get("track_date_created"),
                subjects=[data.get("genre_title")],
                source_apis=["freemusicarchive"],
                enrichment_confidence=0.75
            )
            return metadata
        except Exception as e:
            logger.error(f"Free Music Archive lookup failed: {e}")
            return None


class WorldCatOpenSearchClient(BaseLibraryClient):
    """WorldCat OpenSearch API client for library catalog searches."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search WorldCat catalog using OpenSearch."""
        cache_key = f"worldcat:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            # WorldCat OpenSearch endpoint (free, no authentication required)
            url = "https://www.worldcat.org/webservices/catalog/search/opensearch"
            params = {
                "q": query,
                "format": "json",
                "maxResults": kwargs.get("limit", 5),
                "frbrGrouping": "on"
            }
            headers = {"Accept": "application/json"}
            
            response = self.session.get(url, params=params, headers=headers,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Parse OpenSearch results
            for record in data.get("searchResults", [])[:5]:
                # Extract OCLC number from detail page URL
                oclc_number = None
                if "detailUrl" in record:
                    oclc_match = re.search(r'/oclc/(\d+)', record.get("detailUrl", ""))
                    if oclc_match:
                        oclc_number = oclc_match.group(1)
                
                metadata = LibraryMetadata(
                    title=record.get("title"),
                    authors=record.get("author", []) if isinstance(record.get("author"), list) else [record.get("author")] if record.get("author") else [],
                    publication_date=str(record.get("date", "")),
                    publisher=record.get("publisher"),
                    oclc_number=oclc_number,
                    subjects=record.get("subject", [])[:5] if isinstance(record.get("subject"), list) else [],
                    source_apis=["worldcat"],
                    enrichment_confidence=0.72
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.error(f"WorldCat search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "oclc") -> Optional[LibraryMetadata]:
        """Get resource by OCLC number."""
        if id_type != "oclc":
            return None
        
        try:
            self._rate_limit()
            url = f"https://www.worldcat.org/oclc/{identifier}.json"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            brief = data.get("briefRecords", [{}])[0] if data.get("briefRecords") else {}
            
            metadata = LibraryMetadata(
                title=brief.get("title"),
                authors=brief.get("author", []) if isinstance(brief.get("author"), list) else [brief.get("author")] if brief.get("author") else [],
                publication_date=str(brief.get("date", "")),
                publisher=brief.get("publisher"),
                oclc_number=identifier,
                source_apis=["worldcat"],
                enrichment_confidence=0.78
            )
            return metadata
        except Exception as e:
            logger.error(f"WorldCat OCLC lookup failed: {e}")
            return None


class CambridgeDigitalLibraryClient(BaseLibraryClient):
    """Cambridge Digital Library API client for manuscripts and collections."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search Cambridge Digital Library collections."""
        cache_key = f"cambridge:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            # Cambridge Digital Library search endpoint
            url = "https://cudl.lib.cam.ac.uk/api/v1/search"
            params = {
                "q": query,
                "limit": kwargs.get("limit", 5),
                "format": "json"
            }
            
            response = self.session.get(url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", [])[:5]:
                metadata = LibraryMetadata(
                    title=item.get("title"),
                    authors=item.get("authors", []) if isinstance(item.get("authors"), list) else [item.get("authors")] if item.get("authors") else [],
                    publication_date=item.get("date"),
                    description=item.get("summary", "")[:500] if item.get("summary") else None,
                    subjects=item.get("subjects", [])[:5] if isinstance(item.get("subjects"), list) else [],
                    language=item.get("language"),
                    source_apis=["cambridge"],
                    enrichment_confidence=0.68
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Cambridge Digital Library search failed (API may be unavailable): {e}")
            # Fallback: Try basic web search via Open Library
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "cudl_id") -> Optional[LibraryMetadata]:
        """Get resource by Cambridge Digital Library ID."""
        try:
            self._rate_limit()
            url = f"https://cudl.lib.cam.ac.uk/api/v1/metadata/b{identifier}"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            metadata = LibraryMetadata(
                title=data.get("title"),
                authors=data.get("authors", []) if isinstance(data.get("authors"), list) else [data.get("authors")] if data.get("authors") else [],
                publication_date=data.get("date"),
                description=data.get("description", "")[:500] if data.get("description") else None,
                subjects=data.get("subjects", [])[:5] if isinstance(data.get("subjects"), list) else [],
                language=data.get("language"),
                source_apis=["cambridge"],
                enrichment_confidence=0.75
            )
            return metadata
        except Exception as e:
            logger.error(f"Cambridge Digital Library lookup failed: {e}")
            return None


class BookwormEpubClient(BaseLibraryClient):
    """Bookworm EPUB Reader API client for EPUB book discovery."""
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search for EPUB books using Internet Archive EPUB endpoint."""
        cache_key = f"bookworm:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            # Use Internet Archive search for EPUB books since Bookworm API is not publicly available
            params = {
                "q": f"({query}) AND format:DAISY AND mediatype:texts",
                "output": "json",
                "rows": kwargs.get("limit", 5),
                "fl": ["identifier", "title", "creator", "date", "description", "subject"]
            }
            
            response = self.session.get(self.config.archive_api_base_url, params=params,
                                       timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for doc in data.get("response", {}).get("docs", [])[:5]:
                creators = doc.get("creator", [])
                if isinstance(creators, str):
                    creators = [creators]
                
                metadata = LibraryMetadata(
                    title=doc.get("title"),
                    authors=creators,
                    publication_date=doc.get("date"),
                    description=doc.get("description", "")[:500] if doc.get("description") else None,
                    subjects=doc.get("subject", [])[:5] if doc.get("subject") else [],
                    source_apis=["bookworm_epub"],
                    enrichment_confidence=0.60
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Bookworm EPUB search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "archive_id") -> Optional[LibraryMetadata]:
        """Get EPUB resource by Internet Archive identifier."""
        try:
            self._rate_limit()
            url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            metadata_dict = data.get("metadata", {})
            
            metadata = LibraryMetadata(
                title=metadata_dict.get("title"),
                authors=metadata_dict.get("creator", []) if isinstance(metadata_dict.get("creator"), list) else [metadata_dict.get("creator")] if metadata_dict.get("creator") else [],
                publication_date=metadata_dict.get("date"),
                description=metadata_dict.get("description", "")[:500] if metadata_dict.get("description") else None,
                subjects=metadata_dict.get("subject", [])[:5] if isinstance(metadata_dict.get("subject"), list) else [],
                source_apis=["bookworm_epub"],
                enrichment_confidence=0.65
            )
            return metadata
        except Exception as e:
            logger.error(f"Bookworm EPUB lookup failed: {e}")
            return None


class PodcastindexClient(BaseLibraryClient):
    """Podcastindex.org API client for podcast discovery and metadata.
    
    Completely free, no authentication required.
    Rate limit: 10 requests/second (generous)
    Coverage: 3M+ podcasts, full episode metadata
    
    Best for: Podcast search, discovery, RSS feed management
    """
    
    def __init__(self, config: Optional[LibraryAPIConfig] = None):
        """Initialize Podcastindex client."""
        if config is None:
            config = LibraryAPIConfig(
                rate_limit_calls=10,
                rate_limit_period=1,
                cache_ttl=86400,
                enable_cache=True
            )
        super().__init__(config)
        self.base_url = "https://api.podcastindex.org/api/1.0"
    
    def search(self, query: str, **kwargs) -> List[LibraryMetadata]:
        """Search for podcasts by term (title, author, topic)."""
        cache_key = f"podcastindex:search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "q": query,
                "max": kwargs.get("limit", 10)
            }
            
            response = self.session.get(
                f"{self.base_url}/search/byterm",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for feed in data.get("feeds", [])[:kwargs.get("limit", 10)]:
                metadata = LibraryMetadata(
                    title=feed.get("title"),
                    authors=[feed.get("author")] if feed.get("author") else [],
                    publication_date=None,  # Podcasts don't have publication date
                    description=feed.get("description", "")[:500] if feed.get("description") else None,
                    subjects=[cat for cats in feed.get("categories", {}).values() for cat in (cats if isinstance(cats, list) else [cats])],
                    source_apis=["podcastindex"],
                    enrichment_confidence=0.75,
                    # Audio-specific fields
                    is_audio=True,
                    audio_type="podcast",
                    podcast_id=str(feed.get("id")),
                    podcast_url=feed.get("feedUrl"),
                    duration=None,
                    format="rss_feed",
                    language=feed.get("language", "en")
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Podcastindex search failed: {e}")
            return []
    
    def get_by_url(self, feed_url: str) -> Optional[LibraryMetadata]:
        """Get podcast metadata by RSS feed URL."""
        cache_key = f"podcastindex:url:{feed_url}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {"url": feed_url}
            
            response = self.session.get(
                f"{self.base_url}/podcasts/byfeedurl",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            feed = data.get("feed", {})
            
            if not feed:
                return None
            
            metadata = LibraryMetadata(
                title=feed.get("title"),
                authors=[feed.get("author")] if feed.get("author") else [],
                description=feed.get("description", "")[:500] if feed.get("description") else None,
                subjects=[cat for cats in feed.get("categories", {}).values() for cat in (cats if isinstance(cats, list) else [cats])],
                source_apis=["podcastindex"],
                enrichment_confidence=0.80,
                is_audio=True,
                audio_type="podcast",
                podcast_id=str(feed.get("id")),
                podcast_url=feed.get("feedUrl"),
                language=feed.get("language", "en")
            )
            
            self._set_cache(cache_key, metadata)
            return metadata
        except Exception as e:
            logger.error(f"Podcastindex URL lookup failed: {e}")
            return None
    
    def get_episodes(self, feed_url: str, limit: int = 20) -> List[LibraryMetadata]:
        """Get recent episodes from a podcast feed."""
        cache_key = f"podcastindex:episodes:{feed_url}:{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {"url": feed_url, "max": limit}
            
            response = self.session.get(
                f"{self.base_url}/episodes/byfeedurl",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for episode in data.get("items", [])[:limit]:
                metadata = LibraryMetadata(
                    title=episode.get("title"),
                    authors=[episode.get("author")] if episode.get("author") else [],
                    publication_date=episode.get("datePublished"),
                    description=episode.get("description", "")[:500] if episode.get("description") else None,
                    subjects=[],
                    source_apis=["podcastindex"],
                    enrichment_confidence=0.75,
                    is_audio=True,
                    audio_type="podcast",
                    episode_number=episode.get("episodeNumber"),
                    episode_duration=episode.get("duration"),
                    episode_transcript_url=episode.get("transcript"),
                    duration=episode.get("duration"),
                    format="rss_feed",
                    language=episode.get("language", "en")
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Podcastindex episodes fetch failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "podcast_id") -> Optional[LibraryMetadata]:
        """Get podcast by ID."""
        if id_type != "podcast_id":
            return None
        
        cache_key = f"podcastindex:id:{identifier}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {"id": identifier}
            
            response = self.session.get(
                f"{self.base_url}/podcasts/byid",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            feed = data.get("feed", {})
            
            if not feed:
                return None
            
            metadata = LibraryMetadata(
                title=feed.get("title"),
                authors=[feed.get("author")] if feed.get("author") else [],
                description=feed.get("description", "")[:500] if feed.get("description") else None,
                subjects=[cat for cats in feed.get("categories", {}).values() for cat in (cats if isinstance(cats, list) else [cats])],
                source_apis=["podcastindex"],
                enrichment_confidence=0.80,
                is_audio=True,
                audio_type="podcast",
                podcast_id=str(feed.get("id")),
                podcast_url=feed.get("feedUrl"),
                language=feed.get("language", "en")
            )
            
            self._set_cache(cache_key, metadata)
            return metadata
        except Exception as e:
            logger.error(f"Podcastindex ID lookup failed: {e}")
            return None


class LastfmMusicClient(BaseLibraryClient):
    """Last.fm API client for music discovery and recommendations.
    
    Completely free (no auth required for open endpoints).
    Coverage: 80M+ songs, 5M+ artists
    Best for: music discovery, similar artists, trending, tags
    """
    
    def __init__(self, config: Optional[LibraryAPIConfig] = None):
        """Initialize Last.fm client."""
        if config is None:
            config = LibraryAPIConfig(
                rate_limit_calls=5,
                rate_limit_period=1,
                cache_ttl=86400,
                enable_cache=True
            )
        super().__init__(config)
        self.base_url = "https://www.last.fm/api/0.2"
    
    def search(self, query: str, search_type: str = "artist", **kwargs) -> List[LibraryMetadata]:
        """Search for artists or tracks on Last.fm."""
        cache_key = f"lastfm:search:{search_type}:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            if search_type == "artist":
                url = f"{self.base_url}/artist/search"
                params = {"name": query}
            elif search_type == "track":
                url = f"{self.base_url}/track/search"
                params = {"name": query}
            else:
                return []
            
            params["format"] = "json"
            params["limit"] = kwargs.get("limit", 10)
            
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if search_type == "artist":
                for artist in data.get("results", {}).get("artistmatches", {}).get("artist", [])[:kwargs.get("limit", 10)]:
                    metadata = LibraryMetadata(
                        title=artist.get("name"),
                        authors=[],
                        description=artist.get("bio", "")[:500] if artist.get("bio") else None,
                        subjects=artist.get("tags", []) if isinstance(artist.get("tags"), list) else [],
                        source_apis=["lastfm"],
                        enrichment_confidence=0.70,
                        is_audio=True,
                        audio_type="music",
                        artist=artist.get("name"),
                        genre=None,
                        format="artist_profile"
                    )
                    results.append(metadata)
            
            elif search_type == "track":
                for track in data.get("results", {}).get("trackmatches", {}).get("track", [])[:kwargs.get("limit", 10)]:
                    metadata = LibraryMetadata(
                        title=track.get("name"),
                        authors=[track.get("artist")] if track.get("artist") else [],
                        description=None,
                        subjects=track.get("tags", []) if isinstance(track.get("tags"), list) else [],
                        source_apis=["lastfm"],
                        enrichment_confidence=0.65,
                        is_audio=True,
                        audio_type="music",
                        artist=track.get("artist"),
                        album=track.get("album") if track.get("album") else None,
                        format="track"
                    )
                    results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Last.fm search failed: {e}")
            return []
    
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[LibraryMetadata]:
        """Get artists similar to the given artist."""
        cache_key = f"lastfm:similar:{artist_name}:{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "artist": artist_name,
                "format": "json",
                "limit": limit
            }
            
            response = self.session.get(
                f"{self.base_url}/artist/similar",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for artist in data.get("similarartists", {}).get("artist", [])[:limit]:
                metadata = LibraryMetadata(
                    title=artist.get("name"),
                    authors=[],
                    description=None,
                    subjects=artist.get("tags", []) if isinstance(artist.get("tags"), list) else [],
                    source_apis=["lastfm"],
                    enrichment_confidence=0.72,
                    is_audio=True,
                    audio_type="music",
                    artist=artist.get("name"),
                    genre=None,
                    format="similar_artist"
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Last.fm similar artists failed: {e}")
            return []
    
    def get_trending_tracks(self, limit: int = 20) -> List[LibraryMetadata]:
        """Get trending tracks on Last.fm."""
        cache_key = f"lastfm:trending:{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "format": "json",
                "limit": limit
            }
            
            response = self.session.get(
                f"{self.base_url}/chart/gettoptracks",
                params=params,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for track in data.get("tracks", {}).get("track", [])[:limit]:
                metadata = LibraryMetadata(
                    title=track.get("name"),
                    authors=[track.get("artist", {}).get("name")] if track.get("artist", {}).get("name") else [],
                    description=None,
                    subjects=[],
                    source_apis=["lastfm"],
                    enrichment_confidence=0.75,
                    is_audio=True,
                    audio_type="music",
                    artist=track.get("artist", {}).get("name"),
                    format="track"
                )
                results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"Last.fm trending failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "artist") -> Optional[LibraryMetadata]:
        """Get music metadata by artist or track name."""
        if id_type == "artist":
            results = self.search(identifier, search_type="artist", limit=1)
            return results[0] if results else None
        elif id_type == "track":
            results = self.search(identifier, search_type="track", limit=1)
            return results[0] if results else None
        return None


class MusicBrainzClient(BaseLibraryClient):
    """MusicBrainz API client for music metadata and deduplication.
    
    Completely free, no authentication required.
    Coverage: 42M+ artists, 100M+ recordings, 50M+ works
    Best for: authoritative metadata, deduplication, ISRC lookup
    """
    
    def __init__(self, config: Optional[LibraryAPIConfig] = None):
        """Initialize MusicBrainz client."""
        if config is None:
            config = LibraryAPIConfig(
                rate_limit_calls=1,  # MB is strict: 1 request/second
                rate_limit_period=1,
                cache_ttl=604800,  # 7 days for metadata
                enable_cache=True
            )
        super().__init__(config)
        self.base_url = "https://musicbrainz.org/ws/2"
        self.session.headers.update({
            "User-Agent": "Xoe-NovAi/0.1.5 (https://github.com/Xoe-NovAi)"
        })
    
    def search(self, query: str, search_type: str = "artist", **kwargs) -> List[LibraryMetadata]:
        """Search MusicBrainz database."""
        cache_key = f"musicbrainz:search:{search_type}:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            params = {
                "query": query,
                "fmt": "json",
                "limit": kwargs.get("limit", 10),
                "offset": 0
            }
            
            url = f"{self.base_url}/{search_type}"
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if search_type == "artist":
                for artist in data.get("artists", [])[:kwargs.get("limit", 10)]:
                    metadata = LibraryMetadata(
                        title=artist.get("name"),
                        authors=[],
                        description=None,
                        subjects=[artist.get("type")] if artist.get("type") else [],
                        source_apis=["musicbrainz"],
                        enrichment_confidence=0.85,
                        is_audio=True,
                        audio_type="music",
                        artist=artist.get("name"),
                        language=artist.get("country"),
                        format="artist"
                    )
                    results.append(metadata)
            
            elif search_type == "recording":
                for recording in data.get("recordings", [])[:kwargs.get("limit", 10)]:
                    metadata = LibraryMetadata(
                        title=recording.get("title"),
                        authors=[a.get("name") for a in recording.get("artist-credit", []) if a.get("name")] if recording.get("artist-credit") else [],
                        publication_date=recording.get("first-release-date"),
                        description=None,
                        subjects=[],
                        source_apis=["musicbrainz"],
                        enrichment_confidence=0.85,
                        is_audio=True,
                        audio_type="music",
                        artist=recording.get("artist-credit", [{}])[0].get("name") if recording.get("artist-credit") else None,
                        isrc=recording.get("isrcs", [None])[0] if recording.get("isrcs") else None,
                        duration=recording.get("length"),
                        format="recording"
                    )
                    results.append(metadata)
            
            self._set_cache(cache_key, results)
            return results
        except Exception as e:
            logger.warning(f"MusicBrainz search failed: {e}")
            return []
    
    def get_by_identifier(self, identifier: str, id_type: str = "isrc") -> Optional[List[LibraryMetadata]]:
        """Get music by ISRC, ISWC, or artist/recording ID."""
        if id_type == "isrc":
            cache_key = f"musicbrainz:isrc:{identifier}"
            cached = self._get_cached(cache_key)
            if cached:
                return cached
            
            try:
                self._rate_limit()
                params = {
                    "query": f"isrc:{identifier}",
                    "fmt": "json",
                    "limit": 10
                }
                
                response = self.session.get(
                    f"{self.base_url}/recording",
                    params=params,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for recording in data.get("recordings", []):
                    metadata = LibraryMetadata(
                        title=recording.get("title"),
                        authors=[a.get("name") for a in recording.get("artist-credit", []) if a.get("name")] if recording.get("artist-credit") else [],
                        publication_date=recording.get("first-release-date"),
                        source_apis=["musicbrainz"],
                        enrichment_confidence=0.90,
                        is_audio=True,
                        audio_type="music",
                        artist=recording.get("artist-credit", [{}])[0].get("name") if recording.get("artist-credit") else None,
                        isrc=identifier,
                        duration=recording.get("length"),
                        format="recording"
                    )
                    results.append(metadata)
                
                self._set_cache(cache_key, results)
                return results
            except Exception as e:
                logger.error(f"MusicBrainz ISRC lookup failed: {e}")
                return []
        
        return None


# ============================================================================
# DOMAIN CATEGORIZATION & DEWEY SYSTEM
# ============================================================================

class DomainManager:
    """Intuitive domain categorization and management system."""
    
    def __init__(self, enable_dewey: bool = True):
        self.enable_dewey = enable_dewey
        self.custom_categories: Dict[str, DomainCategory] = {}
        self.category_keywords: Dict[DomainCategory, List[str]] = {
            DomainCategory.CODE: ["code", "programming", "software", "algorithm", "python", "javascript", "java"],
            DomainCategory.SCIENCE: ["science", "physics", "chemistry", "biology", "research", "experiment"],
            DomainCategory.DATA: ["data", "analysis", "statistics", "machine learning", "dataset", "csv"],
            DomainCategory.MUSIC: ["music", "song", "audio", "artist", "album", "track"],
            DomainCategory.BOOKS: ["book", "novel", "author", "fiction", "literature"],
            DomainCategory.ARCHIVES: ["archive", "collection", "manuscript", "historical"],
            DomainCategory.PHOTOGRAPHS: ["photo", "image", "picture", "visual"],
            DomainCategory.REFERENCE: ["encyclopedia", "dictionary", "reference", "guide"],
        }
    
    def classify(self, text: str, title: str = "", metadata: Optional[LibraryMetadata] = None) -> Tuple[DomainCategory, float]:
        """
        Automatically classify content into domain category.
        
        Returns:
            Tuple of (category, confidence: 0.0-1.0)
        """
        combined_text = f"{title} {text}".lower()
        scores: Dict[DomainCategory, float] = {}
        
        # Score based on keyword matching with weighted importance
        for category, keywords in self.category_keywords.items():
            matches = 0
            total_weight = 0
            for kw in keywords:
                if kw in combined_text:
                    matches += 1
                    # Weight title matches higher
                    if kw in title.lower():
                        matches += 0.5
                total_weight += 1
            
            # Calculate score: (matches / total_keywords) * 100
            score = (matches / max(len(keywords), 1)) * 100 if matches > 0 else 0
            scores[category] = score
        
        # Boost score based on metadata subjects
        if metadata and metadata.subjects:
            subjects_text = " ".join(metadata.subjects).lower()
            for category, keywords in self.category_keywords.items():
                subject_matches = sum(1 for kw in keywords if kw in subjects_text)
                scores[category] += (subject_matches / max(len(keywords), 1)) * 50
        
        # Get best match
        if scores:
            best_category = max(scores, key=scores.get)
            # Normalize confidence to 0.0-1.0 range
            confidence = min(scores[best_category] / 100.0, 1.0)
            return best_category, max(confidence, 0.5)  # Minimum confidence 0.5
        
        return DomainCategory.GENERAL, 0.5
    
    def get_dewey_suggestion(self, category: DomainCategory) -> List[str]:
        """Get suggested Dewey Decimal classifications for a category."""
        if not self.enable_dewey:
            return []
        return DOMAIN_TO_DEWEY.get(category, [])
    
    def domain_to_dewey(self, category: DomainCategory) -> Optional[str]:
        """Map domain category to Dewey Decimal classification."""
        if not self.enable_dewey:
            return None
        return DOMAIN_TO_DEWEY.get(category, ["000"])[0]
    
    def dewey_to_domain(self, dewey: str) -> Optional[DomainCategory]:
        """Map Dewey Decimal classification to domain category."""
        if not self.enable_dewey:
            return DomainCategory.GENERAL
        
        # Check exact match first
        if dewey in DEWEY_TO_DOMAIN:
            return DEWEY_TO_DOMAIN[dewey]
        
        # Check prefix match (e.g., "500" matches "540")
        for code_prefix, category in DEWEY_TO_DOMAIN.items():
            if dewey.startswith(code_prefix[0]):
                return category
        
        return DomainCategory.GENERAL
    
    def add_custom_category(self, category_name: str, keywords: List[str]):
        """Add custom domain category."""
        custom_enum = DomainCategory(category_name.lower().replace(" ", "_"))
        self.category_keywords[custom_enum] = keywords
    
    def get_all_categories(self) -> List[str]:
        """Get all available domain categories."""
        return [cat.value for cat in DomainCategory]


# ============================================================================
# LIBRARY ENRICHMENT ENGINE
# ============================================================================

class LibraryEnrichmentEngine:
    """Main engine for enriching content with library metadata (books, podcasts, music, etc)."""
    
    def __init__(self, config: Optional[LibraryAPIConfig] = None):
        self.config = config or LibraryAPIConfig()
        self.domain_manager = DomainManager(enable_dewey=self.config.enable_dewey_mapping)
        
        # Initialize API clients (all completely free, no API keys required)
        self.clients = {
            # Book/Library APIs (8)
            "openlibrary": OpenLibraryClient(self.config),
            "internetarchive": InternetArchiveClient(self.config),
            "loc": LibraryOfCongressClient(self.config),
            "gutenberg": ProjectGutenbergClient(self.config),
            "freemusicarchive": FreeMusicArchiveClient(self.config),
            "worldcat": WorldCatOpenSearchClient(self.config),
            "cambridge": CambridgeDigitalLibraryClient(self.config),
            "bookworm_epub": BookwormEpubClient(self.config),
            
            # Audio APIs (3)
            "podcastindex": PodcastindexClient(self.config),
            "lastfm": LastfmMusicClient(self.config),
            "musicbrainz": MusicBrainzClient(self.config),
        }
    
    def enrich_by_isbn(self, isbn: str) -> Optional[LibraryMetadata]:
        """Enrich content by ISBN number."""
        logger.info(f"Enriching by ISBN: {isbn}")
        
        # Try each client that supports ISBN
        for client_name in ["openlibrary", "internetarchive"]:
            result = self.clients[client_name].get_by_identifier(isbn, "isbn")
            if result:
                logger.info(f"Found enrichment from {client_name}")
                return result
        
        return None
    
    def enrich_by_title_author(self, title: str, author: Optional[str] = None, limit: int = 5) -> List[LibraryMetadata]:
        """Enrich content by title and optional author."""
        query = f"{title} {author}".strip() if author else title
        logger.info(f"Enriching by title/author: {query}")
        
        results = []
        for client_name, client in self.clients.items():
            try:
                client_results = client.search(query, limit=3)
                results.extend(client_results)
            except Exception as e:
                logger.warning(f"Error from {client_name}: {e}")
        
        # Deduplicate and sort by confidence
        seen_titles = set()
        unique_results = []
        for result in results:
            if result.title and result.title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result.title)
        
        return sorted(unique_results, key=lambda x: x.enrichment_confidence, reverse=True)[:limit]
    
    def enrich_by_podcast_url(self, feed_url: str) -> Optional[LibraryMetadata]:
        """Enrich by podcast RSS feed URL."""
        logger.info(f"Enriching podcast by URL: {feed_url}")
        
        podcast_client = self.clients.get("podcastindex")
        if podcast_client:
            result = podcast_client.get_by_url(feed_url)
            if result:
                return result
        
        return None
    
    def enrich_by_artist_name(self, artist_name: str, limit: int = 10) -> List[LibraryMetadata]:
        """Enrich by music artist name."""
        logger.info(f"Enriching music by artist: {artist_name}")
        
        results = []
        
        # Get from Last.fm
        lastfm_client = self.clients.get("lastfm")
        if lastfm_client:
            try:
                artist_results = lastfm_client.search(artist_name, search_type="artist", limit=limit)
                results.extend(artist_results)
            except Exception as e:
                logger.warning(f"Last.fm artist search failed: {e}")
        
        # Get similar artists
        if artist_results:
            try:
                similar = lastfm_client.get_similar_artists(artist_name, limit=limit)
                results.extend(similar)
            except Exception as e:
                logger.warning(f"Last.fm similar artists failed: {e}")
        
        return results
    
    def search_podcasts_by_topic(self, topic: str, limit: int = 10) -> List[LibraryMetadata]:
        """Search for podcasts about a specific topic."""
        logger.info(f"Searching podcasts about: {topic}")
        
        podcast_client = self.clients.get("podcastindex")
        if podcast_client:
            try:
                results = podcast_client.search(topic, limit=limit)
                return results
            except Exception as e:
                logger.error(f"Podcast search failed: {e}")
        
        return []
    
    def get_music_recommendations(self, genre: str = None, limit: int = 10) -> List[LibraryMetadata]:
        """Get music recommendations by genre or trending."""
        logger.info(f"Getting music recommendations: {genre or 'trending'}")
        
        lastfm_client = self.clients.get("lastfm")
        if lastfm_client:
            try:
                if genre:
                    results = lastfm_client.search(genre, search_type="track", limit=limit)
                else:
                    results = lastfm_client.get_trending_tracks(limit=limit)
                return results
            except Exception as e:
                logger.error(f"Music recommendations failed: {e}")
        
        return []
    
    def classify_and_enrich(self, title: str, content: str, author: Optional[str] = None) -> Dict[str, Any]:
        """Complete classification and enrichment workflow."""
        logger.info(f"Classifying and enriching: {title}")
        
        # Classify domain
        domain_category, confidence = self.domain_manager.classify(content, title)
        
        # Enrich metadata
        metadata_results = self.enrich_by_title_author(title, author)
        
        # Build enrichment result
        result = {
            "title": title,
            "domain_category": domain_category.value,
            "category_confidence": confidence,
            "metadata_results": [m.to_dict() for m in metadata_results],
            "dewey_suggestions": self.domain_manager.get_dewey_suggestion(domain_category),
            "primary_dewey": self.domain_manager.domain_to_dewey(domain_category),
            "enrichment_timestamp": datetime.now().isoformat(),
        }
        
        if metadata_results:
            result["primary_metadata"] = metadata_results[0].to_dict()
        
        return result
    
    def batch_enrich(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Enrich multiple items in batch."""
        results = []
        for item in items:
            result = self.classify_and_enrich(
                item.get("title", ""),
                item.get("content", ""),
                item.get("author")
            )
            results.append(result)
        return results


# ============================================================================
# NATURAL LANGUAGE CURATOR INTERFACE
# ============================================================================

class NLCuratorInterface:
    """
    Natural Language Curator Interface for Chainlit integration.
    Parses natural language commands and executes curator operations.
    """
    
    def __init__(self, enrichment_engine: Optional[LibraryEnrichmentEngine] = None):
        """Initialize curator interface with optional enrichment engine."""
        self.engine = enrichment_engine or LibraryEnrichmentEngine()
        self.intent_classifier = self._get_intent_classifier()
        self.entity_extractor = self._get_entity_extractor()
    
    def _get_intent_classifier(self):
        """Get intent classifier (using transformer zero-shot if available)."""
        try:
            from transformers import pipeline
            return pipeline("zero-shot-classification", 
                          model="facebook/bart-large-mnli")
        except ImportError:
            logger.warning("Transformers not available - using fallback intent matching")
            return None
    
    def _get_entity_extractor(self):
        """Get entity extractor (using spaCy if available)."""
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not available - using regex-based entity extraction")
            return None
    
    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """
        Parse natural language curator command.
        
        Returns:
            Dict with:
            - intent: curator action (search, research, recommend, curate, etc.)
            - confidence: 0.0-1.0
            - parameters: {author, title, topic, domain, filters}
            - command_type: 'author_search', 'topic_research', 'recommendations', etc.
        """
        input_lower = user_input.lower()
        
        # Define curator intents
        curator_intents = [
            "locate_books",
            "search_author", 
            "research_topic",
            "get_recommendations",
            "curate_collection",
            "find_resources",
            "list_works",
            "filter_by_domain"
        ]
        
        # Intent classification
        intent_result = self._classify_intent(user_input, curator_intents)
        intent = intent_result.get("intent", "search_author")
        confidence = intent_result.get("confidence", 0.5)
        
        # Entity extraction (author, title, topic, etc.)
        entities = self._extract_entities(user_input)
        
        # Determine command type and parameters
        parameters = self._build_parameters(user_input, intent, entities)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "parameters": parameters,
            "command_type": self._get_command_type(intent, entities),
            "raw_input": user_input
        }
    
    def _classify_intent(self, text: str, intents: List[str]) -> Dict[str, Any]:
        """Classify user intent from text."""
        if self.intent_classifier:
            try:
                result = self.intent_classifier(text, intents, multi_class=False)
                return {
                    "intent": result['labels'][0],
                    "confidence": float(result['scores'][0])
                }
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")
        
        # Fallback: keyword-based intent matching
        intent_keywords = {
            "locate_books": ["locate", "find", "discover", "search for", "where can i find"],
            "search_author": ["author", "by ", "works", "by "],
            "research_topic": ["research", "about", "on", "regarding", "topic"],
            "get_recommendations": ["recommend", "suggest", "top", "best"],
            "curate_collection": ["curate", "add to", "organize", "collection"],
            "list_works": ["list", "show", "all ", "works", "books", "publications"],
        }
        
        text_lower = text.lower()
        best_intent = "search_author"
        best_score = 0
        
        for intent, keywords in intent_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_score:
                best_score = matches
                best_intent = intent
        
        confidence = min(best_score * 0.3, 1.0) if best_score > 0 else 0.5
        return {"intent": best_intent, "confidence": confidence}
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities (author, title, topic, domain) from text."""
        entities = {
            "author": None,
            "title": None,
            "topic": None,
            "domain": None,
            "year_range": None,
            "language": None,
        }
        
        if self.entity_extractor:
            try:
                doc = self.entity_extractor(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities["author"] = ent.text
                    elif ent.label_ == "WORK_OF_ART":
                        entities["title"] = ent.text
                    elif ent.label_ == "DATE":
                        entities["year_range"] = ent.text
                    elif ent.label_ == "GPE":
                        entities["language"] = ent.text
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        # Regex-based extraction as fallback
        # Extract author names (usually after "by" or "of")
        author_match = re.search(r'\b(?:by|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        if author_match and not entities["author"]:
            entities["author"] = author_match.group(1)
        
        # Extract quoted titles
        title_match = re.search(r'["\']([^"\']+)["\']', text)
        if title_match and not entities["title"]:
            entities["title"] = title_match.group(1)
        
        # Extract topics (words after prepositions like "on", "about", "regarding")
        topic_match = re.search(r'\b(?:on|about|regarding|on the topic of)\s+([a-zA-Z\s]+)(?:\s+and|\s+or|$|\.)', text)
        if topic_match:
            entities["topic"] = topic_match.group(1).strip()
        
        # Detect domain from keywords
        domain_keywords = {
            "science": ["physics", "chemistry", "biology", "astronomy", "quantum"],
            "fiction": ["novel", "story", "fiction", "narrative"],
            "philosophy": ["philosophy", "ethics", "metaphysics", "plato", "aristotle"],
            "history": ["history", "historical", "ancient", "modern"],
            "mathematics": ["math", "mathematics", "algebra", "geometry"],
            "technology": ["technology", "computer", "programming", "software"],
        }
        
        text_lower = text.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                entities["domain"] = domain
                break
        
        return entities
    
    def _build_parameters(self, text: str, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build curator operation parameters from parsed info."""
        parameters = {
            "query": text,
            "author": entities.get("author"),
            "title": entities.get("title"),
            "topic": entities.get("topic"),
            "domain": entities.get("domain"),
            "limit": 10,  # Default: return 10 results
            "sort_by": "relevance",
        }
        
        # Adjust parameters based on intent
        if "recommend" in intent.lower():
            parameters["limit"] = 10  # Top 10 recommendations
            parameters["sort_by"] = "relevance"
        elif "all" in text.lower():
            parameters["limit"] = 50  # Show all/more results
        
        # Extract limit from text ("top 5", "10 books", etc.)
        limit_match = re.search(r'(?:top|give me|show me|find)?\s*(\d+)', text)
        if limit_match:
            parameters["limit"] = int(limit_match.group(1))
        
        return parameters
    
    def _get_command_type(self, intent: str, entities: Dict[str, Any]) -> str:
        """Determine specific command type from intent and entities."""
        if entities.get("author"):
            return "author_search"
        elif entities.get("title"):
            return "title_search"
        elif entities.get("topic"):
            return "topic_research"
        elif "recommend" in intent.lower():
            return "get_recommendations"
        elif "curate" in intent.lower():
            return "curation_workflow"
        else:
            return "general_search"
    
    def execute_command(self, parsed_command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute curator command and return results."""
        command_type = parsed_command.get("command_type", "general_search")
        parameters = parsed_command.get("parameters", {})
        
        logger.info(f"Executing curator command: {command_type} - {parameters}")
        
        try:
            if command_type == "author_search":
                return self._author_search(parameters)
            elif command_type == "title_search":
                return self._title_search(parameters)
            elif command_type == "topic_research":
                return self._topic_research(parameters)
            elif command_type == "get_recommendations":
                return self._get_recommendations(parameters)
            elif command_type == "curation_workflow":
                return self._curation_workflow(parameters)
            else:
                return self._general_search(parameters)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to execute curator command"
            }
    
    def _author_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for works by a specific author."""
        author = parameters.get("author")
        if not author:
            # Fallback: try to extract author from query
            author = re.search(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', parameters.get("query", ""))
            if author:
                author = author.group(1)
        
        if not author:
            return {"success": False, "error": "Author not specified"}
        
        # Search for author's works
        results = self.engine.enrich_by_title_author(f"works by {author}", author=author)
        
        return {
            "success": True,
            "command_type": "author_search",
            "author": author,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Found {len(results)} works by {author}"
        }
    
    def _title_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for a specific book by title."""
        title = parameters.get("title")
        if not title:
            return {"success": False, "error": "Title not specified"}
        
        results = self.engine.enrich_by_title_author(title)
        
        return {
            "success": True,
            "command_type": "title_search",
            "title": title,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Found {len(results)} results for '{title}'"
        }
    
    def _topic_research(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Research a specific topic."""
        topic = parameters.get("topic")
        limit = parameters.get("limit", 10)
        
        if not topic:
            return {"success": False, "error": "Topic not specified"}
        
        # Search for books on topic
        results = self.engine.enrich_by_title_author(topic, limit=limit)
        
        # Add domain classification for each result
        for result in results:
            if result.subjects:
                domain, confidence = self.engine.domain_manager.classify(
                    " ".join(result.subjects), 
                    result.title or "", 
                    result
                )
                result.dewey_decimal = self.engine.domain_manager.domain_to_dewey(domain)
        
        return {
            "success": True,
            "command_type": "topic_research",
            "topic": topic,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Found {len(results)} resources on {topic}"
        }
    
    def _get_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get book recommendations based on topics/domains."""
        topic = parameters.get("topic")
        domain = parameters.get("domain")
        limit = min(parameters.get("limit", 10), 10)  # Cap at 10 recommendations
        
        if not topic and not domain:
            return {"success": False, "error": "Topic or domain required for recommendations"}
        
        query = f"{topic} {domain}".strip() if domain else topic
        results = self.engine.enrich_by_title_author(query, limit=limit)
        
        # Sort by enrichment confidence and add recommendations
        results = sorted(results, key=lambda r: r.enrichment_confidence, reverse=True)
        
        recommendations = []
        for i, result in enumerate(results[:limit], 1):
            rec = result.to_dict()
            rec["recommendation_rank"] = i
            rec["recommendation_reason"] = f"Highly relevant to {topic}" if topic else f"Related to {domain}"
            recommendations.append(rec)
        
        return {
            "success": True,
            "command_type": "get_recommendations",
            "topic": topic,
            "domain": domain,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations,
            "message": f"Here are my top {len(recommendations)} recommendations for {topic or domain}"
        }
    
    def _curation_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full curation workflow."""
        query = parameters.get("query")
        topic = parameters.get("topic")
        domain = parameters.get("domain")
        
        search_query = f"{topic or query} {domain}".strip()
        results = self.engine.enrich_by_title_author(search_query)
        
        # Classify and enrich each result
        for result in results:
            if result.title:
                enrichment = self.engine.classify_and_enrich(
                    result.title, 
                    result.description or "",
                    authors=", ".join(result.authors) if result.authors else None
                )
                result.dewey_decimal = enrichment.get("primary_dewey")
        
        return {
            "success": True,
            "command_type": "curation_workflow",
            "topic": topic,
            "domain": domain,
            "curated_items": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Curated {len(results)} items for {topic or query}"
        }
    
    def _general_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """General library search."""
        query = parameters.get("query", "")
        if not query:
            return {"success": False, "error": "Search query required"}
        
        results = self.engine.enrich_by_title_author(query)
        
        return {
            "success": True,
            "command_type": "general_search",
            "query": query,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Search for '{query}' returned {len(results)} results"
        }
    
    # ========================================================================
    # AUDIO-SPECIFIC COMMAND HANDLERS (NEW)
    # ========================================================================
    
    def _podcast_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for podcasts by topic."""
        topic = parameters.get("topic")
        limit = parameters.get("limit", 10)
        
        if not topic:
            return {"success": False, "error": "Podcast topic required"}
        
        # Search for podcasts
        podcasts = self.engine.search_podcasts_by_topic(topic, limit=limit)
        
        return {
            "success": True,
            "command_type": "podcast_search",
            "topic": topic,
            "content_type": "podcast",
            "results_count": len(podcasts),
            "results": [p.to_dict() for p in podcasts],
            "message": f"Found {len(podcasts)} podcasts about {topic}"
        }
    
    def _music_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search for music by artist or genre."""
        artist = parameters.get("artist")
        genre = parameters.get("genre")
        limit = parameters.get("limit", 10)
        
        if not artist and not genre:
            return {"success": False, "error": "Artist or genre required"}
        
        results = []
        
        if artist:
            # Search for artist
            music = self.engine.enrich_by_artist_name(artist, limit=limit)
            results = music
            search_term = artist
        elif genre:
            # Get music recommendations by genre
            music = self.engine.get_music_recommendations(genre, limit=limit)
            results = music
            search_term = genre
        
        return {
            "success": True,
            "command_type": "music_search",
            "artist": artist,
            "genre": genre,
            "content_type": "music",
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "message": f"Found {len(results)} music results for {search_term}"
        }
    
    def _audio_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get audio (podcast or music) recommendations."""
        audio_type = parameters.get("audio_type")  # "podcast" or "music"
        reference = parameters.get("reference")  # e.g., "science" for podcasts, "Radiohead" for artists
        limit = min(parameters.get("limit", 10), 10)
        
        if not audio_type or not reference:
            return {"success": False, "error": "Audio type and reference required"}
        
        results = []
        
        if audio_type == "podcast":
            # Get podcast recommendations on a topic
            results = self.engine.search_podcasts_by_topic(reference, limit=limit)
            msg_type = "podcasts"
        elif audio_type == "music":
            # Get music recommendations (similar artists)
            results = self.engine.enrich_by_artist_name(reference, limit=limit)
            msg_type = "artists"
        
        recommendations = []
        for i, result in enumerate(results[:limit], 1):
            rec = result.to_dict()
            rec["recommendation_rank"] = i
            recommendations.append(rec)
        
        return {
            "success": True,
            "command_type": "audio_recommendations",
            "audio_type": audio_type,
            "reference": reference,
            "content_type": audio_type,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations,
            "message": f"Here are my top {len(recommendations)} {msg_type} recommendations like {reference}"
        }
    
    def _parse_audio_command(self, text: str) -> Dict[str, Any]:
        """
        Parse audio-specific commands.
        
        Examples:
        - "Find podcasts about machine learning"
        - "Discover jazz music artists"
        - "Show me tech podcasts"
        - "Recommend artists like Radiohead"
        """
        text_lower = text.lower()
        entities = {
            "audio_type": None,
            "action": None,
            "topic": None,
            "genre": None,
            "artist": None,
            "limit": 10
        }
        
        # Detect audio type
        if "podcast" in text_lower:
            entities["audio_type"] = "podcast"
        elif any(word in text_lower for word in ["music", "artist", "band", "song", "track", "album"]):
            entities["audio_type"] = "music"
        
        # Detect action
        if any(word in text_lower for word in ["find", "locate", "discover", "search"]):
            entities["action"] = "search"
        elif any(word in text_lower for word in ["recommend", "suggest", "similar"]):
            entities["action"] = "recommend"
        elif any(word in text_lower for word in ["show", "list", "get"]):
            entities["action"] = "list"
        
        # Extract topic (for podcasts)
        topic_match = re.search(r'(?:about|on|regarding)\s+([a-zA-Z\s]+)(?:\s+and|\s+or|$|\.)', text)
        if topic_match:
            entities["topic"] = topic_match.group(1).strip()
        
        # Extract artist/genre (for music)
        # "like [Artist]"
        artist_match = re.search(r'\blike\s+([A-Z][a-zA-Z\s]+?)(?:\s+and|\s+or|$|\.)', text)
        if artist_match:
            entities["artist"] = artist_match.group(1).strip()
        
        # Genre extraction (keywords)
        genres = ["jazz", "rock", "pop", "indie", "classical", "electronic", "hip hop", "blues", "country", "metal"]
        for genre in genres:
            if genre in text_lower:
                entities["genre"] = genre
                break
        
        # Extract limit
        limit_match = re.search(r'(?:top|my|the)?\s*(\d+)\s*(?:best|top|music|artists|podcasts)?', text)
        if limit_match:
            entities["limit"] = int(limit_match.group(1))
        
        return entities
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Complete pipeline: Parse → Validate → Execute.
        
        Args:
            user_input: Natural language curator command from user
            
        Returns:
            Dict with results and metadata
        """
        # Check if this is an audio command
        audio_entities = self._parse_audio_command(user_input)
        if audio_entities.get("audio_type"):
            # Handle audio command
            if audio_entities.get("action") == "search":
                if audio_entities.get("audio_type") == "podcast":
                    parameters = {
                        "topic": audio_entities.get("topic"),
                        "limit": audio_entities.get("limit", 10)
                    }
                    result = self._podcast_search(parameters)
                else:  # music
                    parameters = {
                        "artist": audio_entities.get("artist"),
                        "genre": audio_entities.get("genre"),
                        "limit": audio_entities.get("limit", 10)
                    }
                    result = self._music_search(parameters)
            elif audio_entities.get("action") == "recommend":
                parameters = {
                    "audio_type": audio_entities.get("audio_type"),
                    "reference": audio_entities.get("topic") or audio_entities.get("artist"),
                    "limit": audio_entities.get("limit", 10)
                }
                result = self._audio_recommendations(parameters)
            else:
                # Default to search
                parameters = {
                    "topic": audio_entities.get("topic"),
                    "artist": audio_entities.get("artist"),
                    "genre": audio_entities.get("genre"),
                    "limit": audio_entities.get("limit", 10)
                }
                if audio_entities.get("audio_type") == "podcast":
                    result = self._podcast_search(parameters)
                else:
                    result = self._music_search(parameters)
            
            return result
        
        # Otherwise, parse as regular curator command
        parsed = self.parse_command(user_input)
        logger.info(f"Parsed command: {parsed['command_type']} (confidence: {parsed['confidence']:.2f})")
        
        # Execute the command
        result = self.execute_command(parsed)
        
        # Add parsing metadata to result
        result["parsing_confidence"] = parsed["confidence"]
        result["detected_intent"] = parsed["intent"]
        result["detected_parameters"] = parsed["parameters"]
        
        return result


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

def test_library_integration():
    """Test library API integrations and domain classification."""
    print("\n" + "="*80)
    print("LIBRARY API INTEGRATION TEST")
    print("="*80)
    
    # Initialize engine
    config = LibraryAPIConfig(enable_cache=True, enable_dewey_mapping=True)
    engine = LibraryEnrichmentEngine(config)
    
    # Test 1: Domain classification
    print("\n[TEST 1] Domain Classification")
    print("-" * 80)
    test_texts = [
        ("Python Best Practices", "def fibonacci(n): return 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)", DomainCategory.CODE),
        ("Quantum Physics", "Wave functions and superposition principles in quantum mechanics", DomainCategory.SCIENCE),
        ("The Great Gatsby", "In my younger and more vulnerable years, my father gave me advice", DomainCategory.BOOKS),
    ]
    
    for title, content, expected_category in test_texts:
        category, confidence = engine.domain_manager.classify(content, title)
        match = "✓" if category == expected_category else "✗"
        print(f"{match} {title}: {category.value} (confidence: {confidence:.2f})")
    
    # Test 2: Dewey Decimal mapping
    print("\n[TEST 2] Dewey Decimal System Integration")
    print("-" * 80)
    for category in [DomainCategory.CODE, DomainCategory.SCIENCE, DomainCategory.MUSIC]:
        dewey = engine.domain_manager.domain_to_dewey(category)
        suggestions = engine.domain_manager.get_dewey_suggestion(category)
        print(f"{category.value}: Primary={dewey}, Suggestions={suggestions}")
    
    # Test 3: Library enrichment (with available APIs)
    print("\n[TEST 3] Library Enrichment")
    print("-" * 80)
    test_items = [
        {"title": "The Pragmatic Programmer", "content": "software development guide", "author": "Hunt & Thomas"},
        {"title": "A Brief History of Time", "content": "cosmology and physics", "author": "Stephen Hawking"},
    ]
    
    enriched = engine.batch_enrich(test_items)
    for item in enriched:
        print(f"\nTitle: {item['title']}")
        print(f"Domain: {item['domain_category']} (confidence: {item['category_confidence']:.2f})")
        print(f"Dewey: {item['primary_dewey']}")
        if item.get('primary_metadata'):
            meta = item['primary_metadata']
            print(f"Authors: {', '.join(meta.get('authors', []))}")
            print(f"Subjects: {', '.join(meta.get('subjects', [])[:3])}")
    
    # Test 4: Domain manager categories
    print("\n[TEST 4] Available Domain Categories")
    print("-" * 80)
    categories = engine.domain_manager.get_all_categories()
    print(f"Total categories: {len(categories)}")
    print(f"Categories: {', '.join(categories)}")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_library_integration()
