# Audio Enhancement Strategy: Podcasts & Music Integration
## Xoe-NovAi v0.1.5-audio-enhanced

**Date:** January 3, 2026  
**Status:** Strategy Phase - Ready for Implementation  
**Target Release:** v0.1.5-audio-enhanced  
**Timeline:** Phase 3 (Following Curator Enhancement)

---

## EXECUTIVE SUMMARY

Extend Xoe-NovAi library system with audio content (podcasts + music) using completely free, open APIs. Implementation will:

1. **Add 2 new API clients** (Podcastindex, Last.fm/MusicBrainz combo)
2. **Extend domain system** to cover 14 audio domains
3. **Enhance curator interface** with audio-specific commands
4. **Update crawler** to handle RSS feeds and music discovery
5. **Implement audio metadata** schema for consistent cataloging
6. **Add audio deduplication** via Redis + content hashing
7. **Maintain backward compatibility** with all existing code

---

## PHASE 3 IMPLEMENTATION ROADMAP

### **Stage 1: API Client Implementation** (2-3 hours)

#### 1.1 Podcastindex API Client
**File:** `app/XNAi_rag_app/library_api_integrations.py`

New class `PodcastindexClient`:
```python
class PodcastindexClient:
    """
    Podcastindex.org API client for podcast discovery and metadata.
    
    Completely free, no authentication required.
    Rate limit: 10 requests/second (generous)
    Coverage: 3M+ podcasts, full episode metadata
    """
    
    base_url = "https://api.podcastindex.org/api/1.0"
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]
    def get_by_url(self, rss_url: str) -> Dict[str, Any]
    def get_episodes(self, feed_url: str, limit: int = 20) -> Dict[str, Any]
    def get_categories(self) -> Dict[str, Any]
    def search_by_category(self, category: str, limit: int = 10) -> Dict[str, Any]
```

**Features:**
- Full-text search by podcast name, author, topic
- Get podcast metadata (title, description, image, RSS URL, episode count)
- Get recent episodes from podcast feed
- Browse all categories
- Get recommendations based on similar podcasts

**Metadata extracted:**
- Podcast ID, title, author, description, image
- RSS feed URL, website, language
- Episodes: guid, title, description, pubdate, duration, transcript_url
- Categories, ratings, popularity metrics

**Error handling:**
- Network errors → return empty results with error message
- Rate limit exceeded → exponential backoff with retry
- Invalid RSS feed → skip and continue

---

#### 1.2 Music API Clients
**File:** `app/XNAi_rag_app/library_api_integrations.py`

New class `LastfmMusicClient`:
```python
class LastfmMusicClient:
    """
    Last.fm API client for music discovery and recommendations.
    
    Completely free (no auth required, open endpoints).
    Coverage: 80M+ songs, 5M+ artists
    Best for: discovery, similar artists, trending, tags
    """
    
    base_url = "https://www.last.fm/api/0.2"
    
    def search_artist(self, artist: str, limit: int = 10) -> Dict[str, Any]
    def search_track(self, track: str, artist: str = None, limit: int = 10) -> Dict[str, Any]
    def get_similar_artists(self, artist: str, limit: int = 10) -> Dict[str, Any]
    def get_trending_tracks(self, limit: int = 20) -> Dict[str, Any]
    def search_by_genre(self, genre: str, limit: int = 10) -> Dict[str, Any]
```

**Features:**
- Artist search with metadata (bio, image, followers, tags)
- Track search with artist info
- Similar artists (great for recommendations)
- Trending tracks/artists (discovery)
- Genre-based search and browsing
- Tag-based search (mood, style, era)

**Metadata extracted:**
- Artist: name, bio, image, URL, similar artists, top tracks
- Track: title, artist, album, duration, plays, listeners, tags
- Album: title, artist, release date, image, tracks

---

New class `MusicBrainzClient`:
```python
class MusicBrainzClient:
    """
    MusicBrainz API client for music metadata and deduplication.
    
    Completely free, no authentication required.
    Coverage: 42M+ artists, 100M+ recordings, 50M+ works
    Best for: authoritative metadata, deduplication, ISRC lookup
    """
    
    base_url = "https://musicbrainz.org/ws/2"
    
    def search_artist(self, artist: str, limit: int = 10) -> Dict[str, Any]
    def search_recording(self, title: str, artist: str = None, limit: int = 10) -> Dict[str, Any]
    def get_artist_by_id(self, artist_id: str) -> Dict[str, Any]
    def search_by_isrc(self, isrc: str) -> Dict[str, Any]
    def get_artist_discography(self, artist_id: str) -> Dict[str, Any]
```

**Features:**
- Artist search with complete metadata
- Recording search with ISRC, ISWC codes
- Artist discography browsing
- ISRC/ISWC lookup for deduplication
- Release date, format, barcode information
- Relationship data (artist collaborations, producer info)

**Metadata extracted:**
- Artist: MBID, name, type, country, discography, aliases
- Recording: MBID, ISRC, title, artist, release date, duration, format
- Release: MBID, title, date, country, format, barcode

---

### **Stage 2: Audio Domain System** (1-2 hours)

#### 2.1 Extend Domain Categories

**File:** `app/XNAi_rag_app/library_api_integrations.py`

Add to `DomainCategory` enum:
```python
class DomainCategory(str, Enum):
    # Existing (12)
    CODE = "code"
    SCIENCE = "science"
    # ... etc
    
    # New Audio Domains (3+)
    PODCAST = "podcast"
    MUSIC = "music"
    AUDIOBOOK = "audiobook"
    
    # Audio Sub-domains (10+)
    TECH_PODCAST = "tech_podcast"
    SCIENCE_PODCAST = "science_podcast"
    BUSINESS_PODCAST = "business_podcast"
    ARTS_PODCAST = "arts_podcast"
    
    CLASSICAL_MUSIC = "classical_music"
    JAZZ_MUSIC = "jazz_music"
    ROCK_MUSIC = "rock_music"
    HIP_HOP_MUSIC = "hip_hop_music"
    INDIE_MUSIC = "indie_music"
    ELECTRONIC_MUSIC = "electronic_music"
```

#### 2.2 Audio Metadata Schema

**File:** `app/XNAi_rag_app/library_api_integrations.py`

Extend `LibraryMetadata` dataclass:
```python
@dataclass
class LibraryMetadata:
    # Existing fields
    title: str
    authors: list[str]
    # ... etc
    
    # Audio fields
    is_audio: bool = False  # Flag for audio content
    audio_type: str = None  # "podcast", "music_track", "music_album", "audiobook"
    
    # Podcast fields
    podcast_id: str = None
    podcast_url: str = None
    episode_number: int = None
    episode_duration: int = None  # seconds
    episode_transcript_url: str = None
    season: int = None
    
    # Music fields
    artist: str = None
    album: str = None
    isrc: str = None  # International Standard Recording Code
    iswc: str = None  # International Standard Musical Work Code
    track_number: int = None
    release_date: str = None
    genre: str = None
    mood: list[str] = None
    
    # Audio metadata
    duration: int = None  # seconds
    format: str = None  # "mp3", "aac", "flac", "rss_feed"
    bitrate: int = None  # kbps
    language: str = None
    explicit: bool = False
    
    # Audio cataloging
    audio_hash: str = None  # For deduplication
    cdn_url: str = None  # Streaming URL
    download_available: bool = False
```

---

### **Stage 3: Curator Interface Enhancement** (2-3 hours)

#### 3.1 New Intent Types for Audio

**File:** `app/XNAi_rag_app/library_api_integrations.py`

Add to `NLCuratorInterface`:

```python
# Add to intent types and handlers:

def _audio_search_podcasts(self, parameters: Dict) -> Dict[str, Any]:
    """Handle 'Find podcasts about X' queries."""
    topic = parameters.get('topic')
    category = parameters.get('category')
    limit = parameters.get('limit', 10)
    
    # Use PodcastindexClient
    # 1. Search for podcasts matching topic/category
    # 2. Enrich with domain classification
    # 3. Add to library enrichment engine
    # 4. Return formatted results

def _audio_search_music(self, parameters: Dict) -> Dict[str, Any]:
    """Handle 'Discover [genre] artists/music' queries."""
    genre = parameters.get('genre')
    artist = parameters.get('artist')
    mood = parameters.get('mood')
    limit = parameters.get('limit', 10)
    
    # Use LastfmMusicClient and MusicBrainzClient
    # 1. Search for artists/tracks matching criteria
    # 2. Get similar artists if requested
    # 3. Enrich with metadata from MusicBrainz
    # 4. Deduplicate using ISRC
    # 5. Return formatted results

def _audio_get_recommendations(self, parameters: Dict) -> Dict[str, Any]:
    """Handle 'Recommend [audio type] like X' queries."""
    audio_type = parameters.get('audio_type')  # "podcast" or "music"
    reference = parameters.get('reference')  # podcast name or artist/track
    limit = parameters.get('limit', 10)
    
    # For podcasts: find similar podcasts using Podcastindex
    # For music: find similar artists using Last.fm
    # Return ranked recommendations

def _parse_audio_command(self, user_input: str) -> Dict[str, Any]:
    """Parse audio-specific commands."""
    # Extract: audio_type, action, topic, genre, artist, artist_name, mood, limit
    
    # Examples:
    # "Find podcasts about machine learning" 
    #   → audio_type: podcast, action: search, topic: ML
    # "Show me jazz music recommendations"
    #   → audio_type: music, action: recommend, genre: jazz
    # "Discover artists like Radiohead"
    #   → audio_type: music, action: similar, artist: Radiohead
    # "Get tech podcasts"
    #   → audio_type: podcast, action: category_search, category: tech
```

#### 3.2 New Command Routing

Add audio intent detection:
```python
# Add to _classify_intent():
AUDIO_INTENTS = {
    'find podcasts': 'audio_search_podcasts',
    'discover podcasts': 'audio_search_podcasts',
    'podcast about': 'audio_search_podcasts',
    'find music': 'audio_search_music',
    'discover music': 'audio_search_music',
    'artists like': 'audio_recommend_similar',
    'music like': 'audio_recommend_similar',
    'similar to': 'audio_recommend_similar',
    'music recommendations': 'audio_get_recommendations',
    'podcast recommendations': 'audio_get_recommendations',
}
```

---

### **Stage 4: Crawler Enhancement** (2-3 hours)

#### 4.1 RSS Feed Crawler for Podcasts

**File:** `app/XNAi_rag_app/crawler_curation.py`

Add new function:
```python
def crawl_podcast_feeds(podcast_urls: list[str], 
                       dedup_storage: Redis) -> list[LibraryMetadata]:
    """
    Crawl podcast RSS feeds for new episodes.
    
    Args:
        podcast_urls: List of RSS feed URLs from Podcastindex
        dedup_storage: Redis connection for dedup checking
    
    Returns:
        List of LibraryMetadata for new episodes
    
    Process:
    1. For each podcast feed URL:
       a. Fetch RSS feed using requests
       b. Parse XML (feedparser library)
       c. Extract episode metadata
       d. Create content hash (episode guid + pubdate)
       e. Check Redis for duplicates
       f. Create LibraryMetadata object
       g. Add to library enrichment engine
    2. Return list of new episodes
    """
```

#### 4.2 Music Crawler for New Releases

**File:** `app/XNAi_rag_app/crawler_curation.py`

Add new function:
```python
def crawl_music_new_releases(genres: list[str],
                            days_back: int = 7,
                            dedup_storage: Redis) -> list[LibraryMetadata]:
    """
    Crawl Last.fm for new music releases.
    
    Args:
        genres: Music genres to monitor
        days_back: Only get releases from last N days
        dedup_storage: Redis for deduplication
    
    Returns:
        List of LibraryMetadata for new music
    
    Process:
    1. For each genre:
       a. Query Last.fm API for trending/new releases
       b. For each track:
          - Create content hash (artist + title + ISRC)
          - Check Redis for duplicates
          - Look up ISRC in MusicBrainz for authoritative metadata
          - Create LibraryMetadata object
       c. Add to library enrichment engine
    2. Return list of new tracks
    """
```

---

### **Stage 5: Audio Deduplication System** (1-2 hours)

#### 5.1 Podcast Deduplication

**File:** `app/XNAi_rag_app/library_api_integrations.py`

```python
def deduplicate_podcast(episode_guid: str, 
                       feed_url: str,
                       redis_conn: Redis) -> bool:
    """
    Check if podcast episode is duplicate using episode GUID.
    
    Primary key: episode_guid (universally unique per Podcast RSS spec)
    Secondary: feed_url + title hash
    """
    
    redis_key = f"podcast:episode:{episode_guid}"
    if redis_conn.exists(redis_key):
        return True  # Duplicate
    
    # Store with 30-day TTL (prevents re-crawling)
    redis_conn.setex(redis_key, 86400 * 30, "1")
    return False  # New episode
```

#### 5.2 Music Deduplication

**File:** `app/XNAi_rag_app/library_api_integrations.py`

```python
def deduplicate_music_track(artist: str, 
                           title: str,
                           isrc: str = None,
                           redis_conn: Redis = None) -> bool:
    """
    Check if music track is duplicate using ISRC (primary) or title hash.
    
    Primary: ISRC (International Standard Recording Code - unique worldwide)
    Secondary: SHA256(artist + title)
    Tertiary: Manual check in vector DB
    """
    
    if isrc:
        redis_key = f"music:isrc:{isrc}"
        if redis_conn.exists(redis_key):
            return True  # Duplicate
        redis_conn.setex(redis_key, 86400 * 90, "1")
    
    # Fallback to title hash
    title_hash = hashlib.sha256(f"{artist}|{title}".encode()).hexdigest()
    redis_key = f"music:track:{title_hash}"
    if redis_conn.exists(redis_key):
        return True  # Duplicate
    
    redis_conn.setex(redis_key, 86400 * 90, "1")
    return False  # New track
```

---

### **Stage 6: Integration with Existing System** (1-2 hours)

#### 6.1 Update LibraryEnrichmentEngine

**File:** `app/XNAi_rag_app/library_api_integrations.py`

```python
class LibraryEnrichmentEngine:
    def __init__(self, redis_conn: Redis):
        # Existing clients
        self.open_library = OpenLibraryClient()
        self.internet_archive = InternetArchiveClient()
        # ... existing 8 clients
        
        # NEW: Audio clients
        self.podcastindex = PodcastindexClient()
        self.lastfm = LastfmMusicClient()
        self.musicbrainz = MusicBrainzClient()
    
    def enrich_by_podcast_url(self, rss_url: str) -> LibraryMetadata:
        """Enrich using Podcastindex."""
        podcast_data = self.podcastindex.get_by_url(rss_url)
        # Create LibraryMetadata with audio fields
        # Return enriched data
    
    def enrich_by_artist_name(self, artist: str) -> list[LibraryMetadata]:
        """Enrich using Last.fm and MusicBrainz."""
        # 1. Get artist from Last.fm
        # 2. Get full discography from MusicBrainz
        # 3. Create LibraryMetadata for each album/track
        # 4. Return list
    
    def enrich_by_genre(self, genre: str) -> list[LibraryMetadata]:
        """Get music by genre from Last.fm."""
        # Query trending/popular tracks in genre
        # Enrich with MusicBrainz metadata
        # Deduplicate using ISRC
        # Return list
```

#### 6.2 Update Curation Workflow

**File:** `app/XNAi_rag_app/crawler_curation.py`

```python
# Update domain detection to handle audio:
def detect_domain(metadata: LibraryMetadata) -> DomainCategory:
    """Detect domain for both text and audio content."""
    
    # If audio, use audio domains
    if metadata.is_audio:
        if metadata.audio_type == "podcast":
            # Classify podcast by topic
            # Examples: science_podcast, tech_podcast, business_podcast
        elif metadata.audio_type == "music":
            # Classify by genre
            # Examples: classical_music, jazz_music, indie_music
        elif metadata.audio_type == "audiobook":
            # Treat like book but mark as audio
    
    # Existing logic for books, articles, etc.
```

---

## IMPLEMENTATION SEQUENCE

### **Timeline: 8-12 hours total**

1. **API Client Implementation** (2-3 hours)
   - [ ] Create PodcastindexClient class
   - [ ] Create LastfmMusicClient class
   - [ ] Create MusicBrainzClient class
   - [ ] Test all clients with sample queries

2. **Audio Domain System** (1-2 hours)
   - [ ] Extend DomainCategory enum
   - [ ] Extend LibraryMetadata schema
   - [ ] Update domain detection function

3. **Curator Enhancement** (2-3 hours)
   - [ ] Add audio intent types to NLCuratorInterface
   - [ ] Implement audio command handlers (6 handlers)
   - [ ] Update entity extraction for audio
   - [ ] Test with 10+ audio commands

4. **Crawler Enhancement** (2-3 hours)
   - [ ] Implement podcast RSS crawler
   - [ ] Implement music discovery crawler
   - [ ] Add deduplication system
   - [ ] Integrate with existing crawler

5. **Integration & Testing** (1-2 hours)
   - [ ] Update LibraryEnrichmentEngine
   - [ ] Update curation workflow
   - [ ] Module tests (unit tests on clients)
   - [ ] Chainlit UI tests (audio commands)
   - [ ] Docker integration tests

---

## TEST PLAN

### **Module Testing**
```python
# Test Podcastindex client
curator.process_user_input("Find podcasts about machine learning")
# Expected: 5-10 podcasts with correct metadata

# Test Last.fm client
curator.process_user_input("Discover indie rock artists")
# Expected: 10 artists with similar recommendations

# Test MusicBrainz client
curator.process_user_input("Albums by Radiohead")
# Expected: Complete discography with ISRC codes
```

### **Chainlit UI Testing**
- Test audio search commands in web UI
- Verify markdown formatting for audio metadata
- Test deduplication (search same thing twice)
- Verify curator history with audio items

### **Docker Integration Testing**
- Run full stack with new audio APIs
- Test crawler with podcast feeds
- Verify audio enrichment in background jobs
- Test API performance under load

---

## SUCCESS METRICS

- ✅ All 11 API clients working (8 existing + 3 audio)
- ✅ 14 domain categories (12 existing + 2 audio primary)
- ✅ Audio commands parsing with 85%+ accuracy
- ✅ Podcast search returning 5+ results
- ✅ Music discovery returning 10+ artists/tracks
- ✅ Deduplication working (no duplicates on repeat search)
- ✅ Chainlit UI displaying audio metadata correctly
- ✅ Docker stack handling audio APIs at scale
- ✅ Zero breaking changes to existing code
- ✅ 100% backward compatible with Phase 2

---

## DEPENDENCIES

**New Python packages to add:**
```
feedparser==6.0.10  # RSS feed parsing
requests==2.31.0   # Already in requirements (used for all APIs)
hashlib              # Already built-in (deduplication)
```

**No new database migrations needed:**
- Using existing Redis for deduplication
- Using existing PostgreSQL for metadata (if any)
- LibraryMetadata dataclass handles optional audio fields

---

## RISKS & MITIGATIONS

| Risk | Mitigation |
|------|-----------|
| Podcastindex API reliability | Fallback to iTunes API if needed; cache results |
| Large RSS feeds (1000+ episodes) | Implement pagination; only fetch last 50 episodes |
| Music ISRC lookup slowness | Cache ISRC→MBID mappings in Redis |
| Audio deduplication false positives | Use multi-key approach (ISRC + title hash) |
| Performance with 10K+ audio items | Batch processing; async crawlers with Redis queue |

---

## NEXT STEPS AFTER IMPLEMENTATION

1. Docker integration testing
2. GitHub PR: "Phase 3: Audio Enhancement - Podcasts & Music"
3. Release tag: v0.1.5-audio-enhanced
4. Follow-up enhancements:
   - Audio transcript indexing (for podcasts)
   - Collaborative filtering for recommendations
   - Audio similarity search (advanced RAG)
   - Playlist generation using curator

---

## RESOURCES

**Research completed:** docs/AUDIO_API_RESEARCH_STRATEGY.md  
**Current system:** app/XNAi_rag_app/library_api_integrations.py  
**Crawler module:** app/XNAi_rag_app/crawler_curation.py  
**Chainlit UI:** app/XNAi_rag_app/chainlit_curator_interface.py

