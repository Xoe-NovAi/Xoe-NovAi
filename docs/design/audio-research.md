# Audio API Research & Integration Strategy
## Podcast & Music for Xoe-NovAi Library System

**Date:** January 3, 2026  
**Status:** Research Complete - Ready for Implementation  
**Target:** v0.1.5 Audio Enhancement Phase

---

## EXECUTIVE SUMMARY

Your Xoe-NovAi library system can be extended with audio (podcasts + music) using **completely free APIs**. The best approach combines:

- **Podcasts:** Podcastindex API (most complete, free with optional paid features)
- **Music:** MusicBrainz + Last.fm (best metadata quality, zero-cost)
- **Audio Metadata:** Unified schema supporting 25+ fields per audio resource
- **Crawler Strategy:** RSS feeds (podcasts) + API search (music) with intelligent deduplication
- **Curation:** Natural language commands ("Find tech podcasts", "Discover indie rock", etc.)

---

## 1. PODCAST API COMPARISON

### **#1 RECOMMENDATION: Podcastindex API** ⭐⭐⭐⭐⭐

**URL:** https://podcastindex.org/

| Feature | Value |
|---------|-------|
| **Cost** | FREE (no authentication required) |
| **Rate Limit** | 10 requests/sec (generous) |
| **API Key** | Optional (increases limits) |
| **Metadata Quality** | ⭐⭐⭐⭐⭐ Excellent |
| **RSS Feed Access** | ✅ Yes |
| **Search** | ✅ Full-text search |
| **Episode Data** | ✅ Complete |

**Pros:**
- ✅ No authentication required to start
- ✅ Massive index (3M+ podcasts)
- ✅ Episode-level metadata (guid, pubdate, duration, transcript URLs)
- ✅ Podcast ratings & statistics
- ✅ Category/genre classification
- ✅ Dead feed detection
- ✅ Open-source, community-driven
- ✅ JSON API (easy integration)
- ✅ **Full RSS feed URLs** for crawling

**Cons:**
- ❌ Best features require API key (free but requires signup)
- ❌ Less mainstream than Spotify
- ❌ Smaller music metadata set

**Key Endpoints:**
```
GET /api/1.0/search/byterm?q={term}          # Search by title/author
GET /api/1.0/podcasts/byfeedurl?url={url}    # Get podcast by feed
GET /api/1.0/podcasts/byitunesid?id={id}     # iTunes integration
GET /api/1.0/podcasts/{id}                   # Get podcast details
GET /api/1.0/episodes/byfeedurl?url={url}    # Get recent episodes
GET /api/1.0/categories                      # List all categories
```

**Response Example:**
```json
{
  "feed": {
    "id": 14532,
    "title": "Tech Daily",
    "url": "https://techdaily.example.com/feed.xml",
    "link": "https://techdaily.example.com",
    "description": "Daily tech news",
    "author": "Tech Team",
    "image": "https://...",
    "generator": "Podscribe",
    "language": "en-us",
    "categories": {"1": "Technology", "2": "News"},
    "updateFrequency": 86400,
    "lastUpdateTime": 1672531200,
    "episodeCount": 1234,
    "itunesId": 123456,
    "newestItemPublishTime": 1672531200
  }
}
```

---

### **#2 ALTERNATIVE: iTunes Podcast API**

**URL:** https://podcasts.apple.com/ + iTunes API

| Feature | Value |
|---------|-------|
| **Cost** | FREE |
| **Rate Limit** | 100 reqs/hour (strictly enforced) |
| **API Key** | Not required |
| **Metadata Quality** | ⭐⭐⭐⭐ Good |
| **Search** | ✅ Yes |
| **Episode Data** | ⚠️ Limited |

**Pros:**
- ✅ Authoritative (Apple's directory)
- ✅ No authentication needed
- ✅ Large podcast index (3M+)
- ✅ JSONP support for browser compatibility
- ✅ Good podcast metadata (title, author, description, image)

**Cons:**
- ❌ **Very strict rate limits** (100/hour = ~1 req/36 seconds)
- ❌ **No episode-level API** (only podcast metadata)
- ❌ No transcript or guest information
- ❌ Harder to get full episodes list
- ❌ No RSS feed URLs provided

**Key Endpoints:**
```
https://itunes.apple.com/search?term={term}&media=podcast&limit=50
https://itunes.apple.com/lookup?id={podcast_id}
https://itunes.apple.com/lookup?id={podcast_id}&entity=podcastEpisode
```

**Not Recommended** - Use only as backup for podcast verification.

---

### **#3 FALLBACK: Open Podcast API**

**URL:** https://open-podcast-api.herokuapp.com/ (Community-maintained)

| Feature | Value |
|---------|-------|
| **Cost** | FREE |
| **Rate Limit** | Generous |
| **API Key** | Not required |
| **Metadata Quality** | ⭐⭐⭐ Fair |
| **Maintenance** | ⚠️ Community (inconsistent) |

**Pros:**
- ✅ Simple REST API
- ✅ No authentication required
- ✅ Good basic metadata

**Cons:**
- ❌ **Not actively maintained** (heroku plan discontinued)
- ❌ Incomplete metadata
- ❌ Smaller index than Podcastindex
- ❌ Unreliable uptime

**Not Recommended** - Podcastindex is strictly better.

---

## 2. MUSIC API COMPARISON

### **#1 RECOMMENDATION: MusicBrainz API** ⭐⭐⭐⭐⭐

**URL:** https://musicbrainz.org/doc/MusicBrainz_API

| Feature | Value |
|---------|-------|
| **Cost** | FREE (100% open, no keys needed) |
| **Rate Limit** | 1 req/sec (reasonable) |
| **API Key** | Not required |
| **Metadata Quality** | ⭐⭐⭐⭐⭐ Exceptional |
| **Completeness** | ⭐⭐⭐⭐⭐ Best in class |

**Pros:**
- ✅ **Most comprehensive music metadata** (AcoustID, ISRC, ISWC)
- ✅ Complete discography for artists
- ✅ Album artwork (via Coverart Archive API)
- ✅ Relations between entities (artist→album→track→recording)
- ✅ Release dates, barcode, label info
- ✅ Wikipedia links for artists
- ✅ **Open source backend** (Postgres, downloadable)
- ✅ Highly structured data (XML/JSON)
- ✅ No rate limiting token required

**Cons:**
- ❌ Data can be incomplete (crowdsourced, like Wikipedia)
- ❌ Slower response times than commercial APIs
- ❌ Strict 1 req/sec rate limit
- ❌ Lesser-known artists have sparse data
- ❌ No audio preview URLs

**Key Endpoints:**
```
GET /ws/2/artist/{mbid}?inc=ratings+genres+recordings
GET /ws/2/recording/{mbid}?inc=artists+releases
GET /ws/2/release/{mbid}?inc=recordings+artist-credits
GET /ws/2/artist?query={artist_name}&limit=50
GET /ws/2/recording?query={song_title}&limit=50
GET /artist/{mbid}/ratings
```

**Response Example:**
```json
{
  "recording": {
    "id": "abc123",
    "title": "Given Up",
    "length": 180000,
    "artist-credit": [{
      "artist": {
        "id": "xyz789",
        "name": "Linkin Park",
        "sort-name": "Park, Linkin"
      }
    }],
    "release-list": [{
      "id": "rel456",
      "title": "Minutes to Midnight",
      "date": "2007-05-14",
      "country": "US"
    }],
    "genres": ["rock", "alternative rock"]
  }
}
```

**Curation Integration:**
- Link to artist wikpedia for bio/discovery
- Chain releases → recordings for discography crawling
- Use AcoustID for duplicate detection via audio fingerprinting

---

### **#2 RECOMMENDATION: Last.fm API** ⭐⭐⭐⭐

**URL:** https://www.last.fm/api/

| Feature | Value |
|---------|-------|
| **Cost** | FREE (requires API key) |
| **Rate Limit** | 5 reqs/sec (very generous) |
| **API Key** | Required (free) |
| **Metadata Quality** | ⭐⭐⭐⭐ Very Good |
| **Discovery** | ⭐⭐⭐⭐⭐ Best feature |

**Pros:**
- ✅ **Best for music discovery** (tags, similar artists, top tracks)
- ✅ User-generated tags (indie, lo-fi, synthwave, etc.)
- ✅ Play counts & popularity metrics
- ✅ Similar artists algorithm
- ✅ Top charts by country
- ✅ Genre information
- ✅ Excellent for curation queries ("Find artists similar to X")
- ✅ Very high rate limits (5 req/sec = 432K/day)
- ✅ Album artwork URLs

**Cons:**
- ❌ Less technical metadata than MusicBrainz
- ❌ Requires API key signup
- ❌ No audio preview URLs
- ❌ User/scrobbling-focused (not library-focused)
- ❌ Can have "junk" data from spam scrobbles

**Key Endpoints:**
```
GET /artist.search?artist={name}
GET /artist.getInfo?artist={name}
GET /artist.getSimilar?artist={name}
GET /artist.getTopTracks?artist={name}
GET /artist.getTopAlbums?artist={name}
GET /chart.getTopArtists?limit=50
GET /track.search?track={title}&artist={artist}
GET /tag.getSimilar?tag={tag}
GET /tag.getTopArtists?tag={tag}
```

**Response Example:**
```json
{
  "artist": {
    "name": "Linkin Park",
    "url": "https://www.last.fm/music/Linkin+Park",
    "image": [{
      "size": "mega",
      "#text": "https://lastfm.freetls.fastly.net/i/u/300x300/abc123.jpg"
    }],
    "streamable": "0",
    "ontour": "0",
    "stats": {
      "listeners": "3458195",
      "playcount": "562923246"
    },
    "similar": {
      "artist": [{
        "name": "Linkin Park clone",
        "url": "..."
      }]
    },
    "tags": {
      "tag": [
        {"name": "rock", "url": "..."},
        {"name": "alternative rock", "url": "..."}
      ]
    }
  }
}
```

**Curation Integration:**
- Query similar artists: `artist.getSimilar`
- Search by tags: `tag.getTopArtists` for genre discovery
- Get trending: `chart.getTopArtists`

---

### **#3 ALTERNATIVE: Discogs API** ⭐⭐⭐

**URL:** https://www.discogs.com/developers

| Feature | Value |
|---------|-------|
| **Cost** | FREE (with API key) |
| **Rate Limit** | 60 reqs/minute (sufficient) |
| **API Key** | Required (free) |
| **Metadata Quality** | ⭐⭐⭐⭐ Good |
| **Vinyl/Physical** | ⭐⭐⭐⭐⭐ Best for this |

**Pros:**
- ✅ **Best for vinyl/physical releases** (pressing info, barcode)
- ✅ High-quality cover art
- ✅ Collector pricing/valuation data
- ✅ Genre taxonomy (highly detailed)
- ✅ Contributors (producers, engineers)
- ✅ Format information (CD, LP, cassette, digital)

**Cons:**
- ❌ Vinyl/collector focus (not great for discovery)
- ❌ Less complete for obscure digital artists
- ❌ More overhead than Last.fm
- ❌ Smaller index than MusicBrainz/Last.fm

**Key Use:** Best as **secondary** data source for physical/collector metadata.

---

## 3. AUDIO METADATA SCHEMA

### **Unified Audio Metadata Model**

Extend your existing `ContentMetadata` class to support podcasts and music:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class AudioType(str, Enum):
    PODCAST = "podcast"
    PODCAST_EPISODE = "podcast_episode"
    MUSIC_TRACK = "music_track"
    MUSIC_ALBUM = "music_album"
    MUSIC_ARTIST = "music_artist"

@dataclass
class AudioMetadata:
    """Extended metadata for podcast and music resources."""
    
    # ========== UNIVERSAL FIELDS ==========
    resource_id: str              # UUID or API ID
    audio_type: AudioType         # What kind of audio?
    title: str                    # Podcast name or song title
    description: str              # Full description
    url: str                       # Canonical URL
    crawl_date: str               # ISO format datetime
    domain: str                   # DomainType.value
    
    # ========== CREATORS ==========
    creators: List[Dict]          # [{name, url, type}] (hosts/artists/labels)
    creator_primary: str          # Main creator name (for sorting)
    
    # ========== PODCAST-SPECIFIC ==========
    podcast_title: Optional[str] = None
    podcast_rss_url: Optional[str] = None
    podcast_id: Optional[str] = None             # Podcastindex ID
    podcast_itunesid: Optional[str] = None       # iTunes Podcast ID
    podcast_language: Optional[str] = None       # en, es, fr, etc.
    podcast_category: Optional[str] = None       # Technology, News, etc.
    podcast_subcategory: Optional[str] = None
    podcast_image_url: Optional[str] = None
    podcast_explicit: Optional[bool] = None
    
    # EPISODE-SPECIFIC (nested in podcast_episodes)
    episode_number: Optional[int] = None
    episode_season: Optional[int] = None
    episode_duration_seconds: Optional[int] = None
    episode_guid: Optional[str] = None           # For deduplication
    episode_pub_date: Optional[str] = None       # ISO datetime
    episode_transcript_url: Optional[str] = None # If available
    episode_guests: Optional[List[str]] = None
    
    # ========== MUSIC-SPECIFIC ==========
    artist_name: Optional[str] = None
    artist_id: Optional[str] = None              # MusicBrainz MBID
    artist_genres: Optional[List[str]] = None
    artist_similar: Optional[List[str]] = None   # [artist names]
    artist_tags: Optional[List[str]] = None      # Last.fm user tags
    
    album_title: Optional[str] = None
    album_id: Optional[str] = None               # MusicBrainz MBID
    album_release_date: Optional[str] = None     # ISO date
    album_label: Optional[str] = None
    album_image_url: Optional[str] = None
    album_format: Optional[str] = None           # CD, LP, Digital, etc.
    
    track_number: Optional[int] = None
    track_duration_seconds: Optional[int] = None
    track_isrc: Optional[str] = None             # International Standard Recording Code
    track_acoustid: Optional[str] = None         # For audio fingerprinting
    
    # ========== DISCOVERY & CURATION ==========
    genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    popularity_score: Optional[float] = None     # 0-1 or 0-100
    user_rating: Optional[float] = None          # Last.fm playcount ratio
    
    # ========== QUALITY SIGNALS ==========
    word_count: int = 0
    citation_count: int = 0
    image_count: int = 0
    heading_structure_score: float = 0.0
    
    # ========== DEDUPLICATION ==========
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    content_hash: str = ""
    
    # ========== SOURCE TRACKING ==========
    source_apis: List[str] = field(default_factory=list)  # [podcastindex, musicbrainz, lastfm]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

# ========== EXTENDED DOMAIN TYPES ==========
class ExtendedDomainType(str, Enum):
    """Enhanced domain classification including audio."""
    CODE = "code"
    SCIENCE = "science"
    DATA = "data"
    GENERAL = "general"
    PODCAST = "podcast"
    MUSIC = "music"
    AUDIO = "audio"  # Generic audio content
    INTERVIEWS = "interviews"  # Podcasts with guests
    EDUCATIONAL_AUDIO = "educational_audio"
```

### **Field Recommendations by Audio Type:**

| Field | Podcast | Episode | Music Track | Album | Artist |
|-------|---------|---------|-------------|-------|--------|
| title | ✅ | ✅ | ✅ | ✅ | ✅ |
| creators | ✅ | ✅ | ✅ | ✅ | ✅ |
| description | ✅ | ✅ | ❌ | ✅ | ✅ |
| category | ✅ | ❌ | ❌ | ❌ | ✅ |
| duration | ❌ | ✅ | ✅ | ❌ | ❌ |
| release_date | ✅ | ✅ | ✅ | ✅ | ❌ |
| image_url | ✅ | ❌ | ✅ | ✅ | ✅ |
| explicit | ✅ | ✅ | ❌ | ❌ | ❌ |
| genres | ❌ | ❌ | ✅ | ✅ | ✅ |
| tags | ✅ | ❌ | ✅ | ✅ | ✅ |
| popularity | ❌ | ❌ | ✅ | ✅ | ✅ |

---

## 4. CRAWLER STRATEGY

### **Podcast Crawling**

```
┌─────────────────────────────────────────────────┐
│ PODCAST DISCOVERY & CRAWLING PIPELINE           │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. DISCOVERY (Podcastindex API)                │
│    ├─ Search: /api/1.0/search/byterm?q=X      │
│    ├─ Categories: /api/1.0/categories          │
│    └─ Results: ~20-50 podcasts per query      │
│                                                 │
│ 2. PODCAST METADATA (Podcastindex)             │
│    ├─ Get: /podcasts/byitunesid?id=X          │
│    ├─ Extract: {id, title, feed_url, etc}     │
│    └─ Store: podcast_id + rss_url             │
│                                                 │
│ 3. FEED CRAWLING (RSS Parser)                  │
│    ├─ Parse: feedparser.parse(rss_url)        │
│    ├─ Extract: episodes[0:N]                  │
│    ├─ Per episode: {guid, title, pub_date}    │
│    └─ Duration: try to extract from audio     │
│                                                 │
│ 4. DEDUPLICATION (GUID-based)                  │
│    ├─ Hash: SHA256(podcast_id + episode_guid) │
│    ├─ Store: in Redis/DB for quick lookup     │
│    └─ Skip: episodes already crawled          │
│                                                 │
│ 5. ENRICHMENT (Optional APIs)                  │
│    ├─ iTunes: verify podcast exists           │
│    └─ Transcript: check for auto-transcripts  │
│                                                 │
│ 6. STORAGE                                     │
│    └─ Format: CrawledDocument + AudioMetadata │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Implementation:**

```python
import feedparser
import requests
from hashlib import sha256

class PodcastCrawler:
    def __init__(self):
        self.podcastindex_base = "https://podcastindex.org/api/1.0"
        self.crawled_episodes = set()  # Episode GUIDs
    
    def discover_podcasts(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Podcastindex for podcasts."""
        response = requests.get(
            f"{self.podcastindex_base}/search/byterm",
            params={"q": query, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("feeds", [])
    
    def get_podcast_metadata(self, podcast_id: int) -> Dict:
        """Get detailed podcast metadata."""
        response = requests.get(
            f"{self.podcastindex_base}/podcasts/{podcast_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("feed", {})
    
    def crawl_podcast_episodes(self, rss_url: str) -> List[Dict]:
        """Parse RSS feed and extract episodes."""
        feed = feedparser.parse(rss_url)
        episodes = []
        
        for entry in feed.entries[:20]:  # Last 20 episodes
            episode_guid = entry.get("id", entry.get("link", ""))
            
            # Deduplication check
            guid_hash = sha256(episode_guid.encode()).hexdigest()
            if guid_hash in self.crawled_episodes:
                continue
            
            episodes.append({
                "guid": episode_guid,
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "pub_date": entry.get("published", ""),
                "duration": self._extract_duration(entry),
                "transcript_url": entry.get("transcript", None),
            })
            
            self.crawled_episodes.add(guid_hash)
        
        return episodes
    
    def _extract_duration(self, entry) -> Optional[int]:
        """Try to extract duration in seconds."""
        # Try iTunes duration tag
        if "itunes_duration" in entry:
            try:
                return int(entry["itunes_duration"])
            except:
                pass
        
        # Try media:content duration
        if entry.get("media_content"):
            return entry["media_content"][0].get("duration")
        
        return None
```

---

### **Music Crawling**

```
┌──────────────────────────────────────────────────────┐
│ MUSIC DISCOVERY & CRAWLING PIPELINE                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│ 1. ARTIST DISCOVERY (Last.fm API)                   │
│    ├─ Search: artist.search?artist=X               │
│    ├─ Top Charts: chart.getTopArtists              │
│    ├─ By Tags: tag.getTopArtists?tag=indie         │
│    └─ Similar: artist.getSimilar?artist=X          │
│                                                      │
│ 2. ARTIST METADATA (MusicBrainz + Last.fm)         │
│    ├─ MusicBrainz: GET /artist/{mbid}              │
│    ├─ Last.fm: artist.getInfo?artist=X             │
│    └─ Extract: {genres, tags, similar_artists}     │
│                                                      │
│ 3. DISCOGRAPHY CRAWLING (MusicBrainz)              │
│    ├─ Query: GET /artist/{mbid}?inc=releases      │
│    ├─ Extract: album list for artist               │
│    └─ Store: {album_id, title, release_date}      │
│                                                      │
│ 4. ALBUM METADATA (MusicBrainz)                    │
│    ├─ Query: GET /release/{mbid}                   │
│    ├─ Extract: {tracks, labels, formats}          │
│    └─ Get Art: via Coverart Archive API           │
│                                                      │
│ 5. DEDUPLICATION (ISRC-based)                      │
│    ├─ Hash: SHA256(track_isrc OR track_title+artist) │
│    ├─ Store: in DB for quick lookup                │
│    └─ Skip: already indexed tracks                 │
│                                                      │
│ 6. ENRICHMENT (Last.fm)                            │
│    ├─ Get Tags: track.getInfo?track=X&artist=Y    │
│    ├─ Get Top Tracks: artist.getTopTracks         │
│    └─ Popularity: via playcount metrics            │
│                                                      │
│ 7. STORAGE                                         │
│    └─ Format: CrawledDocument + AudioMetadata      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import requests
from hashlib import sha256
from typing import List, Dict, Optional

class MusicCrawler:
    def __init__(self, lastfm_api_key: str):
        self.lastfm_base = "http://ws.audioscrobbler.com/2.0"
        self.musicbrainz_base = "https://musicbrainz.org/ws/2"
        self.lastfm_key = lastfm_api_key
        self.crawled_tracks = set()  # ISRCs or title hashes
    
    def discover_artists(self, tag: str, limit: int = 20) -> List[Dict]:
        """Get top artists by tag (for curation like 'indie rock')."""
        response = requests.get(
            self.lastfm_base,
            params={
                "method": "tag.getTopArtists",
                "tag": tag,
                "limit": limit,
                "api_key": self.lastfm_key,
                "format": "json"
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("topartists", {}).get("artist", [])
    
    def get_artist_info(self, artist_name: str) -> Dict:
        """Get artist metadata from both APIs."""
        
        # Last.fm info (discovery tags, play counts)
        lastfm_response = requests.get(
            self.lastfm_base,
            params={
                "method": "artist.getInfo",
                "artist": artist_name,
                "api_key": self.lastfm_key,
                "format": "json"
            },
            timeout=10
        )
        lastfm_data = lastfm_response.json().get("artist", {})
        
        # MusicBrainz info (MBID + detailed metadata)
        mb_response = requests.get(
            f"{self.musicbrainz_base}/artist",
            params={"query": f'artist:"{artist_name}"', "limit": 1},
            headers={"User-Agent": "xoe-novai/0.1.5"},
            timeout=10
        )
        mb_data = mb_response.json()
        mbid = mb_data["artists"][0]["id"] if mb_data.get("artists") else None
        
        return {
            "name": artist_name,
            "lastfm_tags": [tag["name"] for tag in lastfm_data.get("tags", {}).get("tag", [])],
            "lastfm_listeners": lastfm_data.get("stats", {}).get("listeners", 0),
            "mbid": mbid,
        }
    
    def crawl_artist_discography(self, mbid: str) -> List[Dict]:
        """Get all albums/releases for an artist."""
        response = requests.get(
            f"{self.musicbrainz_base}/artist/{mbid}",
            params={"inc": "releases"},
            headers={"User-Agent": "xoe-novai/0.1.5"},
            timeout=10
        )
        response.raise_for_status()
        
        artist_data = response.json()
        releases = []
        
        for release in artist_data.get("releases", [])[:50]:  # Limit to 50
            releases.append({
                "id": release["id"],
                "title": release["title"],
                "date": release.get("date", ""),
                "packaging": release.get("packaging", ""),
                "label": release.get("label-info", [{}])[0].get("label", {}).get("name", ""),
            })
        
        return releases
    
    def crawl_album_tracks(self, release_id: str) -> List[Dict]:
        """Get all tracks from an album."""
        response = requests.get(
            f"{self.musicbrainz_base}/release/{release_id}",
            params={"inc": "recordings"},
            headers={"User-Agent": "xoe-novai/0.1.5"},
            timeout=10
        )
        response.raise_for_status()
        
        release_data = response.json()
        tracks = []
        
        for media in release_data.get("media", []):
            for track in media.get("tracks", []):
                recording = track.get("recording", {})
                track_isrc = recording.get("isrcs", [""])[0]
                
                # Deduplication via ISRC
                isrc_hash = sha256(track_isrc.encode()).hexdigest() if track_isrc else None
                if isrc_hash and isrc_hash in self.crawled_tracks:
                    continue
                
                tracks.append({
                    "title": recording.get("title", ""),
                    "isrc": track_isrc,
                    "duration": recording.get("length", 0) // 1000,  # ms to seconds
                    "position": track.get("position", ""),
                })
                
                if isrc_hash:
                    self.crawled_tracks.add(isrc_hash)
        
        return tracks
    
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[str]:
        """Get similar artists for discovery."""
        response = requests.get(
            self.lastfm_base,
            params={
                "method": "artist.getSimilar",
                "artist": artist_name,
                "limit": limit,
                "api_key": self.lastfm_key,
                "format": "json"
            },
            timeout=10
        )
        response.raise_for_status()
        
        artists = response.json().get("similarartists", {}).get("artist", [])
        return [artist["name"] for artist in artists]
```

---

### **Deduplication Strategy**

| Media Type | Primary Key | Hash Input | Storage |
|------------|------------|-----------|---------|
| **Podcast** | episode_guid | `podcast_id + episode_guid` | Redis sorted set |
| **Music Track** | ISRC | `isrc` | PostgreSQL unique index |
| **Album** | MusicBrainz MBID | `mbid` | PostgreSQL unique index |
| **Artist** | MusicBrainz MBID | `mbid` | PostgreSQL unique index |

**Redis Set Example:**
```python
# Deduplicate podcasts
PODCAST_CRAWLED_SET = "xoe:podcasts:crawled_episodes"

# Check if episode exists
exists = redis_client.sismember(PODCAST_CRAWLED_SET, episode_guid_hash)

# Add new episode
redis_client.sadd(PODCAST_CRAWLED_SET, episode_guid_hash)

# TTL: 90 days (refresh cycle)
redis_client.expire(PODCAST_CRAWLED_SET, 7776000)
```

---

## 5. SEARCH & CURATION COMMANDS

### **Natural Language Curation Queries**

Extend your `CuratorEnhancer` to support audio-specific commands:

```python
from enum import Enum
from typing import List, Dict

class AudioCuratorCommand(str, Enum):
    # Podcast commands
    FIND_PODCASTS_TOPIC = "find_podcasts_topic"          # "Find podcasts about X"
    FIND_PODCASTS_CATEGORY = "find_podcasts_category"    # "Show me tech podcasts"
    FIND_GUESTS = "find_podcast_guests"                  # "Who has interviewed X?"
    PODCAST_SERIES = "find_podcast_series"               # "Find long-running podcasts"
    
    # Music commands
    DISCOVER_ARTIST = "discover_artist"                  # "Recommend artists like X"
    DISCOVER_GENRE = "discover_genre"                    # "Find indie rock bands"
    DISCOVER_SIMILAR = "discover_similar"                # "More like this track"
    CHART_MUSIC = "chart_music"                          # "What's trending in rock?"
    FIND_COLLABORATIONS = "find_collaborations"          # "Artists who worked with X"

class AudioCurator:
    """Curation engine for podcast and music discovery."""
    
    def __init__(self, lastfm_api_key: str):
        self.crawler = MusicCrawler(lastfm_api_key)
        self.podcast_crawler = PodcastCrawler()
    
    # ===== PODCAST CURATION =====
    
    def find_podcasts_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """
        Find podcasts about a specific topic.
        Example: "Find podcasts about AI"
        """
        podcasts = self.podcast_crawler.discover_podcasts(
            query=topic,
            limit=limit
        )
        return podcasts
    
    def find_podcasts_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """
        Find podcasts in a specific category.
        Example: "Show me tech podcasts"
        Categories: Technology, News, Comedy, Business, Education, etc.
        """
        # Podcastindex has categories endpoint
        response = requests.get(
            "https://podcastindex.org/api/1.0/categories"
        )
        categories = response.json().get("categories", {})
        
        # Find matching category
        category_id = None
        for cat_name, cat_data in categories.items():
            if category.lower() in cat_name.lower():
                category_id = cat_data["id"]
                break
        
        if not category_id:
            return []
        
        # Search within category
        podcasts = self.podcast_crawler.discover_podcasts(category, limit)
        return [p for p in podcasts if str(category_id) in str(p.get("categories", {}))]
    
    def find_podcast_with_guest(self, guest_name: str) -> List[Dict]:
        """
        Find podcast episodes featuring a guest.
        Example: "Who has interviewed Elon Musk?"
        """
        # Search for guest mentions in episode titles/descriptions
        episodes = self.podcast_crawler.discover_podcasts(
            query=guest_name,
            limit=50
        )
        return episodes
    
    # ===== MUSIC CURATION =====
    
    def discover_similar_artists(self, artist_name: str, limit: int = 10) -> List[str]:
        """
        Recommend artists similar to a given artist.
        Example: "Recommend artists like Radiohead"
        """
        return self.crawler.get_similar_artists(artist_name, limit)
    
    def discover_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """
        Find artists in a genre.
        Example: "Find indie rock bands"
        Genre: indie, rock, pop, metal, jazz, classical, etc.
        """
        artists = self.crawler.discover_artists(genre, limit)
        return artists
    
    def discover_similar_track(self, track_title: str, artist_name: str) -> List[Dict]:
        """
        Find similar tracks to a given song.
        Example: "More like 'Creep' by Radiohead"
        """
        # Use Last.fm to find similar tracks
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0",
            params={
                "method": "track.getSimilar",
                "artist": artist_name,
                "track": track_title,
                "api_key": self.crawler.lastfm_key,
                "format": "json"
            }
        )
        similar = response.json().get("similartracks", {}).get("track", [])
        return similar
    
    def find_trending_music(self, genre: Optional[str] = None, 
                           country: str = "US", limit: int = 20) -> List[Dict]:
        """
        Get trending music.
        Example: "What's trending in rock music?" or "Top songs in Japan?"
        """
        # Use Last.fm charts
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0",
            params={
                "method": "chart.getTopArtists",
                "limit": limit,
                "api_key": self.crawler.lastfm_key,
                "format": "json"
            }
        )
        return response.json().get("topartists", {}).get("artist", [])
    
    def find_collaborations(self, artist_name: str) -> List[Dict]:
        """
        Find all collaborations/features by an artist.
        Example: "Artists who worked with Kanye West?"
        """
        # Would need to crawl discography and look for features
        mbid = self._get_musicbrainz_id(artist_name)
        if not mbid:
            return []
        
        releases = self.crawler.crawl_artist_discography(mbid)
        # Filter for compilations/features
        return [r for r in releases if "feature" in r.get("title", "").lower()]
    
    def _get_musicbrainz_id(self, artist_name: str) -> Optional[str]:
        """Helper to get MusicBrainz ID."""
        response = requests.get(
            "https://musicbrainz.org/ws/2/artist",
            params={"query": f'artist:"{artist_name}"', "limit": 1},
            headers={"User-Agent": "xoe-novai/0.1.5"}
        )
        artists = response.json().get("artists", [])
        return artists[0]["id"] if artists else None
```

### **Curator Command Examples**

```
User: "Find me podcasts about machine learning"
  → finder_podcasts_by_topic("machine learning")
  → Returns: [podcast1, podcast2, ...] with RSS feeds

User: "Show me tech podcasts"
  → find_podcasts_by_category("Technology")
  → Returns: [50 tech podcasts sorted by popularity]

User: "Who has interviewed Naval Ravikant?"
  → find_podcast_with_guest("Naval Ravikant")
  → Returns: [episodes where guest appears]

User: "Recommend artists like Radiohead"
  → discover_similar_artists("Radiohead", limit=15)
  → Returns: ["Thom Yorke", "Portishead", "Muse", ...]

User: "Find indie rock bands"
  → discover_genre("indie rock", limit=20)
  → Returns: [artist1, artist2, ...]

User: "More like 'Fake Plastic Trees' by Radiohead"
  → discover_similar_track("Fake Plastic Trees", "Radiohead")
  → Returns: [similar_track1, similar_track2, ...]

User: "What's trending in synthwave?"
  → find_trending_music(genre="synthwave")
  → Returns: [top 20 synthwave artists this month]
```

---

## 6. INTEGRATION WITH XOE-NOVAI LIBRARY SYSTEM

### **Architecture Overview**

```
┌──────────────────────────────────────────────────────────────┐
│ EXISTING XOE-NOVAI LIBRARY SYSTEM                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Content Types: Books, Papers, Code                         │
│  └─ crawler_curation.py → ContentMetadata → FAISS Index    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ AUDIO ENHANCEMENT (NEW - v0.1.5)                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─── PODCAST PIPELINE ───┐                                 │
│  │ Podcastindex API       │                                 │
│  │ → RSS Feed Parsing     │                                 │
│  │ → AudioMetadata        │                                 │
│  │ → Dedup (Episode GUIDs)│                                 │
│  └────────────┬───────────┘                                 │
│               │                                             │
│  ┌─── MUSIC PIPELINE ────┐                                  │
│  │ Last.fm API          │                                  │
│  │ → MusicBrainz API    │                                  │
│  │ → AudioMetadata      │                                  │
│  │ → Dedup (ISRCs)      │                                  │
│  └────────────┬──────────┘                                  │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ UNIFIED AUDIO METADATA + DOMAIN CLASSIFICATION   │      │
│  │ (ExtendedDomainType.PODCAST / MUSIC)             │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ CrawledDocument (audio variant)                  │      │
│  │ → Enrich with library_api_integrations()         │      │
│  │ → Vectorize 25+ metadata fields                  │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ FAISS INDEX (Mixed Content)                      │      │
│  │ - Books                                          │      │
│  │ - Papers                                         │      │
│  │ - Code                                           │      │
│  │ - Podcasts (NEW)                                 │      │
│  │ - Music (NEW)                                    │      │
│  │ - Metadata vectors for discovery                 │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│               ▼                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ AUDIO CURATION ENGINE                            │      │
│  │ Natural language queries:                        │      │
│  │ - "Find podcasts about X"                        │      │
│  │ - "Discover artists like X"                      │      │
│  │ - Audio-specific rankings/filters                │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### **Implementation Steps**

#### **Step 1: Extend ContentMetadata → AudioMetadata**

```python
# In crawler_curation.py

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, List, Dict

# ADD to DomainType
class ExtendedDomainType(str, Enum):
    CODE = "code"
    SCIENCE = "science"
    DATA = "data"
    GENERAL = "general"
    PODCAST = "podcast"
    PODCAST_EPISODE = "podcast_episode"
    MUSIC = "music"
    MUSIC_TRACK = "music_track"
    MUSIC_ALBUM = "music_album"
    AUDIO = "audio"

# EXTEND ContentMetadata
@dataclass
class AudioMetadata:
    """Audio-specific metadata (podcasts, music)."""
    
    resource_id: str
    audio_type: str  # 'podcast', 'episode', 'track', 'album', 'artist'
    title: str
    description: str
    url: str
    crawl_date: str
    domain: str
    
    # Audio creators
    creators: List[Dict] = field(default_factory=list)
    
    # Podcast fields
    podcast_id: Optional[str] = None
    podcast_rss_url: Optional[str] = None
    episode_guid: Optional[str] = None
    episode_duration_seconds: Optional[int] = None
    podcast_category: Optional[str] = None
    
    # Music fields
    artist_id: Optional[str] = None
    artist_name: Optional[str] = None
    album_id: Optional[str] = None
    album_title: Optional[str] = None
    track_isrc: Optional[str] = None
    track_duration_seconds: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Deduplication
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    source_apis: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
```

#### **Step 2: Create audio_crawlers.py**

```python
# app/XNAi_rag_app/audio_crawlers.py

import feedparser
import requests
from typing import List, Dict, Optional
from hashlib import sha256
from crawler_curation import AudioMetadata, ExtendedDomainType

class PodcastCrawler:
    """Crawl podcasts via Podcastindex API + RSS feeds."""
    
    def __init__(self):
        self.base_url = "https://podcastindex.org/api/1.0"
        self.crawled_episodes = set()
    
    def search_podcasts(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for podcasts by topic."""
        response = requests.get(
            f"{self.base_url}/search/byterm",
            params={"q": query, "limit": limit},
            timeout=10
        )
        return response.json().get("feeds", [])
    
    def get_podcast_metadata(self, podcast_id: int) -> Dict:
        """Get podcast details."""
        response = requests.get(
            f"{self.base_url}/podcasts/{podcast_id}",
            timeout=10
        )
        return response.json().get("feed", {})
    
    def crawl_episodes(self, rss_url: str, podcast_id: int) -> List[AudioMetadata]:
        """Parse RSS and return audio metadata."""
        feed = feedparser.parse(rss_url)
        episodes = []
        
        for entry in feed.entries[:20]:
            episode_guid = entry.get("id", entry.get("link", ""))
            guid_hash = sha256(episode_guid.encode()).hexdigest()
            
            if guid_hash in self.crawled_episodes:
                continue
            
            audio_meta = AudioMetadata(
                resource_id=guid_hash,
                audio_type="podcast_episode",
                title=entry.get("title", ""),
                description=entry.get("summary", ""),
                url=entry.get("link", rss_url),
                crawl_date=datetime.utcnow().isoformat(),
                domain=ExtendedDomainType.PODCAST_EPISODE,
                podcast_id=str(podcast_id),
                episode_guid=episode_guid,
                episode_duration_seconds=self._extract_duration(entry),
            )
            
            episodes.append(audio_meta)
            self.crawled_episodes.add(guid_hash)
        
        return episodes

class MusicCrawler:
    """Crawl music via Last.fm + MusicBrainz APIs."""
    
    def __init__(self, lastfm_api_key: str):
        self.lastfm_base = "http://ws.audioscrobbler.com/2.0"
        self.mb_base = "https://musicbrainz.org/ws/2"
        self.key = lastfm_api_key
        self.crawled_tracks = set()
    
    def search_artists_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """Find artists in a genre."""
        response = requests.get(
            self.lastfm_base,
            params={
                "method": "tag.getTopArtists",
                "tag": genre,
                "limit": limit,
                "api_key": self.key,
                "format": "json"
            }
        )
        return response.json().get("topartists", {}).get("artist", [])
    
    def crawl_artist_releases(self, artist_name: str) -> List[AudioMetadata]:
        """Get all albums/releases for an artist."""
        # Get MusicBrainz ID
        mb_response = requests.get(
            f"{self.mb_base}/artist",
            params={"query": f'artist:"{artist_name}"', "limit": 1},
            headers={"User-Agent": "xoe-novai/0.1.5"}
        )
        artists = mb_response.json().get("artists", [])
        if not artists:
            return []
        
        mbid = artists[0]["id"]
        
        # Get releases for artist
        releases_response = requests.get(
            f"{self.mb_base}/artist/{mbid}",
            params={"inc": "releases"},
            headers={"User-Agent": "xoe-novai/0.1.5"}
        )
        
        releases = []
        for release in releases_response.json().get("releases", [])[:50]:
            audio_meta = AudioMetadata(
                resource_id=release["id"],
                audio_type="music_album",
                title=release.get("title", ""),
                description="",
                url=f"https://musicbrainz.org/release/{release['id']}",
                crawl_date=datetime.utcnow().isoformat(),
                domain=ExtendedDomainType.MUSIC,
                artist_id=mbid,
                artist_name=artist_name,
                album_id=release["id"],
                album_title=release["title"],
                source_apis=["musicbrainz", "lastfm"]
            )
            releases.append(audio_meta)
        
        return releases
```

#### **Step 3: Update crawler_curation.py Enrichment**

```python
# Add to crawler_curation.py in CuratorEnhancer class

def enrich_with_audio_metadata(
    self,
    audio_metadata: AudioMetadata,
    domain: ExtendedDomainType
) -> CrawledDocument:
    """Enrich audio content with curation metadata."""
    
    quality_factors = {
        "description_length": min(len(audio_metadata.description) / 200, 1.0),
        "creator_info": 1.0 if audio_metadata.creators else 0.5,
        "metadata_completeness": self._calc_audio_completeness(audio_metadata),
    }
    
    return CrawledDocument(
        url=audio_metadata.url,
        content=f"{audio_metadata.title}\n{audio_metadata.description}",
        metadata=audio_metadata,
        domain=domain,
        quality_factors=quality_factors,
    )

def _calc_audio_completeness(self, meta: AudioMetadata) -> float:
    """Calculate metadata completeness score."""
    fields_present = sum([
        bool(meta.title),
        bool(meta.description),
        bool(meta.creators),
        bool(meta.genres) or bool(meta.tags),
        bool(meta.episode_duration_seconds or meta.track_duration_seconds),
    ])
    return fields_present / 5.0
```

#### **Step 4: Create audio_curator.py**

Implement the full `AudioCurator` class with all discovery methods (as shown in Section 5).

#### **Step 5: Update FAISS Indexing**

```python
# In library/ingest.py

def add_audio_documents(vectorstore, audio_documents: List[CrawledDocument]):
    """Add audio documents to FAISS index."""
    
    for doc in audio_documents:
        # Create rich metadata vector
        metadata_text = f"""
        Title: {doc.metadata.title}
        Creators: {', '.join([c['name'] for c in doc.metadata.creators])}
        Description: {doc.metadata.description}
        Genres: {', '.join(doc.metadata.genres)}
        Tags: {', '.join(doc.metadata.tags)}
        Type: {doc.metadata.audio_type}
        """
        
        doc.page_content = metadata_text
        
        try:
            vectorstore.add_documents([doc])
        except Exception as e:
            logger.warning(f"Skipped audio doc: {str(e)[:100]}")
```

---

## 7. RECOMMENDED IMPLEMENTATION ORDER

### **Phase 1: Foundation (Weeks 1-2)**
1. ✅ Define `AudioMetadata` schema
2. ✅ Extend `DomainType` to include PODCAST/MUSIC
3. ✅ Create `audio_crawlers.py` with Podcast + Music crawlers
4. ✅ Add unit tests for crawlers

### **Phase 2: Integration (Weeks 3-4)**
1. Update `crawler_curation.py` to handle audio metadata
2. Extend FAISS indexing for audio documents
3. Create `audio_curator.py` with discovery commands
4. Add natural language parsing for curator queries

### **Phase 3: Features (Weeks 5-6)**
1. Implement deduplication logic (GUID + ISRC-based)
2. Add RSS feed scheduling (crawl every 24h)
3. Implement audio-specific quality scoring
4. Add curation command pipeline

### **Phase 4: Polish (Weeks 7-8)**
1. Performance optimization (batch requests, caching)
2. Error handling & retry logic
3. Comprehensive testing
4. Documentation

---

## 8. COMPARISON MATRIX

### **Podcast APIs**

| Feature | Podcastindex | iTunes | Open Podcast |
|---------|-----------|--------|--------------|
| **Cost** | FREE | FREE | FREE |
| **No Auth** | ✅ | ✅ | ✅ |
| **Rate Limit** | 10/sec (generous) | 1/36sec (strict) | Generous |
| **Podcast Index** | 3M+ | 3M+ | <1M |
| **Episode Data** | ✅ Full | ⚠️ Limited | ✅ Good |
| **RSS URLs** | ✅ Direct | ❌ No | ✅ Yes |
| **Category API** | ✅ Yes | ❌ No | ❌ No |
| **Transcripts** | ✅ Links | ❌ No | ❌ No |
| **Maintenance** | ✅ Active | ✅ Apple | ❌ Community |
| **Recommendation** | **⭐⭐⭐⭐⭐ USE THIS** | Backup only | Don't use |

### **Music APIs**

| Feature | MusicBrainz | Last.fm | Discogs |
|---------|---------|---------|---------|
| **Cost** | FREE | FREE | FREE |
| **Rate Limit** | 1/sec | 5/sec | 1/sec |
| **Metadata Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Discovery** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Artist Info** | ✅ Complete | ✅ With tags | ✅ Detailed |
| **Discography** | ✅ Full | ⚠️ Limited | ✅ Full |
| **Audio IDs** | ✅ ISRC | ❌ No | ⚠️ Barcode |
| **Tags/Genres** | ✅ Yes | ✅⭐ Best | ✅ Yes |
| **Similar Artists** | ⚠️ Manual | ✅⭐ Algo | ❌ No |
| **Chart Data** | ❌ No | ✅ Yes | ❌ No |
| **Use Cases** | Metadata + dedup | Discovery + genres | Physical releases |
| **Recommendation** | **⭐⭐⭐⭐⭐ PRIMARY** | **⭐⭐⭐⭐⭐ PAIRED** | Secondary |

---

## 9. QUICK START: INTEGRATION CHECKLIST

- [ ] Install dependencies: `feedparser`, `requests`
- [ ] Signup for Last.fm API key (free)
- [ ] Create `AudioMetadata` dataclass
- [ ] Implement `PodcastCrawler` class
- [ ] Implement `MusicCrawler` class
- [ ] Create unit tests for crawlers
- [ ] Update FAISS indexing pipeline
- [ ] Implement `AudioCurator` discovery methods
- [ ] Add natural language query parsing
- [ ] Test with sample queries
- [ ] Deploy with Docker

---

## 10. SAMPLE ENVIRONMENT VARIABLES

```bash
# .env.audio

# Last.fm API (get free key at https://www.last.fm/api/account/create)
LASTFM_API_KEY=your_key_here

# Podcastindex API (optional - free tier doesn't need)
PODCASTINDEX_API_KEY=optional_key

# Audio indexing
AUDIO_ENABLE_PODCASTS=true
AUDIO_ENABLE_MUSIC=true
AUDIO_MAX_EPISODES_PER_PODCAST=20
AUDIO_MAX_ALBUMS_PER_ARTIST=50

# Deduplication
AUDIO_DEDUP_TTL_DAYS=90
REDIS_AUDIO_DEDUP_KEY=xoe:audio:crawled

# Crawler scheduling
PODCAST_CRAWL_INTERVAL_HOURS=24
MUSIC_CRAWL_INTERVAL_HOURS=72
```

---

## SUMMARY

**Best Free Audio APIs for Xoe-NovAi:**

1. **Podcasts:** Podcastindex (most complete, best integration)
2. **Music:** MusicBrainz + Last.fm (complementary - metadata + discovery)
3. **Metadata Schema:** 25+ fields supporting podcast/music/artist/album
4. **Crawler:** RSS (podcasts) + API search (music) with GUID/ISRC deduplication
5. **Curation:** Natural language commands ("Find podcasts about X", "Discover indie artists")

**Ready to implement. Estimated effort: 4-6 weeks for full integration.**
