#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.5 - Voice Interface (ENHANCED + OBSERVABILITY)
# ============================================================================
# Purpose: Torch-free voice interface with Piper ONNX primary TTS
# Version: v0.1.5 (2026-01-08)
# Features:
#   - Faster Whisper STT (torch-free, CTranslate2 backend)
#   - Piper ONNX TTS primary (torch-free, real-time CPU)
#   - "Hey Nova" wake word detection
#   - Streaming audio support with VAD
#   - Robust input validation and rate limiting
#   - Prometheus metrics for observability
#   - Circuit breaker pattern for resilience
#   - FAISS integration for voice-powered RAG
#   - Redis integration for session persistence
#   - Conversation memory and context tracking
# ============================================================================

import os
import logging
import asyncio
import io
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import time
import threading
from pathlib import Path

# CRITICAL FIX: Import path resolution (Pattern 1)
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Prometheus metrics (optional import)
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Lightweight optional imports (guarded)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except Exception:
    PIPER_AVAILABLE = False
    PiperVoice = None

try:
    import pyttsx3
    PYTTX3_AVAILABLE = True
except Exception:
    PYTTX3_AVAILABLE = False
    pyttsx3 = None

# Redis for session persistence
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics for Voice Subsystem
# ============================================================================

class VoiceMetrics:
    """Prometheus metrics for voice subsystem observability."""
    
    def __init__(self):
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._initialized = False
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all voice-related Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.stt_requests_total = Counter(
            'xoe_voice_stt_requests_total',
            'Total STT transcription requests',
            ['status', 'provider'],
            registry=self._registry
        )
        
        self.tts_requests_total = Counter(
            'xoe_voice_tts_requests_total',
            'Total TTS synthesis requests',
            ['status', 'provider'],
            registry=self._registry
        )
        
        self.wake_word_detections_total = Counter(
            'xoe_voice_wake_word_detections_total',
            'Total wake word detections',
            ['status'],
            registry=self._registry
        )
        
        self.rate_limit_exceeded_total = Counter(
            'xoe_voice_rate_limit_exceeded_total',
            'Total rate limit exceeded events',
            ['client_id'],
            registry=self._registry
        )
        
        self.stt_latency_seconds = Histogram(
            'xoe_voice_stt_latency_seconds',
            'STT transcription latency',
            ['provider'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self._registry
        )
        
        self.tts_latency_seconds = Histogram(
            'xoe_voice_tts_latency_seconds',
            'TTS synthesis latency',
            ['provider'],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self._registry
        )
        
        self.audio_input_level = Gauge(
            'xoe_voice_audio_input_level',
            'Current audio input level (0-1)',
            registry=self._registry
        )
        
        self.stt_model_loaded = Gauge(
            'xoe_voice_stt_model_loaded',
            'Whether STT model is loaded',
            ['provider'],
            registry=self._registry
        )
        
        self.tts_model_loaded = Gauge(
            'xoe_voice_tts_model_loaded',
            'Whether TTS model is loaded',
            ['provider'],
            registry=self._registry
        )
        
        self.circuit_breaker_open = Gauge(
            'xoe_voice_circuit_breaker_open',
            'Whether circuit breaker is open',
            ['component'],
            registry=self._registry
        )
        
        self.voice_info = Info(
            'xoe_voice',
            'Voice subsystem configuration',
            registry=self._registry
        )
        self.voice_info.info({
            'version': 'v0.1.5',
            'stt_provider': 'faster_whisper',
            'tts_provider': 'piper_onnx',
        })
        
        self._initialized = True
    
    def record_stt_request(self, status: str, provider: str, latency: float):
        if not self._initialized:
            return
        self.stt_requests_total.labels(status=status, provider=provider).inc()
        self.stt_latency_seconds.labels(provider=provider).observe(latency)
    
    def record_tts_request(self, status: str, provider: str, latency: float):
        if not self._initialized:
            return
        self.tts_requests_total.labels(status=status, provider=provider).inc()
        self.tts_latency_seconds.labels(provider=provider).observe(latency)
    
    def record_wake_word(self, success: bool):
        if not self._initialized:
            return
        status = "success" if success else "false_positive"
        self.wake_word_detections_total.labels(status=status).inc()
    
    def record_rate_limit_exceeded(self, client_id: str):
        if not self._initialized:
            return
        self.rate_limit_exceeded_total.labels(client_id=client_id).inc()
    
    def update_model_loaded(self, component: str, provider: str, loaded: bool):
        if not self._initialized:
            return
        if component == "stt":
            self.stt_model_loaded.labels(provider=provider).set(1 if loaded else 0)
        elif component == "tts":
            self.tts_model_loaded.labels(provider=provider).set(1 if loaded else 0)
    
    def update_circuit_breaker(self, component: str, open: bool):
        if not self._initialized:
            return
        self.circuit_breaker_open.labels(component=component).set(1 if open else 0)
    
    def get_metrics(self) -> bytes:
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return b"# Voice metrics unavailable"
        return generate_latest(self._registry)


voice_metrics = VoiceMetrics()


# ============================================================================
# Circuit Breaker for Resilience
# ============================================================================

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class VoiceCircuitBreaker:
    """Circuit breaker pattern for voice operations."""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and (time.time() - self._last_failure_time) > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
            return self._state
    
    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN
    
    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= 3:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            voice_metrics.update_circuit_breaker(self.name, open=False)
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            voice_metrics.update_circuit_breaker(self.name, open=True)


# ============================================================================
# Configuration & Enums
# ============================================================================

class STTProvider(str, Enum):
    FASTER_WHISPER = "faster_whisper"
    WHISPER_TURBO = "whisper_turbo"

class TTSProvider(str, Enum):
    PIPER_ONNX = "piper_onnx"
    PYTTSX3 = "pyttsx3"

class WhisperModel_(str, Enum):
    DISTIL_LARGE = "distil-large-v3"


@dataclass
class VoiceConfig:
    stt_provider: STTProvider = STTProvider.FASTER_WHISPER
    whisper_model: WhisperModel_ = WhisperModel_.DISTIL_LARGE
    stt_device: str = "cpu"
    stt_compute_type: str = "float16"
    stt_beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    stt_timeout_seconds: int = 60
    
    tts_provider: TTSProvider = TTSProvider.PIPER_ONNX
    piper_model: str = "en_US-john-medium"
    tts_timeout_seconds: int = 30
    
    wake_word: str = "hey nova"
    wake_word_enabled: bool = True
    wake_word_sensitivity: float = 0.8
    
    language: str = "en"
    language_code: str = "en"
    
    max_audio_size_bytes: int = 10 * 1024 * 1024
    max_audio_duration_seconds: int = 300
    rate_limit_per_minute: int = 10
    rate_limit_window_seconds: int = 60
    
    streaming_enabled: bool = True
    streaming_buffer_size: int = 4096
    
    offline_mode: bool = True
    preload_models: bool = False
    
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 1000
    
    def validate(self) -> Tuple[bool, str]:
        errors = []
        if self.max_audio_size_bytes < 1024:
            errors.append("max_audio_size_bytes must be at least 1KB")
        if not 0.0 <= self.wake_word_sensitivity <= 1.0:
            errors.append("wake_word_sensitivity must be between 0.0 and 1.0")
        if errors:
            return False, "; ".join(errors)
        return True, "Configuration valid"


# ============================================================================
# Rate Limiter
# ============================================================================

class VoiceRateLimiter:
    """Token bucket rate limiter for voice API."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def allow_request(self, client_id: str) -> Tuple[bool, str]:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove expired requests
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window_seconds]
        
        if len(self.requests[client_id]) >= self.max_requests:
            voice_metrics.record_rate_limit_exceeded(client_id)
            remaining = 0
            return False, f"Rate limit exceeded. {remaining}/{self.max_requests} requests remaining"
        
        self.requests[client_id].append(now)
        remaining = self.max_requests - len(self.requests[client_id])
        return True, f"{remaining}/{self.max_requests} requests remaining"


# ============================================================================
# Wake Word Detection
# ============================================================================

class WakeWordDetector:
    """'Hey Nova' wake word detection using regex patterns."""
    
    def __init__(self, wake_word: str = "hey nova", sensitivity: float = 0.8):
        self.wake_word = wake_word.lower().strip()
        self.sensitivity = sensitivity
        self.patterns = self._build_patterns()
        self.stats = {"total_checks": 0, "detections": 0, "false_positives": 0}
    
    def _build_patterns(self) -> List:
        import re
        patterns = []
        wake_words = self.wake_word.split()
        if len(wake_words) >= 2:
            first, second = wake_words[0], wake_words[1]
            patterns.append(re.compile(rf'\b{re.escape(first)}\s+{re.escape(second)}\b', re.IGNORECASE))
            patterns.append(re.compile(rf'\b{re.escape(first)}\s*[!?.]*\s*{re.escape(second)}\b', re.IGNORECASE))
        return patterns
    
    def detect(self, transcription: str) -> Tuple[bool, float]:
        if not transcription:
            return False, 0.0
        
        self.stats["total_checks"] += 1
        text_lower = transcription.lower().strip()
        
        for pattern in self.patterns:
            match = pattern.search(text_lower)
            if match:
                match_ratio = len(match.group()) / len(text_lower) if text_lower else 0
                position_bonus = 1.0 - (match.start() / len(text_lower)) if text_lower else 0
                confidence = min(1.0, match_ratio * 0.3 + position_bonus * 0.5 + self.sensitivity * 0.2)
                
                if confidence >= 0.5:
                    self.stats["detections"] += 1
                    voice_metrics.record_wake_word(True)
                    return True, confidence
        
        self.stats["false_positives"] += 1
        voice_metrics.record_wake_word(False)
        return False, 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total_checks"]
        return {
            **self.stats,
            "detection_rate": self.stats["detections"] / total if total > 0 else 0.0,
        }


# ============================================================================
# Voice Session Manager (Redis Persistence)
# ============================================================================

class VoiceSessionManager:
    """
    Manages voice conversation sessions with Redis persistence.
    
    Follows stack patterns from docs/reference/blueprint.md:
    - Session tracking with TTL: 1 hour
    - Conversation memory storage
    - Context retrieval for RAG queries
    
    Redis key patterns:
    - xnai:voice:session:{session_id} - Full session data
    - xnai:voice:conversation:{session_id} - Conversation history
    - xnai:voice:context:{session_id} - LLM context window
    """
    
    SESSION_TTL = 3600  # 1 hour
    CONTEXT_TTL = 3600
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        redis_client: Optional[Any] = None,
        redis_host: str = "redis",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._redis_client = redis_client
        self._redis_config = {
            "host": redis_host,
            "port": redis_port,
            "password": redis_password,
        }
        self._connected = False
        self._connect()
        
        # In-memory cache for fast access
        self._session_data: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "conversation_history": [],
            "user_preferences": {},
            "metrics": {
                "total_interactions": 0,
                "total_transcriptions": 0,
                "total_responses": 0,
            },
        }
    
    def _connect(self):
        """Connect to Redis using stack patterns."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - session persistence disabled")
            return
        
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis(
                    host=self._redis_config["host"],
                    port=self._redis_config["port"],
                    password=self._redis_config["password"],
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self._redis_client.ping()
                self._connected = True
                logger.info(f"Voice session Redis connected: {self._redis_config['host']}:{self._redis_config['port']}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._redis_client is not None
    
    def _get_key(self, key_type: str) -> str:
        """Generate Redis key with namespace."""
        return f"xnai:voice:{key_type}:{self.session_id}"
    
    def save_session(self) -> bool:
        """Persist session to Redis."""
        if not self.is_connected:
            return False
        
        try:
            session_key = self._get_key("session")
            self._redis_client.setex(
                session_key,
                self.SESSION_TTL,
                json.dumps(self._session_data, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self) -> bool:
        """Load session from Redis."""
        if not self.is_connected:
            return False
        
        try:
            session_key = self._get_key("session")
            data = self._redis_client.get(session_key)
            if data:
                self._session_data = json.loads(data)
                self.session_id = self._session_data.get("session_id", self.session_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add conversation turn to history."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user" or "assistant"
            "content": content,
            "metadata": metadata or {},
        }
        
        self._session_data["conversation_history"].append(interaction)
        self._session_data["metrics"]["total_interactions"] += 1
        
        if role == "user":
            self._session_data["metrics"]["total_transcriptions"] += 1
        else:
            self._session_data["metrics"]["total_responses"] += 1
        
        # Persist to Redis
        self.save_session()
        
        # Also save to conversation-specific key for RAG context
        if self.is_connected:
            try:
                conv_key = self._get_key("conversation")
                self._redis_client.rpush(conv_key, json.dumps(interaction, default=str))
                self._redis_client.expire(conv_key, self.SESSION_TTL)
            except Exception:
                pass
    
    def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get conversation history for LLM context."""
        history = self._session_data.get("conversation_history", [])
        recent = history[-max_turns:]
        
        context_parts = []
        for turn in recent:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_session(self):
        """Clear session data and Redis keys."""
        self._session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "conversation_history": [],
            "user_preferences": {},
            "metrics": {
                "total_interactions": 0,
                "total_transcriptions": 0,
                "total_responses": 0,
            },
        }
        
        if self.is_connected:
            try:
                pattern = self._get_key("*")
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to clear session: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "connected": self.is_connected,
            "created_at": self._session_data.get("created_at"),
            "total_interactions": self._session_data["metrics"]["total_interactions"],
            "total_transcriptions": self._session_data["metrics"]["total_transcriptions"],
            "total_responses": self._session_data["metrics"]["total_responses"],
            "conversation_turns": len(self._session_data.get("conversation_history", [])),
        }


# ============================================================================
# Voice FAISS Client (Knowledge Retrieval)
# ============================================================================

class VoiceFAISSClient:
    """
    FAISS-powered knowledge retrieval for voice queries.
    
    Integrates with voice interface for RAG-powered responses.
    Supports both indexed documents and on-the-fly embedding.
    """
    
    DEFAULT_TOP_K = 3
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embeddings_model: Optional[Any] = None,
    ):
        self.index_path = index_path or "/app/XNAi_rag_app/faiss_index"
        self.embeddings_model = embeddings_model
        self.index = None
        self._index_loaded = False
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index from disk."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - RAG disabled")
            return
        
        index_file = Path(self.index_path) / "index.faiss"
        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                self._index_loaded = True
                logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
        else:
            logger.warning(f"FAISS index not found at {index_file}")
    
    @property
    def is_available(self) -> bool:
        return self._index_loaded and self.index is not None
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search knowledge base for query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if not self.is_available:
            return [{"error": "FAISS index not available", "content": ""}]
        
        top_k = top_k or self.DEFAULT_TOP_K
        
        # Get embedding for query
        if self.embeddings_model is None:
            # Simple keyword fallback if no embeddings
            return self._keyword_search(query, top_k)
        
        try:
            query_embedding = self.embeddings_model.encode([query])
            
            # Search index
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:
                    continue
                results.append({
                    "rank": i + 1,
                    "index": int(idx),
                    "score": float(dist),
                    "metadata": {"source": "faiss_index"},
                })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return [{"error": str(e), "content": ""}]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search fallback."""
        # This is a placeholder - in production, you'd use a document store
        return [{
            "rank": 1,
            "score": 0.0,
            "content": f"Keyword match for: {query}",
            "metadata": {"source": "keyword_fallback"},
        }]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        if not self.is_available:
            return {"available": False}
        
        return {
            "available": True,
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d if hasattr(self.index, 'd') else "unknown",
        }


# ============================================================================
# Audio Stream Processor
# ============================================================================


class AudioStreamProcessor:
    """Streaming audio processor with VAD."""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_buffer = bytearray()
        self.chunk_size = config.streaming_buffer_size
        self.silence_threshold = 0.02
        self.silence_duration = 1.5
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_start_time = None
        self.stats = {"total_chunks": 0, "total_bytes": 0, "speech_segments": 0, "silence_segments": 0}
    
    def add_chunk(self, audio_data: bytes) -> bool:
        if not audio_data:
            return False
        
        self.audio_buffer.extend(audio_data)
        self.stats["total_chunks"] += 1
        self.stats["total_bytes"] += len(audio_data)
        
        try:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            is_speech = energy > (self.silence_threshold * 32767)
            
            current_time = datetime.now()
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.stats["speech_segments"] += 1
                self.last_speech_time = current_time
            else:
                if self.is_speaking and self.last_speech_time:
                    silence_duration = (current_time - self.last_speech_time).total_seconds()
                    if silence_duration >= self.silence_duration:
                        self.is_speaking = False
                        self.stats["silence_segments"] += 1
            
            return self.is_speaking
        except Exception:
            return True
    
    def get_audio_data(self) -> bytes:
        data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return data
    
    def reset(self):
        self.audio_buffer.clear()
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_start_time = None


# ============================================================================
# Core Voice Interface
# ============================================================================

class VoiceInterface:
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.session_id = datetime.now().isoformat()
        self.stt_model = None
        self.tts_model = None
        self.stt_provider_name = "faster_whisper"
        self.tts_provider_name = "piper_onnx"
        
        # Circuit breakers
        self.stt_circuit = VoiceCircuitBreaker("stt")
        self.tts_circuit = VoiceCircuitBreaker("tts")
        
        # Metrics
        self.metrics = {
            "total_transcriptions": 0,
            "total_voice_outputs": 0,
            "avg_stt_latency_ms": 0.0,
            "avg_tts_latency_ms": 0.0,
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        if self.config.offline_mode and not self.config.preload_models:
            logger.info("Offline mode: Deferring model loading")
            return
        
        # STT model loading
        if self.config.stt_provider == STTProvider.FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
            try:
                self.stt_model = WhisperModel(
                    self.config.whisper_model.value,
                    device=self.config.stt_device,
                    compute_type=self.config.stt_compute_type,
                )
                voice_metrics.update_model_loaded("stt", self.stt_provider_name, True)
                logger.info("Faster Whisper loaded (torch-free).")
            except Exception as e:
                logger.error(f"Failed to load Faster Whisper: {e}")
                voice_metrics.update_model_loaded("stt", self.stt_provider_name, False)

        # TTS: Piper ONNX primary
        if self.config.tts_provider == TTSProvider.PIPER_ONNX and PIPER_AVAILABLE:
            try:
                self.tts_model = PiperVoice.load(f"{self.config.piper_model}.onnx")
                voice_metrics.update_model_loaded("tts", self.tts_provider_name, True)
                logger.info("Piper ONNX TTS loaded (torch-free).")
            except Exception as e:
                logger.error(f"Failed to load Piper ONNX: {e}")
                voice_metrics.update_model_loaded("tts", self.tts_provider_name, False)

        # Fallback to pyttsx3
        if self.tts_model is None and PYTTX3_AVAILABLE:
            try:
                self.tts_model = pyttsx3.init()
                self.tts_provider_name = "pyttsx3"
                voice_metrics.update_model_loaded("tts", self.tts_provider_name, True)
                logger.info("pyttsx3 system TTS initialized (offline).")
            except Exception as e:
                logger.error(f"pyttsx3 init failed: {e}")

    async def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> Tuple[str, float]:
        """Transcribe audio with timeout protection and circuit breaker."""
        if not audio_data:
            return "[No audio data]", 0.0
        
        if len(audio_data) > self.config.max_audio_size_bytes:
            return "[Audio too large]", 0.0
        
        if self.stt_model is None:
            return "[STT Model not loaded]", 0.0
        
        # Check circuit breaker
        if not self.stt_circuit.allow_request():
            return "[STT temporarily unavailable]", 0.0

        t0 = time.time()
        audio_file = io.BytesIO(audio_data)

        try:
            if hasattr(asyncio, 'timeout'):
                async with asyncio.timeout(self.config.stt_timeout_seconds):
                    segments, info = self.stt_model.transcribe(
                        audio_file,
                        beam_size=self.config.stt_beam_size,
                        language=self.config.language_code,
                        vad_filter=self.config.vad_filter,
                    )
            else:
                segments, info = await asyncio.wait_for(
                    self._transcribe_impl(audio_file),
                    timeout=self.config.stt_timeout_seconds
                )
            
            transcription = " ".join([segment.text for segment in segments])
            confidence = getattr(info, "language_probability", 0.95)
            latency = time.time() - t0
            
            self.stt_circuit.record_success()
            voice_metrics.record_stt_request("success", self.stt_provider_name, latency)
            
        except asyncio.TimeoutError:
            logger.error(f"STT transcription timed out after {self.config.stt_timeout_seconds}s")
            self.stt_circuit.record_failure()
            voice_metrics.record_stt_request("timeout", self.stt_provider_name, 0)
            return "[Transcription timeout]", 0.0
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.stt_circuit.record_failure()
            voice_metrics.record_stt_request("error", self.stt_provider_name, 0)
            return "[Transcription error]", 0.0

        self.metrics["total_transcriptions"] += 1
        if self.metrics["total_transcriptions"] > 1:
            self.metrics["avg_stt_latency_ms"] = (
                self.metrics["avg_stt_latency_ms"] * (self.metrics["total_transcriptions"] - 1) +
                latency * 1000
            ) / self.metrics["total_transcriptions"]
        else:
            self.metrics["avg_stt_latency_ms"] = latency * 1000

        return transcription, confidence

    async def _transcribe_impl(self, audio_file: io.BytesIO):
        """Internal transcription implementation."""
        segments, info = self.stt_model.transcribe(
            audio_file,
            beam_size=self.config.stt_beam_size,
            language=self.config.language_code,
            vad_filter=self.config.vad_filter,
        )
        return list(segments), info

    async def synthesize_speech(self, text: str, speaker_wav: Optional[str] = None, language: str = "en") -> Optional[bytes]:
        """Synthesize speech with circuit breaker protection."""
        if self.tts_model is None:
            return None
        
        if not self.tts_circuit.allow_request():
            logger.warning("TTS circuit breaker open - request rejected")
            return None

        t0 = time.time()
        audio_bytes = None

        try:
            if self.tts_provider_name == "piper_onnx":
                buf = io.BytesIO()
                self.tts_model.synthesize(text, buf)
                audio_bytes = buf.getvalue()
            elif self.tts_provider_name == "pyttsx3":
                temp_path = "/tmp/xoe_voice_output.wav"
                self.tts_model.save_to_file(text, temp_path)
                self.tts_model.runAndWait()
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
            latency = time.time() - t0
            self.tts_circuit.record_success()
            voice_metrics.record_tts_request("success", self.tts_provider_name, latency)
            
            self.metrics["total_voice_outputs"] += 1
            if self.metrics["total_voice_outputs"] > 1:
                self.metrics["avg_tts_latency_ms"] = (
                    self.metrics["avg_tts_latency_ms"] * (self.metrics["total_voice_outputs"] - 1) +
                    latency * 1000
                ) / self.metrics["total_voice_outputs"]
            else:
                self.metrics["avg_tts_latency_ms"] = latency * 1000
            
            logger.info(f"TTS complete: {len(audio_bytes)} bytes, {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            self.tts_circuit.record_failure()
            voice_metrics.record_tts_request("error", self.tts_provider_name, 0)
            return None

        return audio_bytes

    def get_session_stats(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "stt_circuit_state": self.stt_circuit.state.value,
            "tts_circuit_state": self.tts_circuit.state.value,
        }


# Global helpers
_voice_instance: Optional[VoiceInterface] = None

def setup_voice_interface(config: Optional[VoiceConfig] = None) -> VoiceInterface:
    global _voice_instance
    _voice_instance = VoiceInterface(config or VoiceConfig())
    return _voice_instance

def get_voice_interface() -> Optional[VoiceInterface]:
    return _voice_instance


# Demo
if __name__ == "__main__":
    print("Xoe-NovAi Voice Interface v0.1.5")
    print("=" * 40)
    
    cfg = VoiceConfig()
    v = VoiceInterface(cfg)
    print(f"Config valid: {cfg.validate()}")
    
    # Test wake word
    detector = WakeWordDetector(wake_word="hey nova", sensitivity=0.8)
    for phrase in ["Hey Nova, hello", "Good morning Nova", "What is AI?"]:
        detected, conf = detector.detect(phrase)
        print(f"  [{'DETECTED' if detected else 'MISS'}] '{phrase}' (conf: {conf:.2f})")
    
    print(f"\nMetrics endpoint: /metrics (Prometheus format)")
    print(voice_metrics.get_metrics()[:500].decode())
