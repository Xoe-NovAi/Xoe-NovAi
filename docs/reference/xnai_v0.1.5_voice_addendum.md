# Xoe-NovAi v0.1.5 Voice Integration Addendum

**Version:** v0.1.5 (2026-01-08)  
**Status:** Production Ready  
**Purpose:** Voice-to-Voice AI with Redis session persistence and FAISS knowledge retrieval

---

## New Components (v0.1.5)

### Core Voice Classes

| Class | File | Purpose |
|-------|------|---------|
| `VoiceSessionManager` | voice_interface.py | Redis session persistence (1-hour TTL) |
| `VoiceFAISSClient` | voice_interface.py | FAISS knowledge retrieval |
| `VoiceMetrics` | voice_interface.py | Prometheus metrics for voice |
| `VoiceCircuitBreaker` | voice_interface.py | Fault tolerance |
| `WakeWordDetector` | voice_interface.py | "Hey Nova" detection |

### Redis Session Keys

```python
# Session persistence pattern
xnai:voice:session:{session_id}      # Full session data (TTL: 3600s)
xnai:voice:conversation:{session_id}  # Conversation history
```

### Prometheus Voice Metrics

```bash
# View voice metrics
curl http://localhost:8002/metrics | grep xoe_voice

# Key metrics:
xoe_voice_stt_requests_total{status,provider}
xoe_voice_tts_requests_total{status,provider}
xoe_voice_circuit_breaker_open{component}
xoe_voice_stt_latency_seconds{provider}
```

---

## Voice Architecture

```
Microphone → Wake Word ("Hey Nova") → VAD → Faster Whisper STT
                                                ↓
                                         Redis Session
                                                ↓
                                         FAISS Search
                                                ↓
                                         RAG API + Context
                                                ↓
                                         Piper ONNX TTS → Speaker
```

---

## Configuration

### config.toml

```toml
[voice]
enabled = true
stt_provider = "faster_whisper"
tts_provider = "piper_onnx"
wake_word = "hey nova"
wake_word_enabled = true
wake_word_sensitivity = 0.8
offline_mode = true
```

---

## Usage Example

```python
from voice_interface import (
    VoiceSessionManager,
    VoiceFAISSClient,
)

# Initialize Redis session
session = VoiceSessionManager(
    session_id="conv_abc123",
    redis_host="redis",
    redis_port=6379,
)

# Save conversation turn
session.add_interaction("user", "Find books about AI")
session.add_interaction("assistant", "Here are some books about AI...")

# Get context for RAG
context = session.get_conversation_context(max_turns=5)

# Search knowledge base
faiss = VoiceFAISSClient(index_path="/app/XNAi_rag_app/faiss_index")
results = faiss.search("artificial intelligence", top_k=3)
```

---

## Integration Points

### Chainlit Integration

```python
from chainlit_app_voice import (
    on_chat_start,
    on_message,
    process_voice_input,
    generate_ai_response,
    generate_voice_response,
)
```

### RAG API Context

```json
{
  "query": "What is AI?",
  "use_rag": true,
  "voice_input": true,
  "conversation_context": "user: Find books about AI\nassistant: Here are some books...",
  "knowledge_context": "Vector match: 'AI is...'"
}
```

---

## Changelog Summary

### v0.1.5 (2026-01-08)
- Voice-to-Voice AI with "Hey Nova" wake word
- Redis session persistence (VoiceSessionManager)
- FAISS knowledge retrieval (VoiceFAISSClient)
- Prometheus voice metrics
- Circuit breaker resilience
- Piper ONNX TTS (torch-free)
- Faster Whisper STT (torch-free)
- Streaming audio with VAD

---

*This addendum should be read in conjunction with `docs/reference/blueprint.md` for complete stack documentation.*
