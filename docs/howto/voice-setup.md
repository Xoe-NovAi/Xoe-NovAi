# Voice Interface Quick Start Guide

**Created:** January 3, 2026  
**Version:** v0.1.5-voice-enabled  
**Status:** ✅ Ready for Deployment

---

## What's New in v0.1.5

### 🎯 "Hey Nova" Wake Word Detection

Activate voice mode hands-free by saying **"Hey Nova"**:

```
You: "Hey Nova, find books about AI"
Nova: "Sure! Searching for books about AI..."
```

**Features:**
- No button press needed to start listening
- Adjustable sensitivity (0.5 - 1.0)
- 100% local - no cloud dependencies

### 🔊 Piper ONNX TTS (100% Offline)

**Industry-leading CPU-optimized TTS:**
- Zero PyTorch dependency
- Real-time synthesis (<100ms latency)
- ONNX runtime for maximum efficiency
- Multiple voice models available

**Setup (Automatic):**
```bash
# Already included in requirements-chainlit.txt
pip install piper-onnx
```

### ⚡ Streaming Audio Support

Real-time voice-to-voice conversation:

1. **Wake Word**: "Hey Nova" activates listening
2. **Streaming VAD**: Automatic silence detection
3. **Faster Whisper STT**: Transcribes in real-time
4. **Piper ONNX TTS**: Speaks response immediately

### 🛡️ Rate Limiting & Input Validation

**Protection features:**
- 10 requests/minute per client
- 10MB audio size limit
- 5-minute audio duration limit
- Token bucket algorithm

### 💾 Redis Session Persistence

**Voice conversations are persisted to Redis:**
- Session ID: `xnai:voice:session:{id}`
- Conversation history: `xnai:voice:conversation:{id}`
- 1-hour TTL automatically applied
- Survives app restarts

### 🔍 FAISS Knowledge Retrieval

**Voice queries search the knowledge base:**
- Queries FAISS index at `/app/XNAi_rag_app/faiss_index`
- Top-3 semantic matches returned
- Context passed to RAG pipeline
- Falls back to keyword search if embeddings unavailable

### 📊 Prometheus Metrics

**Voice operations monitored:**
- `xoe_voice_stt_requests_total{status,provider}`
- `xoe_voice_tts_requests_total{status,provider}`
- `xoe_voice_circuit_breaker_open{component}`
- `xoe_voice_stt_latency_seconds{provider}`

### 🔄 Circuit Breaker Resilience

**Prevents cascade failures:**
- Opens after 5 consecutive failures
- 30-second recovery timeout
- Half-open state for health checks
- Automatic metrics updates

---

## Quick Start

### Option 1: Chainlit Voice App

```bash
chainlit run app/XNAi_rag_app/chainlit_app_voice.py -w --port 8001
```

### Option 2: Docker

```bash
docker-compose up -d chainlit
```

---

## Wake Word Configuration

```toml
# config.toml
[voice]
wake_word = "hey nova"
wake_word_enabled = true
wake_word_sensitivity = 0.8
```

**Commands:**
- Say "Hey Nova" to activate
- Say "stop voice chat" to deactivate
- Say "voice settings" to adjust sensitivity

---

## TTS Provider Selection

| Provider | Quality | Latency | Offline | Dependencies |
|----------|---------|---------|---------|--------------|
| **Piper ONNX** | ⭐⭐⭐⭐ | <100ms | ✅ | None |
| pyttsx3 | ⭐⭐⭐ | Variable | ✅ | System voices |
| GTTS | ⭐⭐⭐⭐ | 200ms | ❌ | Internet |
| ElevenLabs | ⭐⭐⭐⭐⭐ | 50ms | ❌ | API key |

**Recommended: Piper ONNX** (default in v0.1.5)

---

## Voice Commands Reference

### Activation
| Command | Action |
|---------|--------|
| "Hey Nova" | Activate voice mode |
| "Stop voice chat" | Deactivate voice mode |
| "Voice settings" | Open settings panel |

### Playback Control
| Command | Action |
|---------|--------|
| "Speak slower" | 0.5x speed |
| "Speak faster" | 1.5x speed |
| "Higher pitch" | +0.2 pitch |
| "Lower pitch" | -0.2 pitch |
| "Louder" | +20% volume |
| "Quieter" | -20% volume |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice-to-Voice Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Microphone  │───▶│ Wake Word   │───▶│ VAD (Silence)   │  │
│  │             │    │ "Hey Nova"  │    │ Detection       │  │
│  └─────────────┘    └─────────────┘    └────────┬────────┘  │
│                                                  │           │
│                                                  ▼           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Faster Whisper STT                         │ │
│  │  - torch-free CTranslate2 backend                      │ │
│  │  - VAD filtering enabled                               │ │
│  │  - 5-beam search                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              RAG Pipeline                                │ │
│  │  - Query understanding                                  │ │
│  │  - Knowledge retrieval                                  │ │
│  │  - Response generation                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Piper ONNX TTS                              │ │
│  │  - ONNX runtime                                         │ │
│  │  - 100ms latency                                        │ │
│  │  - Multiple speakers                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────┐                                               │
│  │ Speaker     │                                               │
│  └─────────────┘                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Offline Mode

**100% offline voice pipeline:**

```toml
[voice]
offline_mode = true
stt_provider = "faster_whisper"
tts_provider = "piper_onnx"
```

**No external dependencies:**
- No OpenAI API calls
- No Google TTS
- No ElevenLabs
- All processing local

---

## Performance Optimization

### CPU Optimization (AMD Ryzen)

```toml
[voice]
stt_compute_type = "float16"
stt_beam_size = 5
streaming_buffer_size = 4096
```

**Expected Performance:**
- Transcription: <500ms for 10s audio
- Synthesis: <100ms response time
- Wake word: <200ms detection

### Memory Optimization

```toml
[voice]
max_audio_size_bytes = 10485760  # 10MB
cache_ttl_seconds = 3600
cache_max_entries = 1000
```

---

## Troubleshooting

### Wake Word Not Detecting

```bash
# Check wake word detector initialization
python3 -c "
from app.XNAi_rag_app.voice_interface import WakeWordDetector
detector = WakeWordDetector(wake_word='hey nova', sensitivity=0.8)
print('Wake word detector ready')
"
```

### Piper ONNX Not Loading

```bash
# Verify installation
pip show piper-onnx

# Check model files exist
ls -la models/piper/
```

### Audio Quality Issues

1. **Low volume**: Adjust microphone gain
2. **Noise**: Use VAD filtering (enabled by default)
3. **Dropouts**: Increase streaming buffer size

---

## Configuration Reference

```toml
[voice]
enabled = true

# Wake Word
wake_word = "hey nova"
wake_word_enabled = true
wake_word_sensitivity = 0.8

# STT
stt_provider = "faster_whisper"
stt_device = "cpu"
stt_compute_type = "float16"
stt_beam_size = 5
stt_timeout_seconds = 60
vad_filter = true
vad_min_silence_duration_ms = 500

# TTS
tts_provider = "piper_onnx"
piper_model = "en_US-john-medium"
tts_timeout_seconds = 30

# Limits
max_audio_size_bytes = 10485760
max_audio_duration_seconds = 300
rate_limit_per_minute = 10
rate_limit_window_seconds = 60

# Streaming
streaming_enabled = true
streaming_buffer_size = 4096

# Offline
offline_mode = true
preload_models = false

# Cache
enable_cache = true
cache_ttl_seconds = 3600
cache_max_entries = 1000
```

---

## Developer Integration

```python
from app.XNAi_rag_app.voice_interface import (
    VoiceInterface,
    VoiceConfig,
    STTProvider,
    TTSProvider,
    WakeWordDetector,
    AudioStreamProcessor,
    VoiceRateLimiter,
)

# Initialize with config
config = VoiceConfig(
    stt_provider=STTProvider.FASTER_WHISPER,
    tts_provider=TTSProvider.PIPER_ONNX,
    wake_word="hey nova",
    wake_word_enabled=True,
    wake_word_sensitivity=0.8,
    offline_mode=True,
)

voice = VoiceInterface(config)

# Check wake word
detector = WakeWordDetector(wake_word="hey nova", sensitivity=0.8)
detected, confidence = detector.detect("Hey Nova, hello")
# detected=True, confidence=0.95

# Rate limiting
limiter = VoiceRateLimiter(max_requests=10, window_seconds=60)
allowed, msg = limiter.allow_request("client_1")
# allowed=True

# Stream processing
stream = AudioStreamProcessor(config)
is_speech = stream.add_chunk(audio_data)
```


---

## Getting Started (5 minutes)

### Step 1: Install Voice Dependencies

```bash
cd /home/arcana-novai/Documents/GitHub/Xoe-NovAi

# Install voice packages
pip install -r requirements-chainlit.txt

# Optional: High-accuracy speech recognition
pip install openai
export OPENAI_API_KEY="sk-your-key-here"

# Optional: Premium voice synthesis
pip install elevenlabs
export ELEVENLABS_API_KEY="sk_your-key-here"
```

### Step 2: Disable Telemetry (Recommended)

```bash
export CHAINLIT_NO_TELEMETRY=true
```

### Step 3: Start Chainlit

```bash
chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001
```

The app will start and open automatically at `http://localhost:8001`

### Step 4: Select a Chat Profile

Choose one of three profiles:

- **🎤 Voice Assistant**: Voice-first interaction (mic button always ready)
- **📚 Library Curator**: Search for books/papers (text or voice)
- **🔍 Research Helper**: Find academic papers (voice or text)

### Step 5: Use Voice

Click the **🎤 microphone button** in the chat to:
1. **Record** your voice command
2. **I transcribe** it to text
3. **Process** your request
4. **Respond** with text and audio

---

## Voice Command Examples

### Library Searching (Say These)

> "Find all works by Plato"

> "Research quantum mechanics and give me top 10 recommendations"

> "Show me popular science fiction novels"

> "What are the best resources on machine learning?"

### Voice Control (Say These)

> "Speak slower" ← Reduces speech rate

> "Speak faster" ← Increases speech rate

> "Higher pitch" ← Makes voice higher

> "Lower your pitch" ← Makes voice lower

> "Louder" / "Quieter" ← Volume control

> "Switch to Spanish" ← Change language

---

## Voice Settings

### Speech Speed

- `0.5x` - Half speed (for processing time)
- `1.0x` - **Normal speed** (default)
- `1.5x` - 50% faster
- `2.0x` - Double speed

Command: "speak slower" / "speak faster"

### Pitch (Voice Height)

- `0.5` - Very low (bass-heavy hearing)
- `1.0` - **Normal** (default)
- `1.5` - High (treble sensitivity)
- `2.0` - Very high

Command: "use a higher pitch" / "lower your pitch"

### Volume

- `30%` - Quiet (sensitive hearing)
- `80%` - **Normal** (default)
- `100%` - Full volume

Command: "quieter" / "louder"

### Language (12 Options)

- 🇺🇸 English (US): `en-US`
- 🇬🇧 English (UK): `en-GB`
- 🇪🇸 Spanish: `es-ES`
- 🇫🇷 French: `fr-FR`
- 🇩🇪 German: `de-DE`
- 🇯🇵 Japanese: `ja-JP`
- 🇨🇳 Chinese (Simplified): `zh-CN`
- 🇹🇼 Chinese (Traditional): `zh-TW`
- 🇧🇷 Portuguese (Brazil): `pt-BR`
- 🇮🇹 Italian: `it-IT`
- 🇰🇷 Korean: `ko-KR`

Command: "switch to Spanish" / "use Japanese"

---

## Voice Providers (How It Works)

### Speech-to-Text (STT)

**Web Speech API (Default - Browser-based)**
- No server load
- Works in browser
- Real-time transcription
- ~85-92% accuracy
- No cost

**OpenAI Whisper (Optional - High accuracy)**
- 95-98% accuracy
- Server-side processing
- Better with accents/noise
- Requires API key
- Cost: $0.02/minute

Setup:
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

### Text-to-Speech (TTS)

**pyttsx3 (Default - Local/Offline)**
- No network needed
- Private (no data sent anywhere)
- Works on Windows/Mac/Linux
- ⚠️ Voice quality varies by OS
- No cost

**Google TTS (Free online)**
- Good quality
- Requires internet
- Free tier available
- Natural sounding voices
- Slight latency (~200-500ms)

**ElevenLabs (Premium - Best quality)**
- Highest quality voices
- Most natural sounding
- Requires paid API
- No latency
- Cost: ~$5-30/month

Setup:
```bash
export ELEVENLABS_API_KEY="sk_..."
```

---

## Accessibility Features

Perfect for disabled users:

### For Vision-Impaired Users

- ✅ Voice input (speak instead of type)
- ✅ Voice output (hear responses)
- ✅ Large, clear font options
- ✅ Screen reader compatible
- ✅ High contrast themes

### For Hearing-Impaired Users

- ✅ Text transcription of voice input
- ✅ Visual indicators (no audio needed)
- ✅ Adjustable text size
- ✅ Real-time captions

### For Cognitive Disabilities

- ✅ Slower speech (0.5x to 1.0x)
- ✅ Simpler language options
- ✅ Step-by-step guidance
- ✅ Clear voice (adjustable pitch)
- ✅ Audio feedback on actions

### For Motor Disabilities

- ✅ Voice commands (no typing needed)
- ✅ Hands-free interaction
- ✅ Eye-gaze control (future)
- ✅ Touch/gesture control (future)

---

## Future Features (Roadmap)

### Phase 2 (Q1 2026): Full Computer Control

Control your computer entirely by voice:

- "Open Firefox browser"
- "Download a book from Project Gutenberg"
- "Go to /home/username/Downloads"
- "Open the PDF I just downloaded"
- "Save this file to Documents"
- "Create a new folder"

### Phase 3 (Q2 2026): Accessibility Suite

- Custom voice profiles per user
- Screen reader integration
- Voice-only mode
- Eye-gaze + voice control
- Personalized shortcuts

### Phase 4 (Q3 2026): Multi-Modal Agent

- Voice + gesture + eye-gaze
- Intelligent context awareness
- Learning user preferences
- Natural language to code
- Hands-free computing

---

## Troubleshooting

### No 🎤 Button Visible

**Solution:**
- Use Chainlit 2.8.3+ (check: `pip show chainlit`)
- Use modern browser (Chrome, Firefox, Edge)
- HTTPS required for audio permissions (localhost is OK)

### Audio Not Working

**Solution:**
1. Check microphone permissions in browser
2. Click 🎤 and allow microphone access
3. Test with "Speak slower" command first
4. Check volume levels

### Transcription Failing

**Solution:**
1. Try switching STT provider (Settings)
2. Speak more slowly
3. Reduce background noise
4. Try Whisper for accuracy: `pip install openai && export OPENAI_API_KEY="..."`

### Voice Sounds Robotic

**Solution:**
1. Try different TTS provider:
   - pyttsx3: Check system voice settings
   - GTTS: Better quality (install: `pip install gtts`)
   - ElevenLabs: Best quality (requires API key)
2. Adjust pitch/speed: "speak slower", "higher pitch"

### High Latency/Slow Response

**Solution:**
1. Use Web Speech API (browser-based, faster)
2. Use pyttsx3 TTS (local, offline)
3. Reduce background network usage
4. Check internet connection

---

## Developers: Integration Guide

### Basic Usage in Code

```python
from app.XNAi_rag_app.voice_interface import (
    VoiceConfig,
    VoiceProvider,
    setup_voice_interface,
    get_voice_interface,
)

# Setup with custom config
config = VoiceConfig(
    stt_provider=VoiceProvider.WEB_SPEECH,
    tts_provider=VoiceProvider.GTTS,
    language="es-ES",
    speech_rate=0.8,  # Slower
)

setup_voice_interface(config)
voice = get_voice_interface()

# Get config
settings = voice.get_session_stats()
print(f"Total recordings: {settings['stats']['total_recordings']}")
```

### Chainlit Integration

```python
import chainlit as cl
from app.XNAi_rag_app.voice_interface import (
    process_voice_input,
    generate_voice_output,
)

@cl.on_audio_chunk
async def handle_audio(chunk: cl.AudioChunk):
    # Transcribe
    text = await process_voice_input(chunk.data)
    
    # Respond with voice
    response = f"I heard: {text}"
    audio = await generate_voice_output(response)
    
    if audio:
        await cl.Audio(data=audio, name="response.wav").send()
    await cl.Message(response).send()
```

See [docs/VOICE_INTERFACE_GUIDE.md](../docs/VOICE_INTERFACE_GUIDE.md) for detailed developer documentation.

---

## Performance Tips

### For Faster Response

- ✅ Use Web Speech API (browser-based STT)
- ✅ Use pyttsx3 TTS (local, no network)
- ✅ Shorter speech rate (0.7x - 1.0x)
- ✅ Reduce background processes

### For Better Accuracy

- ✅ Use Whisper STT (95-98% accurate)
- ✅ Reduce background noise
- ✅ Speak clearly and slowly
- ✅ Use ElevenLabs TTS for natural output

### For Privacy

- ✅ Use Web Speech API (stays in browser)
- ✅ Use pyttsx3 TTS (local processing)
- ✅ No API keys needed
- ✅ No data sent to external services

---

## Deployment

### Docker Deployment

Voice dependencies are already included in `requirements-chainlit.txt`:

```bash
docker-compose up -d
```

Voice will be available at `http://localhost:8001` automatically.

### Custom Deployment

```bash
# Install all dependencies
pip install -r requirements-chainlit.txt

# Optional speech recognition packages
pip install openai elevenlabs

# Run app
chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001
```

---

## Support & Feedback

For issues, feature requests, or accessibility needs:

1. Check troubleshooting section above
2. Review [VOICE_INTERFACE_GUIDE.md](../docs/VOICE_INTERFACE_GUIDE.md) for detailed docs
3. Test voice features directly: `python3 -c "from app.XNAi_rag_app.voice_interface import *; print('Voice module ready!')"`
4. Contact: Xoe-NovAi Team

---

## Next Steps

**Start Using Voice:**
1. Run: `chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001`
2. Open: `http://localhost:8001`
3. Click 🎤 button
4. Speak naturally!

**Learn More:**
- [Voice Interface Guide](../docs/VOICE_INTERFACE_GUIDE.md) - Comprehensive documentation
- [Chainlit Docs](https://docs.chainlit.io) - UI framework documentation
- [Whisper Docs](https://platform.openai.com/docs/guides/speech-to-text) - STT accuracy
- [ElevenLabs Docs](https://elevenlabs.io/docs) - Premium voices

---

**Ready to experience voice-enabled library curation with accessibility for all users!** 🎤📚♿
