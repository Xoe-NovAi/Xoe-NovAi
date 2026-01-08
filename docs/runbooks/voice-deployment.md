# Enterprise Voice Deployment & Integration Guide

**How to integrate Xoe-NovAi's enterprise voice system into your application**

---

## 📋 PRE-DEPLOYMENT CHECKLIST

```
Hardware Requirements:
- [ ] NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX 3060+ recommended)
- [ ] CUDA 12.x with cuBLAS and cuDNN 9 installed
- [ ] 12GB+ VRAM (16GB+ for production)
- [ ] 8GB+ system RAM minimum
- [ ] Microphone and speaker connected

Software Requirements:
- [ ] Python 3.9+
- [ ] pip package manager
- [ ] FFmpeg (for audio processing)
- [ ] Git (for version control)

Network/Optional:
- [ ] Internet for model downloads (first run)
- [ ] HTTPS for production (Chainlit)
```

---

## 📦 STEP 1: INSTALLATION

### 1.1 Clone/Setup Repository

```bash
cd /path/to/Xoe-NovAi
git pull origin main  # Get latest version
```

### 1.2 Install Dependencies

```bash
# Install all voice system dependencies
pip install -r requirements-chainlit.txt

# Verify CUDA installation
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU details
nvidia-smi
```

### 1.3 Verify Installation

```bash
# Run full test suite (takes 2-3 minutes)
python test_enterprise_voice.py

# Expected output:
# ✓ ALL TESTS PASSED - ENTERPRISE VOICE SYSTEM READY
```

---

## 🔗 STEP 2: INTEGRATE WITH RAG PIPELINE

### 2.1 Import Voice System

```python
# In your RAG application
from app.XNAi_rag_app.voice_interface import (
    EnterpriseVoiceConfig,
    EnterpriseVoiceInterface,
    setup_enterprise_voice,
)

from app.XNAi_rag_app.voice_command_handler import (
    VoiceCommandHandler,
    VoiceCommandParser,
)
```

### 2.2 Initialize Voice System

```python
import asyncio

async def initialize_voice_system():
    """Initialize voice system on app startup"""
    
    # Create configuration
    config = EnterpriseVoiceConfig(
        language="en",
        faiss_enabled=True,
        enable_voice_commands=True,
    )
    
    # Initialize interface
    voice_interface = EnterpriseVoiceInterface(config)
    
    # Initialize command handler
    command_parser = VoiceCommandParser()
    command_handler = VoiceCommandHandler(
        faiss_index=your_faiss_index,  # Your FAISS instance
        embeddings_model=your_embeddings,  # Your embeddings model
        confirmation_required=True,
    )
    
    return voice_interface, command_handler

# On app startup
voice_interface, command_handler = asyncio.run(initialize_voice_system())
```

### 2.3 Connect to Existing FAISS Index

```python
from voice_command_handler import VoiceCommandHandler

# Assuming you have a FAISS index from curator
def setup_voice_commands(faiss_index, embeddings_model):
    """Setup voice commands with existing FAISS"""
    
    handler = VoiceCommandHandler(
        faiss_index=faiss_index,
        embeddings_model=embeddings_model,
        confirmation_required=True,
    )
    
    return handler

# In your RAG app
voice_command_handler = setup_voice_commands(
    faiss_index=curator.index,
    embeddings_model=curator.embeddings_model,
)
```

---

## 🎤 STEP 3: PROCESS VOICE INPUT

### 3.1 From Microphone

```python
import asyncio
import pyaudio
import numpy as np

async def capture_and_process():
    """Capture audio from microphone and transcribe"""
    
    # Record audio (assuming pyaudio setup)
    CHUNK = 1024
    FORMAT = 8  # 16-bit
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 10
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                   input=True, frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to bytes
    import io
    import wave
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(b''.join(frames))
    
    # Transcribe
    text, confidence = await voice_interface.transcribe_audio(
        audio_bytes.getvalue()
    )
    
    return text, confidence
```

### 3.2 From Web (Chainlit)

```python
# Already handled in chainlit_app_voice.py
# But here's the pattern:

import chainlit as cl
from voice_interface import get_voice_interface

@cl.on_audio_chunk
async def on_audio_chunk(audio_chunk: cl.AudioChunk):
    """Handle audio from Chainlit UI"""
    
    voice = get_voice_interface()
    
    # Transcribe
    text, confidence = await voice.transcribe_audio(audio_chunk.data)
    
    # Show to user
    await cl.Message(content=f"📝 You said: *{text}*").send()
    
    # Process as regular message
    await cl.Message(content=text).send()
```

### 3.3 From File

```python
async def transcribe_file(filepath: str):
    """Transcribe audio file"""
    
    # Read audio file
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
    
    # Transcribe
    text, confidence = await voice_interface.transcribe_audio(audio_bytes)
    
    return text, confidence

# Usage
text, conf = await transcribe_file("/path/to/audio.wav")
```

---

## 🎙️ STEP 4: ROUTE TO VOICE COMMANDS

### 4.1 Command Detection & Execution

```python
async def process_voice_input(transcription: str):
    """Process transcribed text for voice commands"""
    
    # Parse command
    parsed = command_parser.parse(transcription)
    
    # Check confidence
    if parsed.confidence < 0.6:
        return {
            "status": "unclear",
            "message": "I didn't understand that. Could you repeat?",
        }
    
    # Execute handler
    result = await command_handler.process_command(
        transcription,
        auto_confirm=False,  # Require user confirmation
    )
    
    return result

# Usage
result = await process_voice_input("Insert my research notes")
print(result["message"])  # → "Saved to vault: my research notes"
```

### 4.2 Chainlit Integration Example

```python
@cl.on_message
async def on_message(message: cl.Message):
    """Process messages including voice commands"""
    
    voice = get_voice_interface()
    parser = get_command_parser()
    handler = get_command_handler()
    
    # Check if message came from voice (has audio transcription)
    if hasattr(message, 'is_from_voice') and message.is_from_voice:
        # Try voice command
        parsed = parser.parse(message.content)
        
        if parsed.command_type.value != "unknown":
            # Execute as voice command
            result = await handler.process_command(message.content)
            
            await cl.Message(
                content=f"🎯 {result['message']}"
            ).send()
            
            return
    
    # Process as regular RAG query
    response = await rag_system.query(message.content)
    await cl.Message(content=response).send()
```

---

## 🔊 STEP 5: GENERATE VOICE RESPONSES

### 5.1 Text-to-Speech

```python
async def generate_voice_response(text: str, language: str = "en"):
    """Generate audio response from text"""
    
    # Synthesize
    audio_bytes = await voice_interface.synthesize_speech(
        text=text,
        language=language,
    )
    
    if audio_bytes:
        # Save or stream audio
        return audio_bytes
    else:
        return None

# Usage
audio_response = await generate_voice_response("Here are your search results...")
```

### 5.2 Integration with RAG Response

```python
async def rag_query_with_voice(query: str):
    """Query RAG and return voice response"""
    
    # Get text response
    text_response = await rag_system.query(query)
    
    # Generate voice response
    audio_response = await voice_interface.synthesize_speech(
        text=text_response,
        language="en",
    )
    
    return {
        "text": text_response,
        "audio": audio_response,
    }
```

---

## 📊 STEP 6: MONITOR & OPTIMIZE

### 6.1 Performance Monitoring

```python
async def monitor_voice_performance():
    """Monitor voice system performance"""
    
    stats = voice_interface.get_session_stats()
    
    print("\n📊 VOICE SYSTEM STATISTICS")
    print("="*50)
    print(f"Session: {stats['session_id']}")
    print(f"\nConfiguration:")
    for key, value in stats['configuration'].items():
        print(f"  {key}: {value}")
    print(f"\nPerformance Metrics:")
    for key, value in stats['performance_metrics'].items():
        print(f"  {key}: {value}")
    
    # Alert if latency high
    avg_stt = float(stats['performance_metrics']['avg_stt_latency_ms'])
    if avg_stt > 2000:  # >2 seconds
        print("\n⚠️  WARNING: STT latency high! Consider using smaller model.")
    
    return stats

# Periodic monitoring
import asyncio
async def monitor_loop():
    while True:
        await monitor_voice_performance()
        await asyncio.sleep(60)  # Every 60 seconds
```

### 6.2 Error Handling & Recovery

```python
async def robust_transcribe(audio_bytes: bytes, retries: int = 3):
    """Transcribe with error recovery"""
    
    for attempt in range(retries):
        try:
            text, confidence = await voice_interface.transcribe_audio(audio_bytes)
            return text, confidence
        
        except Exception as e:
            logger.error(f"Transcription failed (attempt {attempt+1}): {e}")
            
            if attempt == retries - 1:
                # Last attempt failed
                raise
            
            # Wait before retry
            await asyncio.sleep(1)
    
    return None, 0.0
```

---

## 🚀 STEP 7: DOCKER DEPLOYMENT

### 7.1 Dockerfile Modification

```dockerfile
# Add to existing Dockerfile or create new one

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements-chainlit.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements-chainlit.txt

# Copy application
COPY app/ /app/app/
COPY docs/ /app/docs/

# Environment
ENV CUDA_VISIBLE_DEVICES=0
ENV CHAINLIT_NO_TELEMETRY=true

# Run Chainlit app
CMD ["chainlit", "run", "app/XNAi_rag_app/chainlit_app_voice.py", "-h", "--port", "8000"]
```

### 7.2 Docker Compose

```yaml
version: '3.8'

services:
  voice-service:
    build:
      context: .
      dockerfile: Dockerfile.voice
    
    container_name: xnai-voice
    
    runtime: nvidia
    
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CHAINLIT_NO_TELEMETRY=true
      - VOICE_DEVICE=cuda
      - VOICE_COMPUTE_TYPE=float16
    
    ports:
      - "8000:8000"
    
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 🔒 STEP 8: SECURITY CONSIDERATIONS

### 8.1 Input Validation

```python
from enum import Enum
import re

MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB

def validate_audio_input(audio_bytes: bytes) -> bool:
    """Validate audio input"""
    
    # Check size
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise ValueError("Audio file too large")
    
    # Check format (basic)
    if not (audio_bytes.startswith(b'RIFF') or audio_bytes.startswith(b'ID3')):
        raise ValueError("Invalid audio format")
    
    return True

def validate_text_input(text: str, max_length: int = 5000) -> bool:
    """Validate text input"""
    
    if len(text) > max_length:
        raise ValueError("Text too long")
    
    # Basic injection checks
    if any(dangerous in text for dangerous in ['<script>', 'DROP TABLE']):
        raise ValueError("Suspicious content detected")
    
    return True
```

### 8.2 Authentication

```python
# In Chainlit app
import chainlit as cl
from functools import wraps

@cl.password_auth_callback
def auth_user(username: str, password: str):
    """Authenticate user for voice system"""
    # Implement your auth logic
    if check_credentials(username, password):
        return cl.User(identifier=username, metadata={"role": "user"})
    else:
        return None

@cl.on_chat_start
async def start():
    """Only authenticated users can access"""
    user = cl.user_session.get("user")
    if not user:
        raise ValueError("Not authenticated")
    
    await setup_voice_interface()
```

---

## 📈 STEP 9: PERFORMANCE TUNING

### 9.1 Model Selection for Your Hardware

```python
# RTX 3060 (6GB) - Entry Level
config = EnterpriseVoiceConfig(
    whisper_model=WhisperModel_.BASE,
    stt_compute_type="int8",
)

# RTX 3070/4080 (8GB) - Balanced (RECOMMENDED)
config = EnterpriseVoiceConfig(
    whisper_model=WhisperModel_.DISTIL_LARGE,
    stt_compute_type="float16",
)

# RTX 3080/A6000 (24GB) - High Performance
config = EnterpriseVoiceConfig(
    whisper_model=WhisperModel_.LARGE_V3,
    stt_compute_type="float16",
    batch_processing=True,
    batch_size=8,
)

# A100 (80GB) - Enterprise
config = EnterpriseVoiceConfig(
    whisper_model=WhisperModel_.LARGE_V3,
    stt_compute_type="float32",
    batch_processing=True,
    batch_size=16,
)
```

### 9.2 Caching & Optimization

```python
# Cache TTS model between requests
_tts_cache = None

async def cached_synthesize(text: str):
    """Use cached TTS model"""
    global _tts_cache
    
    if _tts_cache is None:
        from TTS.api import TTS
        _tts_cache = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    # First call: 3-5s (model load), subsequent: <500ms
    return _tts_cache.tts(text=text)

# Batch processing for multiple requests
async def batch_transcribe(audio_list):
    """Process multiple audio files efficiently"""
    
    # Use batch processing for faster throughput
    config.batch_processing = True
    config.batch_size = 8
    
    # Process multiple at once
    results = await asyncio.gather(*[
        voice_interface.transcribe_audio(audio)
        for audio in audio_list
    ])
    
    return results
```

---

## ✅ DEPLOYMENT CHECKLIST

```
Infrastructure:
- [ ] GPU available and tested
- [ ] CUDA 12.x installed and verified
- [ ] Python 3.9+ installed
- [ ] All dependencies installed
- [ ] Models downloaded (first run, ~6GB)

Code Integration:
- [ ] Voice system imported correctly
- [ ] Configuration created
- [ ] FAISS integration connected
- [ ] Error handling implemented
- [ ] Logging configured

Testing:
- [ ] test_voice.py passes
- [ ] Microphone/speaker tested
- [ ] Sample audio transcribed
- [ ] Voice commands working
- [ ] FAISS operations verified
- [ ] Performance benchmarks acceptable
- [ ] Error scenarios handled
- [ ] Security checks passed

Deployment:
- [ ] Docker image built (if using containers)
- [ ] Environment variables set
- [ ] Health checks configured
- [ ] Monitoring setup
- [ ] Backup/recovery plan
- [ ] Documentation updated
- [ ] Team trained

Production:
- [ ] Load testing completed
- [ ] Performance metrics within SLA
- [ ] 24/7 monitoring active
- [ ] Incident response plan ready
```

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues During Deployment

| Issue | Solution |
|-------|----------|
| GPU memory error | Reduce model size or use int8 quantization |
| Slow transcription | Switch to distil-large-v3 or batch process |
| Audio quality issues | Check microphone settings, sample rate |
| High latency | Use GPU, enable fp16 mode, reduce batch size |
| Models won't download | Check internet, disk space, firewall |
| Chainlit connection issues | Check port 8000, firewall rules |

### Getting Help

1. Run diagnostics: `python test_enterprise_voice.py`
2. Check logs: `~/.chainlit/logs/app.log`
3. Consult guide: `docs/VOICE_ENTERPRISE_IMPLEMENTATION_GUIDE.py`
4. Review examples in source files

---

## 🎯 NEXT MILESTONES

- ✅ v0.2.0 - Voice (Current)
- 🔄 v0.3.0 - Open Voice Alternative
- 🚀 v0.4.0 - Computer Control
- 🌟 v1.0.0 - Full OS Integration

---

**Deployment Status**: ✅ Ready for Production  
**Last Updated**: January 3, 2026  
**Support**: See documentation files
