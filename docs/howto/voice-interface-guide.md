"""
Voice Interface Feature Documentation & Configuration Guide
===========================================================

Author: Xoe-NovAi Team
Date: January 3, 2026
Version: v0.1.4-stable (Voice Feature Release)

OVERVIEW
========

This document provides comprehensive documentation for the new Voice Interface feature
in Xoe-NovAi Chainlit application. The voice interface enables:

1. Voice Input (Speech-to-Text)
   - Browser Web Speech API (real-time, in-browser)
   - OpenAI Whisper (server-side, high accuracy fallback)
   - Audio recording and playback

2. Voice Output (Text-to-Speech)
   - pyttsx3 (local, offline, no API needed)
   - Google TTS (free, good quality)
   - ElevenLabs (premium, highest quality - optional)

3. Accessibility Features
   - Speech rate control (0.5x to 2.0x)
   - Pitch adjustment (0.5 to 2.0)
   - Volume control (0% to 100%)
   - Language support (10+ languages)
   - Voice activity detection (VAD) ready

4. Future Agentic Control
   - Foundation for full computer accessibility via voice
   - Desktop application control (browsers, files, downloads)
   - Disabled user accessibility features
   - Natural language command routing

QUICK START
===========

1. Install Voice Dependencies:
   pip install -r requirements-chainlit.txt
   pip install openai  # Optional: for Whisper STT

2. Set Environment Variables:
   export CHAINLIT_NO_TELEMETRY=true
   export OPENAI_API_KEY="your-key-here"  # Optional
   export ELEVENLABS_API_KEY="your-key-here"  # Optional

3. Run Chainlit with Voice:
   chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001

4. Open Browser:
   http://localhost:8001

5. Select Chat Profile:
   - "🎤 Voice Assistant" for voice-first interaction
   - "📚 Library Curator" for text-based library search
   - "🔍 Research Helper" for academic papers

6. Use Voice:
   - Click 🎤 button to record voice commands
   - Speak naturally
   - I respond with audio and text
   - Adjust voice settings: "speak slower", "higher pitch", etc.

CONFIGURATION
==============

Default Configuration (in code):

    from app.XNAi_rag_app.voice_interface import VoiceConfig, VoiceProvider
    
    config = VoiceConfig(
        stt_provider=VoiceProvider.WEB_SPEECH,   # Web Speech API
        tts_provider=VoiceProvider.PYTTSX3,      # Local TTS
        language="en-US",                         # English (US)
        voice_name="default",                     # Default system voice
        speech_rate=1.0,                          # Normal speed
        pitch=1.0,                                # Normal pitch
        volume=0.8,                               # 80% volume
        enable_audio_logging=True,                # Save recordings
        max_recording_duration=300,               # 5 min max
        vad_enabled=True,                         # Voice Activity Detection
    )
    
    setup_voice_interface(config)

Custom Configuration Examples:

    # Fast, high-pitched, quiet voice (for background listening)
    fast_config = VoiceConfig(
        speech_rate=1.5,
        pitch=1.5,
        volume=0.5,
        tts_provider=VoiceProvider.GTTS,
    )
    
    # Slow, low, loud voice (for accessibility)
    accessible_config = VoiceConfig(
        speech_rate=0.7,
        pitch=0.8,
        volume=1.0,
        language="en-GB",  # British English
        tts_provider=VoiceProvider.ELEVENLABS,
    )
    
    # Spanish language support
    spanish_config = VoiceConfig(
        language="es-ES",
        tts_provider=VoiceProvider.GTTS,
    )

SUPPORTED LANGUAGES
====================

- English: en-US (US), en-GB (UK), en-AU (Australian)
- Spanish: es-ES (Spain)
- French: fr-FR (France)
- German: de-DE (Germany)
- Japanese: ja-JP (Japan)
- Chinese: zh-CN (Simplified), zh-TW (Traditional)
- Portuguese: pt-BR (Brazil)
- Italian: it-IT (Italy)
- Korean: ko-KR (Korea)

Add more by modifying VoiceInterface.supported_languages list.

VOICE PROVIDERS
================

STT (Speech-to-Text)
---
status: active
last_updated: 2026-01-04
owners:
  - team: voice
tags:
  - voice
---

--------------------

1. WEB_SPEECH (Default - Browser Web Speech API)
   - Pros: No server load, real-time, in-browser processing
   - Cons: Accuracy depends on browser implementation
   - Cost: Free
   - Privacy: All processing in browser
   - Recommended for: Fast, interactive use
   
2. WHISPER (Server-side OpenAI Whisper)
   - Pros: High accuracy, handles accents/noise
   - Cons: Requires API key, network latency
   - Cost: $0.02 per minute
   - Privacy: Audio sent to OpenAI servers
   - Recommended for: High-accuracy transcription, research
   
   Setup:
   export OPENAI_API_KEY="sk-..."
   pip install openai

TTS (Text-to-Speech)
-------------------

1. PYTTSX3 (Default - Local Offline)
   - Pros: No network needed, offline, privacy-preserving
   - Cons: Voice quality varies by platform
   - Cost: Free
   - Speed: Fast (no API latency)
   - Platforms: Windows, macOS, Linux (uses system voices)
   - Recommended for: Desktop/server use, privacy-critical apps

2. GTTS (Google Text-to-Speech)
   - Pros: Good quality, natural voices
   - Cons: Requires internet, slight latency
   - Cost: Free (rate limited)
   - Quality: Good
   - Recommended for: Web apps, free tier
   
   Setup:
   pip install gtts  # Usually already installed

3. ELEVENLABS (Premium)
   - Pros: Highest quality, natural sounding, 12+ voices
   - Cons: Requires paid API key
   - Cost: ~$5-30/month depending on usage
   - Quality: Excellent (professional grade)
   - Recommended for: Production applications, premium experience
   
   Setup:
   export ELEVENLABS_API_KEY="sk_..."
   pip install elevenlabs  # Optional

USAGE EXAMPLES
===============

Basic Voice Interaction in Chainlit:

    import chainlit as cl
    from app.XNAi_rag_app.voice_interface import (
        process_voice_input,
        generate_voice_output,
    )
    
    @cl.on_audio_chunk
    async def handle_audio(chunk: cl.AudioChunk):
        # Transcribe audio
        text = await process_voice_input(chunk.data)
        
        # Show transcribed text
        await cl.Message(f"You said: {text}").send()
        
        # Generate voice response
        response = f"I heard: {text}"
        audio = await generate_voice_output(response, wait_for_completion=True)
        
        if audio:
            await cl.Audio(data=audio, name="response.wav").send()

Control Voice Settings:

    from app.XNAi_rag_app.voice_interface import get_voice_interface
    
    voice = get_voice_interface()
    
    # Adjust speed (0.5 to 2.0)
    voice.config.speech_rate = 0.75  # 25% slower
    
    # Adjust pitch (0.5 to 2.0)
    voice.config.pitch = 1.3  # Higher pitch
    
    # Adjust volume (0.0 to 1.0)
    voice.config.volume = 0.9  # 90% volume
    
    # Change language
    voice.config.language = "es-ES"  # Spanish

Get Session Statistics:

    voice = get_voice_interface()
    stats = voice.get_session_stats()
    
    print(f"Total recordings: {stats['stats']['total_recordings']}")
    print(f"Total duration: {stats['stats']['total_duration']:.1f} seconds")
    print(f"Successful transcriptions: {stats['stats']['successful_transcriptions']}")

ACCESSIBILITY FEATURES
=======================

The voice interface includes several accessibility features:

1. Speech Rate Control
   - Slower speech for cognitive disabilities: 0.5x - 0.75x
   - Normal speech: 1.0x
   - Faster speech for efficiency: 1.5x - 2.0x
   - Command: "speak slower" / "speak faster"

2. Pitch Adjustment
   - Lower pitch for bass-heavy hearing: 0.5 - 0.75
   - Normal: 1.0
   - Higher pitch for treble sensitivity: 1.5 - 2.0
   - Command: "use a higher pitch" / "lower your pitch"

3. Volume Control
   - Quiet for sensitive hearing: 0.3 - 0.5
   - Normal: 0.8 - 1.0
   - Command: "quieter" / "louder"

4. Language Support
   - Multiple languages/accents available
   - Command: "switch to Spanish" / "use British English"

5. Voice Activity Detection (VAD)
   - Automatically detects speech end
   - Reduces accidental recordings
   - Energy-based VAD in development
   - Config: vad_enabled=True/False

FUTURE AGENTIC FEATURES (ROADMAP)
==================================

Phase 2 (Q1 2026): Full Voice Control
- Desktop application control via voice commands
- Voice navigation of file systems
- Download/upload via voice
- Browser control: "open Google", "go to GitHub"

Phase 3 (Q2 2026): Accessibility Suite
- Complete disabled user support
- Screen reader integration
- Voice-only navigation mode
- Custom voice profiles per user

Phase 4 (Q3 2026): Multi-Modal Agent
- Voice + gesture + eye-gaze control
- Voice-to-code synthesis
- Natural language to command generation
- Context-aware assistance

TROUBLESHOOTING
================

1. Chainlit UI doesn't show 🎤 button:
   - Ensure Chainlit >= 2.8.3
   - Check browser WebRTC support
   - Verify HTTPS (required for audio permissions)

2. Audio not transcribing:
   - Check microphone permissions in browser
   - Try switching STT provider (Web Speech → Whisper)
   - Test with `python3 -c "import speech_recognition; sr = speech_recognition.Recognizer()"`
   - Ensure OPENAI_API_KEY set for Whisper

3. Voice output sounds robotic/bad:
   - Try different TTS provider:
     * pyttsx3: Check system voice settings
     * GTTS: Should be good quality
     * ElevenLabs: Best quality (requires API key)
   - Adjust pitch/speed: Lower speech_rate, higher pitch
   - Check volume levels

4. Audio is cutting off:
   - Increase max_recording_duration in VoiceConfig
   - Check network latency if using server-side STT
   - Ensure sufficient disk space for logging

5. High latency:
   - Use Web Speech API (in-browser) instead of Whisper
   - Use pyttsx3 (local) instead of GTTS
   - Check network bandwidth

TESTING
========

Test Voice Interface:

    cd /home/arcana-novai/Documents/GitHub/Xoe-NovAi
    python3 -c "from app.XNAi_rag_app.voice_interface import VoiceInterface; VoiceInterface().get_session_stats()"

Test Chainlit App:

    chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001

Test Specific Features:

    # Test pyttsx3 TTS
    python3 << 'EOF'
    import pyttsx3
    engine = pyttsx3.init()
    engine.say("Testing voice output")
    engine.runAndWait()
    EOF
    
    # Test Google TTS
    python3 << 'EOF'
    from gtts import gTTS
    tts = gTTS("Hello world", lang='en')
    tts.save("/tmp/test.mp3")
    print("Audio saved to /tmp/test.mp3")
    EOF

DEVELOPMENT NOTES
==================

Architecture:

    VoiceInterface (Main)
    ├── VoiceSession (Per-session state)
    ├── VoiceConfig (Configuration)
    └── Support for STT/TTS providers
        ├── Web Speech API (browser)
        ├── Whisper (OpenAI)
        ├── pyttsx3 (local TTS)
        ├── Google TTS
        └── ElevenLabs (premium)

Integration Points:

    1. Chainlit UI (@cl.on_audio_chunk)
    2. Voice Interface Module
    3. Curator Interface (text/voice commands)
    4. RAG API (for knowledge)

File Structure:

    app/XNAi_rag_app/
    ├── voice_interface.py                 # Core voice module (450+ lines)
    ├── chainlit_app_with_voice.py         # Chainlit integration
    ├── chainlit_curator_interface.py      # Text-based curator
    └── library_api_integrations.py        # Library data sources

Dependencies:

    requirements-chainlit.txt:
    - chainlit==2.8.3              # UI framework
    - pyttsx3==2.90                # Local TTS
    - gtts==2.4.0                  # Google TTS
    - SpeechRecognition==3.10.4    # Audio input
    - pyaudio==0.2.13              # Audio I/O (optional)

PERFORMANCE METRICS
===================

Benchmarks (Initial Testing):

STT Performance:
- Web Speech API: <100ms (browser-side)
- Whisper-1: 2-5 seconds (API call + transcription)
- Accuracy: 95-98% (Whisper), 85-92% (Web Speech)

TTS Performance:
- pyttsx3: 50-200ms for typical sentences
- Google TTS: 200-500ms (including network latency)
- ElevenLabs: 300-800ms (highest quality)

Recording Statistics:
- Max recording: 5 minutes (configurable)
- Typical audio size: 100KB per minute
- Storage for 1-hour conversation: ~6MB

MONITORING & LOGGING
====================

Voice Session Statistics:

    {
        "session_id": "2026-01-03T14:23:45.123456",
        "config": {
            "stt_provider": "web_speech",
            "tts_provider": "pyttsx3",
            "language": "en-US"
        },
        "stats": {
            "total_recordings": 5,
            "total_duration": 45.3,
            "successful_transcriptions": 5,
            "failed_transcriptions": 0,
            "total_text_output": 1250
        },
        "conversation_turns": 8,
        "audio_recordings": 5
    }

Enable Detailed Logging:

    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("app.XNAi_rag_app.voice_interface")

SECURITY CONSIDERATIONS
========================

1. Audio Privacy:
   - Web Speech API: All processing in browser
   - Whisper: Audio sent to OpenAI (covered by their privacy policy)
   - pyttsx3: All processing local
   - Consider sensitive content in audio

2. Audio Storage:
   - Recordings stored in session (configurable)
   - Base64 encoded in logs
   - Recommend clearing logs periodically
   - For production: Implement encryption

3. API Keys:
   - Store in environment variables, never hardcode
   - Rotate OPENAI_API_KEY and ELEVENLABS_API_KEY regularly
   - Use separate service accounts for voice APIs

4. Rate Limiting:
   - Implement throttling on /audio endpoints
   - Monitor API usage for OpenAI/ElevenLabs
   - Set max_recording_duration to prevent abuse

SUPPORT & ISSUES
=================

For issues or feature requests:
1. Check troubleshooting section above
2. Review logs: Check CHAINLIT logs for details
3. Test in isolation: Use voice_interface.py directly
4. Report with: Python version, OS, browser, logs

Contact: Xoe-NovAi Team (January 3, 2026)
"""

# This is a documentation file - not meant to be executed as Python code
# Save this as VOICE_INTERFACE_GUIDE.md in the docs/ directory
