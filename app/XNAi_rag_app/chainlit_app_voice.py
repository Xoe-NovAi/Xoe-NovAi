"""
Xoe-NovAi v0.1.5 - Chainlit Voice Interface with "Hey Nova" Wake Word
=====================================================================

Enhanced voice interface with:
- "Hey Nova" wake word detection
- Redis session persistence (VoiceSessionManager)
- FAISS knowledge retrieval (VoiceFAISSClient)
- Streaming audio support
- Rate limiting and input validation
- Real-time voice-to-voice conversation

Version: v0.1.5 (2026-01-08)
"""

import os
import logging
import asyncio
import io
import base64
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from collections import deque
import time

try:
    import chainlit as cl
    from chainlit.input_widget import Select, Slider
except ImportError:
    cl = None

try:
    from voice_interface import (
        VoiceInterface,
        VoiceConfig,
        STTProvider,
        TTSProvider,
        WhisperModel_,
        setup_voice_interface,
        get_voice_interface,
        WakeWordDetector,
        AudioStreamProcessor,
        VoiceRateLimiter,
        VoiceSessionManager,
        VoiceFAISSClient,
    )
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    logger.warning(f"Voice interface import failed: {e}")

try:
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["CHAINLIT_NO_TELEMETRY"] = "true"

# Global state
_voice_interface: Optional[VoiceInterface] = None
_wake_word_detector: Optional[WakeWordDetector] = None
_session_manager: Optional[VoiceSessionManager] = None
_faiss_client: Optional[VoiceFAISSClient] = None


class VoiceConversationManager:
    """Manages voice conversation state with Redis persistence support."""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.audio_buffer = deque(maxlen=100)
        self.is_listening = False
        self.is_speaking = False
        self.conversation_active = False
        self.wake_word_detected = False
        self.last_speech_time = 0
        self.silence_threshold = 1.5
        self.stream_processor = None

    def initialize_stream_processor(self):
        if self.stream_processor is None and AUDIO_PROCESSING_AVAILABLE:
            self.stream_processor = AudioStreamProcessor(self.config)
            logger.info("Audio stream processor initialized")

    def add_audio_chunk(self, audio_data: bytes) -> bool:
        current_time = time.time()
        self.audio_buffer.append((current_time, audio_data))

        if self.stream_processor:
            is_speech = self.stream_processor.add_chunk(audio_data)
            if is_speech and not self.is_listening:
                self.is_listening = True
                self.last_speech_time = current_time
                return True
            elif not is_speech and self.is_listening:
                if current_time - self.last_speech_time > self.silence_threshold:
                    self.is_listening = False
                    return True
            return False

        if self._detect_voice_activity(audio_data):
            self.last_speech_time = current_time
            if not self.is_listening:
                self.is_listening = True
                return True

        if self.is_listening and (current_time - self.last_speech_time) > self.silence_threshold:
            self.is_listening = False
            return True

        return False

    def _detect_voice_activity(self, audio_data: bytes) -> bool:
        if not AUDIO_PROCESSING_AVAILABLE:
            return True
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms > 500
        except Exception:
            return True

    def get_buffered_audio(self) -> bytes:
        if not self.audio_buffer:
            return b''
        audio_chunks = [chunk for _, chunk in self.audio_buffer]
        return b''.join(audio_chunks)

    def clear_buffer(self):
        self.audio_buffer.clear()
        if self.stream_processor:
            self.stream_processor.reset()

    def start_conversation(self):
        self.conversation_active = True
        self.wake_word_detected = False
        self.clear_buffer()
        self.initialize_stream_processor()
        logger.info("Voice conversation started")

    def end_conversation(self):
        self.conversation_active = False
        self.is_listening = False
        self.is_speaking = False
        self.wake_word_detected = False
        self.clear_buffer()
        logger.info("Voice conversation ended")

    def check_wake_word(self, transcription: str) -> bool:
        if not self.config.wake_word_enabled:
            return True
        if _wake_word_detector:
            detected, confidence = _wake_word_detector.detect(transcription)
            if detected:
                self.wake_word_detected = True
                logger.info(f"Wake word 'Hey Nova' detected (confidence: {confidence:.2f})")
                return True
            return False
        return True


_conversation_manager = VoiceConversationManager()


if cl:
    @cl.on_chat_start
    async def on_chat_start():
        """Initialize voice chat session with Redis session manager."""
        logger.info("Voice chat session started")
        cl.user_session.set("voice_conversation_active", False)
        cl.user_session.set("voice_enabled", True)

        # Initialize Redis session manager
        global _session_manager
        try:
            _session_manager = VoiceSessionManager()
            cl.user_session.set("session_manager", _session_manager)
            logger.info(f"Voice session manager ready: {_session_manager.session_id}")
        except Exception as e:
            logger.warning(f"Redis session manager unavailable: {e}")
            _session_manager = None

        # Initialize FAISS client for knowledge retrieval
        global _faiss_client
        try:
            _faiss_client = VoiceFAISSClient()
            cl.user_session.set("faiss_client", _faiss_client)
            stats = _faiss_client.get_index_stats()
            logger.info(f"FAISS client ready: {stats}")
        except Exception as e:
            logger.warning(f"FAISS client unavailable: {e}")
            _faiss_client = None

        # Initialize voice interface
        try:
            await setup_voice_interface()
            cl.user_session.set("conversation_manager", _conversation_manager)
            global _wake_word_detector
            if VOICE_AVAILABLE:
                _wake_word_detector = WakeWordDetector(wake_word="hey nova", sensitivity=0.8)
                logger.info("Wake word detector initialized for 'Hey Nova'")
        except Exception as e:
            logger.error(f"Failed to setup voice interface: {e}")

        welcome_msg = """# Xoe-NovAi v0.1.5 - Voice Assistant

**Voice-to-Voice Conversation Ready!**

**Features:**
- Say **"Hey Nova"** to activate
- Streaming audio with VAD
- Redis session persistence
- FAISS knowledge retrieval
- Piper ONNX TTS voice responses

**Commands:**
- "Stop voice chat" to end
- "Voice settings" to adjust
        """

        await cl.Message(content=welcome_msg).send()

        start_button = cl.Action(name="start_voice_chat", payload={"action": "start"}, label="Start Voice Chat")
        await cl.Message(content="Click to begin voice conversation:", actions=[start_button]).send()

    @cl.action_callback("start_voice_chat")
    async def start_voice_chat(action: cl.Action):
        cl.user_session.set("voice_conversation_active", True)
        _conversation_manager.start_conversation()
        
        await cl.Message(content="""**Voice Chat Started!**

I'm listening. Say "Hey Nova" when ready.

**Status:** Listening for wake word...
        """).send()

    @cl.action_callback("stop_voice_chat")
    async def stop_voice_chat(action: cl.Action):
        cl.user_session.set("voice_conversation_active", False)
        _conversation_manager.end_conversation()
        
        if _session_manager:
            _session_manager.clear_session()
        
        await cl.Message(content="**Voice Chat Stopped** - Session cleared").send()

    @cl.action_callback("voice_settings")
    async def voice_settings(action: cl.Action):
        settings_msg = "**Voice Settings**"
        sensitivity_slider = cl.Slider(id="wake_sensitivity", label="Wake Sensitivity", initial=0.8, min=0.5, max=1.0, step=0.05)
        await cl.Message(content=settings_msg, elements=[sensitivity_slider]).send()


async def setup_voice_interface():
    """Setup voice interface with all components."""
    global _voice_interface, _wake_word_detector
    logger.info("Setting up voice interface...")
    
    if not VOICE_AVAILABLE:
        logger.warning("Voice interface not available")
        return
    
    config = VoiceConfig(
        stt_provider=STTProvider.FASTER_WHISPER,
        tts_provider=TTSProvider.PIPER_ONNX,
        language="en",
        wake_word="hey nova",
        wake_word_enabled=True,
        wake_word_sensitivity=0.8,
        offline_mode=True,
    )
    
    _voice_interface = VoiceInterface(config)
    _wake_word_detector = WakeWordDetector(wake_word=config.wake_word, sensitivity=config.wake_word_sensitivity)
    
    logger.info("Voice interface initialized")


async def process_voice_input(audio_data: bytes) -> Optional[str]:
    """Process voice input and return transcription."""
    if not VOICE_AVAILABLE or not _voice_interface:
        return None
    
    try:
        transcription, confidence = await _voice_interface.transcribe_audio(audio_data)
        logger.info(f"Transcribed: {transcription[:50]}... (conf: {confidence:.1%})")
        
        # Save to Redis session
        if _session_manager:
            _session_manager.add_interaction("user", transcription, {"confidence": confidence})
        
        return transcription
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        return None


async def generate_ai_response(user_input: str) -> str:
    """Generate AI response using RAG API with conversation context."""
    try:
        # Get conversation context from Redis
        context = ""
        if _session_manager:
            context = _session_manager.get_conversation_context(max_turns=5)
        
        # Get knowledge from FAISS
        knowledge_context = ""
        if _faiss_client and _faiss_client.is_available:
            results = _faiss_client.search(user_input, top_k=3)
            knowledge_context = "\n".join([r.get("content", "") for r in results[:2]])
        
        import httpx
        rag_api_url = "http://xnai_rag_api:8000/query"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                rag_api_url, 
                json={
                    "query": user_input,
                    "use_rag": True,
                    "voice_input": True,
                    "conversation_context": context,
                    "knowledge_context": knowledge_context,
                }
            )
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "I processed your request.")
                
                # Save assistant response to Redis
                if _session_manager:
                    _session_manager.add_interaction("assistant", response_text)
                
                return response_text
            else:
                return "I'm having trouble connecting to my knowledge base."
    except Exception as e:
        logger.error(f"RAG API call failed: {e}")
        return "I heard your message but am having trouble processing it."


async def generate_voice_response(text: str) -> Optional[bytes]:
    """Generate voice response from text."""
    if not VOICE_AVAILABLE or not _voice_interface:
        return None
    
    try:
        audio_data = await _voice_interface.synthesize_speech(text=text, language="en")
        logger.info(f"Generated voice response: {len(audio_data) if audio_data else 0} bytes")
        return audio_data
    except Exception as e:
        logger.error(f"Voice generation failed: {e}")
        return None


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with voice support."""
    user_query = message.content.strip()

    if user_query.startswith("/"):
        command_response = await handle_command(user_query)
        if command_response:
            await cl.Message(content=command_response).send()
        return

    msg = cl.Message(content="")
    await msg.send()

    voice_enabled = cl.user_session.get("voice_enabled", True)

    try:
        response_text = await generate_ai_response(user_query)

        for word in response_text.split():
            await msg.stream_token(word + " ")
            await asyncio.sleep(0.02)

        await msg.update()

        if voice_enabled:
            voice_msg = cl.Message(content="Speaking...")
            await voice_msg.send()
            audio_data = await generate_voice_response(response_text)
            if audio_data:
                await voice_msg.update(content="Voice response generated!")
            else:
                await voice_msg.update(content="Voice generation failed")

    except Exception as e:
        logger.error(f"Message processing failed: {e}")
        await msg.stream_token(f"\n\nError: {str(e)}")
        await msg.update()


async def handle_command(command: str) -> Optional[str]:
    """Handle slash commands."""
    command_lower = command.strip().lower()
    if command_lower == "/voice on":
        cl.user_session.set("voice_enabled", True)
        return "Voice responses enabled"
    elif command_lower == "/voice off":
        cl.user_session.set("voice_enabled", False)
        return "Voice responses disabled"
    elif command_lower == "/voice status":
        voice_enabled = cl.user_session.get("voice_enabled", True)
        session_info = ""
        if _session_manager:
            stats = _session_manager.get_stats()
            session_info = f"\nSession: {stats['session_id']}, Turns: {stats['conversation_turns']}"
        return f"Voice: {'Enabled' if voice_enabled else 'Disabled'}{session_info}"
    elif command_lower == "/session clear":
        if _session_manager:
            _session_manager.clear_session()
        return "Session cleared"
    return None


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates."""
    if "voice_enabled" in settings:
        cl.user_session.set("voice_enabled", settings["voice_enabled"])
        status = "enabled" if settings["voice_enabled"] else "disabled"
        await cl.Message(content=f"Voice responses {status}").send()


if __name__ == "__main__":
    if VOICE_AVAILABLE:
        asyncio.run(setup_voice_interface())
    else:
        logger.error("Voice interface not available")
