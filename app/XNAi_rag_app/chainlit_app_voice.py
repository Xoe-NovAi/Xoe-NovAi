"""
Chainlit RAG App with Voice Interface
====================================

Chainlit application with the voice interface implementation (renamed from enterprise variant).
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import chainlit as cl
    from chainlit.input_widget import Select, Slider
except ImportError:
    cl = None
    print("Warning: Chainlit not installed")

try:
    from voice_interface import (
        VoiceInterface,
        VoiceConfig,
        STTProvider,
        TTSProvider,
        WhisperModel_,
        setup_voice_interface,
        get_voice_interface,
    )
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Warning: voice_interface not available")

try:
    from voice_command_handler import (
        VoiceCommandHandler,
        VoiceCommandParser,
        VoiceCommandOrchestrator,
        VoiceCommandType,
    )
    COMMANDS_AVAILABLE = True
except ImportError:
    COMMANDS_AVAILABLE = False
    print("Warning: Voice command handler not available")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["CHAINLIT_NO_TELEMETRY"] = "true"

# Global state
_voice_interface: Optional[VoiceInterface] = None
_command_handler: Optional[object] = None
_command_parser: Optional[object] = None


SYSTEM_PERSONAS = {
    "voice_assistant": {
        "name": "Voice Assistant",
        "description": "Conversational voice-enabled assistant with FAISS knowledge vault",
        "system_prompt": """You are Xoe, an intelligent voice-enabled assistant with access to a personal knowledge vault (FAISS).""",
        "color": "blue",
    }
}


if cl:
    @cl.on_chat_start
    async def on_chat_start():
        logger.info("Chat session started")
        await setup_voice_interface()

    @cl.on_audio_chunk
    async def on_audio_chunk(audio_chunk: cl.AudioChunk):
        logger.info(f"Received audio chunk: {len(audio_chunk.data)} bytes")
        voice_interface = cl.user_session.get("voice_interface")
        if not voice_interface:
            await cl.Message(content="⚠️ Voice interface not initialized").send()
            return
        msg = await cl.Message(content="🎤 Listening...", disable_human_feedback=True).send()
        try:
            transcription, confidence = await voice_interface.transcribe_audio(audio_chunk.data)
            await msg.update(content=f"📝 **You said:** *{transcription}*\n\n(Confidence: {confidence:.1%})")
            await cl.Message(content=transcription).send()
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            await msg.update(content=f"❌ Audio processing failed: {str(e)}")


async def setup_voice_interface():
    global _voice_interface, _command_parser, _command_handler
    logger.info("Setting up voice interface...")
    try:
        if not VOICE_AVAILABLE:
            logger.warning("Voice interface not available - skipping setup")
            return
        config = VoiceConfig(
            stt_provider=STTProvider.FASTER_WHISPER,
            whisper_model=WhisperModel_.DISTIL_LARGE,
            tts_provider=TTSProvider.PIPER_ONNX,
            language="en",
            language_code="en",
            faiss_enabled=True,
            faiss_top_k=3,
            enable_voice_commands=True,
            enable_logging=True,
        )
        _voice_interface = VoiceInterface(config)
        if COMMANDS_AVAILABLE:
            _command_parser = VoiceCommandParser(confidence_threshold=0.6)
            _command_handler = VoiceCommandHandler(confirmation_required=True)
        logger.info("✓ Voice interface initialized")
    except Exception as e:
        logger.error(f"Failed to initialize voice interface: {e}")
        raise


if __name__ == "__main__":
    if VOICE_AVAILABLE:
        asyncio.run(setup_voice_interface())
    else:
        logger.error("Voice interface not available")
