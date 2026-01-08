"""
Chainlit App with Voice Interface Integration
==============================================

Complete Chainlit application with voice input/output capabilities.

Features:
- Voice input (browser Web Speech API + Whisper fallback)
- Voice output (text-to-speech with accessibility settings)
- Library curation commands (text or voice)
- Conversation history with audio logging
- Accessibility controls (speech rate, pitch, language)

To run:
    chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001

Environment Variables:
    CHAINLIT_NO_TELEMETRY=true         # Disable telemetry
    ELEVENLABS_API_KEY=<key>           # Optional: Premium voice synthesis
    OPENAI_API_KEY=<key>               # Optional: Whisper API key
    DEBUG=true                         # Enable debug logging

Author: Xoe-NovAi Team
Last Updated: 2026-01-03
"""

import logging
from typing import Optional

try:
    import chainlit as cl
except ImportError:
    print("❌ Chainlit not installed. Install with: pip install -r requirements-chainlit.txt")
    exit(1)

from app.XNAi_rag_app.chainlit_curator_interface import (
    get_curator_interface,
    process_curator_command,
)
from app.XNAi_rag_app.voice_interface import (
    setup_voice_interface,
    get_voice_interface,
    process_voice_input,
    generate_voice_output,
    get_voice_config,
    VoiceConfig,
    VoiceProvider,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CHAINLIT APP CONFIGURATION
# ============================================================================

@cl.set_chat_profiles
async def chat_profiles():
    """Define chat profiles for different interaction modes."""
    return [
        cl.ChatProfile(
            name="📚 Library Curator",
            markdown_description="AI curator for finding books, research papers, and media",
            icon="https://cdn-icons-png.flaticon.com/512/1940/1940405.png",
        ),
        cl.ChatProfile(
            name="🎤 Voice Assistant",
            markdown_description="Voice-based conversation with audio input/output",
            icon="https://cdn-icons-png.flaticon.com/512/3128/3128557.png",
        ),
        cl.ChatProfile(
            name="🔍 Research Helper",
            markdown_description="Help finding academic papers and research materials",
            icon="https://cdn-icons-png.flaticon.com/512/4436/4436481.png",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    
    # Initialize both curator and voice interfaces
    from app.XNAi_rag_app.chainlit_curator_interface import get_curator_interface
    from app.XNAi_rag_app.voice_interface import setup_voice_interface
    
    # Setup curator
    curator = get_curator_interface()
    
    # Setup voice with custom config
    voice_config = VoiceConfig(
        stt_provider=VoiceProvider.WEB_SPEECH,  # Browser Web Speech API
        tts_provider=VoiceProvider.PYTTSX3,      # Local TTS
        language="en-US",
        speech_rate=1.0,
        pitch=1.0,
        volume=0.8,
    )
    setup_voice_interface(voice_config)
    voice = get_voice_interface()
    
    # Get selected profile
    profile = cl.user_session.get("chat_profile")
    
    if profile == "🎤 Voice Assistant":
        welcome_msg = """
# 🎤 Voice Assistant

Welcome to Xoe-NovAi's Voice Interface! You can:

**Voice Interaction:**
- Click the 🎤 button to record voice commands
- Speak naturally and I'll respond with audio
- Adjust voice settings: speed, pitch, language

**What you can do:**
- "Find books by Shakespeare"
- "What are the latest machine learning papers?"
- "Show me science fiction recommendations"
- "Change the voice to sound slower"

**Voice Controls:**
- Click 🎤 to start recording
- I'll speak back responses automatically
- Use speed/pitch controls for accessibility

---
*Voice synthesis by pyttsx3. Powered by Whisper for transcription.*
        """
    elif profile == "📚 Library Curator":
        welcome_msg = """
# 📚 Library Curator

Welcome to your personal AI library curator! I help you discover and manage books, papers, and media.

**Ask me about:**
- Finding books by specific authors
- Researching topics
- Getting personalized recommendations
- Exploring different domains (fiction, academic, music, podcasts)

**Example commands:**
- "Find all works by Plato"
- "Research quantum mechanics and give me top 10 books"
- "Show me popular science fiction novels"
- "What are the best resources on machine learning?"

**You can also use voice!** Click the 🎤 button to give voice commands.

---
*Powered by 7 major library APIs + Xoe-NovAi enrichment engine*
        """
    else:  # Research Helper
        welcome_msg = """
# 🔍 Research Helper

Find academic papers, research materials, and scholarly resources.

**Ask me to:**
- Locate papers on specific topics
- Find research by particular authors
- Recommend related papers
- Search across multiple academic databases
- Export research collections

**Example commands:**
- "Find papers on quantum computing"
- "Show me research by Richard Feynman"
- "Recommend papers related to neural networks"
- "Create a collection on climate change"

**Voice enabled!** Use the 🎤 button for hands-free research.

---
*Access to multiple academic databases via Xoe-NovAi*
        """
    
    await cl.Message(welcome_msg).send()
    
    logger.info(f"✓ Chat session started - Profile: {profile}")
    logger.info(f"✓ Voice interface ready (TTS: {voice.config.tts_provider.value})")


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user messages - support text, curator commands, and voice transcriptions."""
    
    user_input = message.content.strip()
    profile = cl.user_session.get("chat_profile")
    
    # Check if this is a curator command (text keywords or voice detection)
    curator_keywords = [
        "find", "locate", "search", "research", "recommend", "suggest",
        "book", "author", "works", "by", "on", "about", "top",
        "curate", "collection", "show", "list", "discover"
    ]
    
    is_curator_command = any(kw in user_input.lower() for kw in curator_keywords)
    
    # Route based on profile and command type
    if profile == "🎤 Voice Assistant":
        await handle_voice_assistant(user_input)
    elif profile == "📚 Library Curator" and is_curator_command:
        await handle_curator_command(user_input)
    elif profile == "🔍 Research Helper":
        await handle_research_helper(user_input)
    else:
        await handle_general_chat(user_input, profile)


async def handle_voice_assistant(user_input: str):
    """Handle voice assistant mode."""
    
    msg = cl.Message("")
    msg.status = "⏳ Processing..."
    await msg.send()
    
    try:
        voice = get_voice_interface()
        
        # Check for voice control commands
        if any(kw in user_input.lower() for kw in ["slower", "faster", "speed", "pitch", "voice"]):
            response = await handle_voice_settings(user_input, voice)
        else:
            # Regular voice assistant response
            response = f"🎤 You said: *{user_input}*\n\n"
            response += "I'm listening! Try asking me to:\n"
            response += "- Find books about your interests\n"
            response += "- Get recommendations\n"
            response += "- Change voice settings\n"
            
            # Check if it's a curator-style command
            if any(kw in user_input.lower() for kw in ["find", "search", "book", "author"]):
                curator_result = await process_curator_command(user_input)
                response = curator_result
        
        msg.content = response
        msg.status = "✓ Complete"
        await msg.update()
        
        # Generate voice output
        voice_output = await generate_voice_output(response, wait_for_completion=False)
        if voice_output:
            logger.info(f"✓ Audio response generated ({len(voice_output)} bytes)")
        
    except Exception as e:
        logger.error(f"Error in voice assistant: {e}")
        msg.content = f"❌ Error: {str(e)}"
        msg.status = "✗ Error"
        await msg.update()


async def handle_voice_settings(user_input: str, voice) -> str:
    """Handle voice settings adjustment."""
    
    response = "🎤 Voice Settings:\n\n"
    
    # Parse voice control commands
    if "slower" in user_input.lower() or "slow down" in user_input.lower():
        voice.config.speech_rate = max(0.5, voice.config.speech_rate - 0.25)
        response += f"✓ Voice speed set to {voice.config.speech_rate:.2f}x\n"
    
    if "faster" in user_input.lower() or "speed up" in user_input.lower():
        voice.config.speech_rate = min(2.0, voice.config.speech_rate + 0.25)
        response += f"✓ Voice speed set to {voice.config.speech_rate:.2f}x\n"
    
    if "higher" in user_input.lower() or "higher pitch" in user_input.lower():
        voice.config.pitch = min(2.0, voice.config.pitch + 0.25)
        response += f"✓ Pitch set to {voice.config.pitch:.2f}\n"
    
    if "lower" in user_input.lower() or "lower pitch" in user_input.lower():
        voice.config.pitch = max(0.5, voice.config.pitch - 0.25)
        response += f"✓ Pitch set to {voice.config.pitch:.2f}\n"
    
    if "louder" in user_input.lower():
        voice.config.volume = min(1.0, voice.config.volume + 0.2)
        response += f"✓ Volume set to {voice.config.volume:.0%}\n"
    
    if "quieter" in user_input.lower() or "softer" in user_input.lower():
        voice.config.volume = max(0.1, voice.config.volume - 0.2)
        response += f"✓ Volume set to {voice.config.volume:.0%}\n"
    
    # Show current settings
    response += "\n**Current Settings:**\n"
    response += f"- Speed: {voice.config.speech_rate:.2f}x\n"
    response += f"- Pitch: {voice.config.pitch:.2f}\n"
    response += f"- Volume: {voice.config.volume:.0%}\n"
    
    return response


async def handle_curator_command(user_input: str):
    """Handle library curator commands."""
    
    msg = cl.Message("")
    msg.status = "⏳ Searching libraries..."
    await msg.send()
    
    try:
        result = await process_curator_command(user_input)
        msg.content = result
        msg.status = "✓ Complete"
        await msg.update()
        
        # Generate voice summary
        voice = get_voice_interface()
        summary = f"Found results for: {user_input}. Showing top results."
        await generate_voice_output(summary, wait_for_completion=False)
        
    except Exception as e:
        logger.error(f"Error processing curator command: {e}")
        msg.content = f"❌ Error: {str(e)}"
        msg.status = "✗ Error"
        await msg.update()


async def handle_research_helper(user_input: str):
    """Handle research helper mode."""
    
    msg = cl.Message("")
    msg.status = "🔍 Searching academic databases..."
    await msg.send()
    
    try:
        # Route research commands to curator
        research_input = f"research papers on {user_input}" if not any(
            kw in user_input.lower() for kw in ["find", "search", "paper"]
        ) else user_input
        
        result = await process_curator_command(research_input)
        msg.content = result
        msg.status = "✓ Complete"
        await msg.update()
        
    except Exception as e:
        logger.error(f"Error in research helper: {e}")
        msg.content = f"❌ Error: {str(e)}"
        msg.status = "✗ Error"
        await msg.update()


async def handle_general_chat(user_input: str, profile: Optional[str]):
    """Handle general chat messages."""
    
    response = f"""I'm Xoe-NovAi, your AI assistant for library curation and research.

You said: *{user_input}*

I work best with specific requests like:
- "Find books by {user_input.split()[0] if user_input else 'Plato'}"
- "Research {user_input if len(user_input) > 3 else 'quantum mechanics'}"
- "Show me recommendations about {user_input if len(user_input) > 3 else 'science fiction'}"

**Profile:** {profile}

Try reformulating your request or select "Library Curator" mode for library searches!
    """
    
    await cl.Message(response).send()


@cl.on_session_end
async def end():
    """End chat session and log statistics."""
    
    try:
        voice = get_voice_interface()
        stats = voice.get_session_stats()
        
        logger.info(f"✓ Chat session ended")
        logger.info(f"  - Voice stats: {stats.get('stats', {})}")
    except Exception as e:
        logger.warning(f"Error logging session stats: {e}")


# ============================================================================
# OPTIONAL: AUDIO CHUNK HANDLER (For direct audio input from browser)
# ============================================================================

@cl.on_audio_chunk
async def handle_audio_chunk(chunk: cl.AudioChunk):
    """
    Handle audio chunks from browser (requires Chainlit 1.1+).
    
    Note: This requires the browser to support Web Audio API recording.
    """
    try:
        msg = cl.Message("")
        msg.status = "🎤 Transcribing audio..."
        await msg.send()
        
        # Process audio data
        transcription = await process_voice_input(chunk.data)
        
        msg.content = f"🎤 **Transcribed:** {transcription}"
        msg.status = "✓ Complete"
        await msg.update()
        
        # Process as message
        message = cl.Message(transcription)
        await handle_message(message)
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("XOE-NOVAI CHAINLIT APP WITH VOICE INTERFACE")
    logger.info("="*80)
    logger.info("\nStartup Instructions:")
    logger.info("1. Run: chainlit run app/XNAi_rag_app/chainlit_app_with_voice.py -w --port 8001")
    logger.info("2. Open: http://localhost:8001")
    logger.info("3. Select chat profile (Library Curator / Voice Assistant / Research Helper)")
    logger.info("4. Click 🎤 button to record voice commands or type text")
    logger.info("\nEnvironment Variables:")
    logger.info("  CHAINLIT_NO_TELEMETRY=true      (Disable tracking)")
    logger.info("  ELEVENLABS_API_KEY=<key>        (Optional: Premium voices)")
    logger.info("  OPENAI_API_KEY=<key>            (Optional: Whisper fallback)")
    logger.info("="*80 + "\n")
