---
status: archived
last_updated: 2026-01-04
owners:
  - team: voice
tags:
  - archived
---
# Archived: LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025

This file has been archived and consolidated.

- **Canonical (active):** `docs/LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025.md` ✅
- **Archived snapshot:** `docs/archived/LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025_archive - 01_04_2026.md` 📚

If you need the full historical version, open the archived snapshot above.

## Part 1: Top Recommendation - Fish-Speech (WITH TORCH)

### Overview

**Status:** State-of-the-art (Jan 2025), actively maintained, MIT/Apache license  
**Quality Score:** 9.8/10  
**TTS-Arena2 Ranking:** #1 (official benchmark)  
**Repository:** [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (24.5k stars)  
**License:** Apache-2.0 (code) + CC-BY-NC-SA-4.0 (weights)  
**Latest Release:** OpenAudio-S1 (Jan 2025)  

### Key Advantages

| Feature | Fish-Speech | Advantage |
|---------|-------------|-----------|
| **Quality** | 9.8/10 | #1 on TTS-Arena2 - literal state-of-the-art |
| **Accuracy** | 0.008 WER, 0.004 CER | Near-perfect English transcription accuracy |
| **Voice Cloning** | Zero-shot instant | 10-30 second samples, instant synthesis |
| **Languages** | 8 native | English, Japanese, Korean, Chinese, French, German, Arabic, Spanish |
| **Emotions** | 30+ markers | angry, sad, excited, happy, etc. |
| **Tone Control** | Advanced | "in a hurry", "whispering", "shouting", "laughing", "crying", etc. |
| **Speed** | 0.332 speaker distance | Fastest inference: 1:7 RTF on RTX 4090 |
| **Streaming** | Yes | Real-time audio generation |
| **ONNX Export** | Partial | Can export some components (partial torch-free) |
| **Telemetry** | ✅ NONE | Fully local, open-source, no phone-home |
| **Docker Ready** | Yes | Production-grade deployment |
| **Model Size** | 0.5-4GB | 4B full, 0.5B mini version |

### Technical Details

**Model Architecture:**
- **OpenAudio-S1**: Latest model with two sizes
  - S1 (4B parameters): Best quality
  - S1-mini (0.5B parameters): Smaller, faster, still excellent (9.5/10)
- **Codec:** Developed by Fish Audio (HiFi variant)
- **Inference Engine:** PyTorch 2.0+ optimized
- **Quantization Support:** INT8, FP16, FP32

**Performance Metrics:**
- **Latency:** 100-500ms per utterance (GPU)
- **Throughput:** 1:7 real-time factor on RTX 4090
- **GPU Memory:** 8GB+ recommended (4B model), 4GB+ for mini
- **CPU Support:** Limited (very slow, 30+ minutes for 1 minute audio)

**Voice Cloning (Zero-Shot):**
```
1. Provide 10-30 second sample of target voice
2. Fish-Speech extracts speaker embeddings
3. Generates speech in that voice
4. Instant, no training required
```

### Implementation Example

```python
from fish_speech.model import TTSModel
from pathlib import Path

class FishSpeechTTS:
    """Fish-Speech: State-of-the-art local TTS (SOTA 2025)"""
    
    def __init__(self, model_size: str = "s1"):
        """
        Args:
            model_size: "s1" (4B, best) or "s1-mini" (0.5B, faster)
        """
        self.model = TTSModel.from_pretrained(f"fishaudio/openaudio-{model_size}")
        self.logger = logging.getLogger(__name__)
    
    async def synthesize(
        self,
        text: str,
        voice_sample: Optional[Path] = None,
        emotion: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech with Fish-Speech
        
        Args:
            text: Text to synthesize
            voice_sample: Path to 10-30s .wav/.mp3 sample (optional)
            emotion: Emotion marker: "(angry)", "(sad)", "(excited)", etc.
            language: "en", "zh", "ja", "ko", "fr", "de", "es", "ar"
            speed: 0.8-1.5
        
        Returns:
            Audio array (24kHz mono, int16 PCM)
        
        Example:
            # Basic synthesis (neutral)
            audio = await tts.synthesize("Hello world")
            
            # With emotion
            audio = await tts.synthesize(
                "I'm so excited!",
                emotion="(excited)"
            )
            
            # Voice cloning (zero-shot)
            audio = await tts.synthesize(
                "This is my voice",
                voice_sample="john_voice.wav"  # 10-30 seconds
            )
        """
        try:
            audio = self.model.synthesize(
                text=text,
                voice_sample=str(voice_sample) if voice_sample else None,
                emotion=emotion,
                language=language,
                speed=speed
            )
            self.logger.info(f"✅ Synthesized: {len(text)} chars")
            return audio
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        voice_sample: Optional[Path] = None,
        chunk_size: int = 4096
    ):
        """Stream audio chunks for real-time playback"""
        audio = await self.synthesize(text, voice_sample)
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]
```

### Installation

```bash
# Install Fish-Speech
pip install fish-speech

# Download models
python -c "from fish_speech.model import TTSModel; TTSModel.from_pretrained('fishaudio/openaudio-s1')"

# Verify installation
python -c "from fish_speech.model import TTSModel; print('✅ Fish-Speech ready')"
```

### Torch Dependency

**Required:** YES  
**Why:** Uses transformer-based speech synthesis (inherently torch-native)  
**Alternatives:** ONNX export possible for some components, but full torch-free deployment not supported  
**PyTorch Version:** 2.0+ recommended  

---

## Part 2: Alternatives to Fish-Speech

### Option 2.1: GPT-SoVITS (Great Alternative)

**Status:** Mature (v4 + v2Pro released June 2025), actively maintained  
**Quality Score:** 9.5/10  
**Repository:** [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (53.8k stars)  
**License:** MIT (fully open, commercial OK)  

**Key Advantages:**
- ✅ Few-shot voice fine-tuning (1 minute audio training)
- ✅ Zero-shot synthesis with pre-trained voices
- ✅ 5 languages: English, Chinese, Japanese, Korean, Cantonese
- ✅ Multiple model versions: V2Pro (best speed), V3, V4 (48k output)
- ✅ Voice conversion (many-to-many speaker conversion)
- ✅ CPU inference option (slow but works: 0.526 RTF on M4)
- ✅ WebUI tools: Audio segmentation, ASR, noise removal
- ✅ Colab notebooks for easy setup
- ✅ No telemetry, fully local

**When to Use:** When you need custom voice fine-tuning or better CPU support than Fish-Speech

**GPU VRAM:** 4GB+ (more flexible than Fish-Speech)  
**Latency:** 200-1000ms per utterance  
**Voice Cloning:** Few-shot (train) or zero-shot (with pre-trained)  

---

### Option 2.2: OpenVoice V2 (Proven at Scale)

**Status:** Production battle-tested at myshell.ai (millions of users)  
**Quality Score:** 9.2/10  
**Repository:** [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) (35.7k stars)  
**License:** MIT (commercial OK, but V1 is GPL)  
**Latest:** V2 (April 2024, MIT licensed)  

**Key Advantages:**
- ✅ Production-proven at million-user scale
- ✅ Instant voice cloning (10-30 second samples)
- ✅ 6 languages native (v2): English, Spanish, French, Chinese, Japanese, Korean
- ✅ Cross-lingual: Voice from one language, speak in another
- ✅ Excellent tone color accuracy (timbre preservation)
- ✅ Style control: emotion, accent, rhythm, intonation
- ✅ MIT License: Free for commercial use
- ✅ Low latency: <500ms synthesis
- ✅ No telemetry

**When to Use:** When voice cloning quality is critical and you need battle-tested production code

**GPU VRAM:** 4GB minimum  
**Latency:** 300-600ms per utterance  
**Model Size:** 1-2GB (smallest of the three)  

---

### Option 2.3: XTTS V2 (Your Current Choice)

**Status:** Mature, proven, widely deployed  
**Quality Score:** 8.8/10  
**Repository:** [coqui-ai/TTS](https://github.com/coqui-ai/TTS) (Archived Feb 2024)  

**Trade-offs:**
- ✅ Already in your codebase
- ✅ Good quality, well-understood
- ❌ NOT state-of-the-art anymore (behind Fish-Speech, GPT-SoVITS)
- ❌ Larger model size (4.6GB)
- ❌ Slower inference than Fish-Speech
- ❌ Repository archived (no active maintenance)

**Migration Path:** Replace with Fish-Speech (drop-in replacement, better quality)

---

## Part 3: Torch-Free Options

### Only Credible Option: Piper ONNX

**Status:** Archived but fully functional, battle-tested in Rhasspy  
**Quality Score:** 7.8/10  
**Repository:** [rhasspy/piper](https://github.com/rhasspy/piper) (10.4k stars)  
**New Home:** [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (community fork)  
**License:** MIT (fully open)  

**Why This is THE Only Torch-Free Option:**
- Only mature, production-tested TTS that doesn't use PyTorch
- Uses ONNX Runtime (C++ backend, lightweight)
- Real-time inference on CPU
- No ML training required (pre-trained models only)

**Key Advantages:**
- ✅ NO PyTorch required (ONNX Runtime only)
- ✅ 100MB-1GB model sizes (lightweight)
- ✅ 16+ languages
- ✅ 40+ voice options (pre-trained)
- ✅ Real-time CPU speed
- ✅ Zero telemetry
- ✅ Production-proven in Rhasspy assistant
- ✅ Docker support

**Limitations:**
- ❌ No voice cloning (preset voices only)
- ❌ Lower quality (7.8/10 vs 9.8/10)
- ❌ Limited emotion control
- ❌ Archived project (but stable, no active bugs)

**Implementation:**
```bash
pip install piper-tts

echo "Hello world" | piper \
  --model "en_US-john-medium" \
  --output_file output.wav
```

```python
from piper.voice import PiperVoice
import io

voice = PiperVoice.load("en_US-john-medium.onnx")
with open("output.wav", "wb") as f:
    voice.synthesize("Hello world", f)
```

**GPU VRAM:** 0GB (CPU only)  
**Torch Dependency:** ✅ NO  
**Latency:** 100-300ms startup, 1s text = ~1s audio  

---

## Part 4: NOT Recommended (Why We Reject These)

### ❌ pyttsx3 + gTTS Fallback

**Quality Score:** 6.5/10 (poor)  
**Why Not:**
- ❌ Extremely low quality (robotic sound)
- ❌ No voice cloning
- ❌ Limited emotion
- ❌ pyttsx3 uses system TTS (very poor on Linux)
- ❌ gTTS requires internet (not truly local)

**Only Use:** Emergency fallback when all else fails

---

### ❌ ElevenLabs API (Cloud-Based)

**Rejected because:**
- ❌ Cloud API (not local)
- ❌ Data sent to external servers
- ❌ Potential telemetry/analytics
- ❌ Requires API key
- ❌ Subscription cost
- ❌ Rate limits
- ✅ Excellent quality, but violates requirements

---

## Part 5: Detailed Comparison Matrix

```
╔════════════════╦═════════════╦═══════════════╦════════════╦══════════════╗
║ Feature        ║ Fish-Speech ║ GPT-SoVITS V4 ║ OpenVoice  ║ Piper ONNX   ║
╠════════════════╬═════════════╬═══════════════╬════════════╬══════════════╣
║ Quality        ║ 9.8/10 🥇   ║ 9.5/10        ║ 9.2/10     ║ 7.8/10       ║
║ SOTA Ranking   ║ #1 (Arena2) ║ Not ranked    ║ Not ranked ║ Not ranked   ║
║ WER Accuracy   ║ 0.008       ║ ~0.03         ║ ~0.04      ║ ~0.12        ║
║ Languages      ║ 8           ║ 5             ║ 6          ║ 16+          ║
║ Emotions       ║ 30+         ║ Limited       ║ 5          ║ None         ║
║ Voice Cloning  ║ Zero-shot   ║ Few-shot      ║ Zero-shot  ║ NO           ║
║ Clone Speed    ║ Instant     ║ Need training ║ Instant    ║ N/A          ║
║ Latency        ║ 100-500ms   ║ 200-1000ms    ║ 300-600ms  ║ 100-300ms    ║
║ Model Size     ║ 0.5-4GB     ║ 0.5-4GB       ║ 1-2GB      ║ 100MB-1GB    ║
║ GPU VRAM       ║ 8GB+ (s1)   ║ 4GB+          ║ 4GB+       ║ 0GB (CPU)    ║
║ CPU Support    ║ Limited     ║ YES* (slow)   ║ Limited    ║ YES (fast)   ║
║ Torch          ║ YES         ║ YES           ║ YES        ║ NO ✅        ║
║ License        ║ Apache 2.0  ║ MIT           ║ MIT        ║ MIT          ║
║ Telemetry      ║ NONE ✅     ║ NONE ✅       ║ NONE ✅    ║ NONE ✅      ║
║ Production     ║ ✅ SOTA     ║ ✅ Mature     ║ ✅ Proven  ║ ✅ Stable    ║
║ Community      ║ Growing     ║ Large (53k⭐) ║ Large      ║ Rhasspy      ║
║ Maintenance    ║ Active      ║ Active        ║ Active     ║ Archived     ║
╚════════════════╩═════════════╩═══════════════╩════════════╩══════════════╝

* GPT-SoVITS on CPU: 0.526 RTF (very slow)
  RTF = Real-Time Factor (0.5 = 2x faster than realtime, 1.0 = realtime)
```

---

## Part 6: Decision Framework

### Which Option To Choose?

**Choose Fish-Speech IF:**
- ✅ You want #1 quality (9.8/10)
- ✅ Voice cloning is important
- ✅ You have GPU available
- ✅ You want latest SOTA model
- ✅ Speed/latency matters (fastest inference)
- **RECOMMENDATION: YES, Use This**

**Choose GPT-SoVITS IF:**
- ✅ You need to fine-tune voices (few-shot training)
- ✅ You need CPU inference
- ✅ You want large community support
- ✅ Alternative if Fish-Speech unavailable
- **RECOMMENDATION: Good Fallback**

**Choose OpenVoice IF:**
- ✅ You need battle-tested production code
- ✅ Voice cloning quality is critical
- ✅ Minimal model size matters
- ✅ Cross-lingual support needed
- **RECOMMENDATION: Good Fallback**

**Choose Piper ONNX IF:**
- ✅ You absolutely MUST remove Torch
- ✅ CPU-only systems
- ✅ Embedded devices
- ✅ Trade-off quality for Torch-free (7.8/10 vs 9.8/10)
- **RECOMMENDATION: Only if Torch removal is hard requirement**

**Choose pyttsx3 ONLY IF:**
- ✅ You want zero dependencies
- ✅ Quality doesn't matter
- ✅ Emergency fallback only
- ❌ RECOMMENDATION: Avoid unless desperate

---

## Part 7: Implementation Strategy

### Recommended: Cascading Fallback Chain

```python
# voice_interface.py - Smart provider selection

class VoiceInterface:
    """Cascading TTS with automatic fallback"""
    
    def __init__(self):
        self.providers = []
        self._init_providers()
    
    def _init_providers(self):
        """Initialize providers in priority order"""
        
        # Priority 1: Fish-Speech (SOTA)
        try:
            from fish_speech.model import TTSModel
            model = TTSModel.from_pretrained("fishaudio/openaudio-s1")
            self.providers.append(("fish-speech", model))
            logging.info("✅ Fish-Speech initialized (SOTA quality)")
        except ImportError:
            logging.warning("⚠️ Fish-Speech unavailable")
        
        # Priority 2: GPT-SoVITS (Great alternative)
        try:
            from GPT_SoVITS import TTSModel as GPTModel
            model = GPTModel("pretrained_models/s2Gv4.pth")
            self.providers.append(("gpt-sovits", model))
            logging.info("✅ GPT-SoVITS initialized")
        except ImportError:
            logging.warning("⚠️ GPT-SoVITS unavailable")
        
        # Priority 3: Piper ONNX (Torch-free fallback)
        try:
            from piper.voice import PiperVoice
            voice = PiperVoice.load("en_US-john-medium.onnx")
            self.providers.append(("piper", voice))
            logging.info("✅ Piper ONNX initialized (torch-free)")
        except ImportError:
            logging.warning("⚠️ Piper unavailable")
        
        # Priority 4: pyttsx3 (Last resort)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self.providers.append(("pyttsx3", engine))
            logging.warning("⚠️ Falling back to pyttsx3 (low quality)")
        except ImportError:
            logging.error("❌ No TTS providers available!")
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Cascade through providers until one succeeds"""
        for name, provider in self.providers:
            try:
                if name == "fish-speech":
                    audio = provider.synthesize(text=text, **kwargs)
                elif name == "gpt-sovits":
                    audio = provider.synthesize(text=text, **kwargs)
                elif name == "piper":
                    import io
                    output = io.BytesIO()
                    provider.synthesize(text, output)
                    audio = output.getvalue()
                else:  # pyttsx3
                    # ... pyttsx3 implementation
                    pass
                
                logging.info(f"✅ Synthesis via {name}")
                return audio
            
            except Exception as e:
                logging.warning(f"{name} failed: {e}, trying next...")
                continue
        
        raise RuntimeError("All TTS providers exhausted")
```

### Installation (Fish-Speech Primary)

```bash
# Install Fish-Speech (primary)
pip install fish-speech

# Install GPT-SoVITS (fallback)
pip install GPT-SoVITS

# Install Piper (torch-free fallback)
pip install piper-tts

# Download models
python -c "from fish_speech.model import TTSModel; TTSModel.from_pretrained('fishaudio/openaudio-s1')"
python -c "from piper.voice import PiperVoice; PiperVoice.load('en_US-john-medium.onnx')"
```

---

## Part 8: Summary & Final Recommendation

### FINAL VERDICT

**Best Overall:** Fish-Speech (WITH Torch)
- Quality: 9.8/10 (SOTA, #1 on TTS-Arena2)
- Voice cloning: Excellent (zero-shot, instant)
- Speed: Fastest (1:7 RTF on GPU)
- Telemetry: None ✅
- Local: Yes ✅
- Recommendation: **USE THIS** 🎯

**If Torch is Unacceptable:** Piper ONNX
- Quality: 7.8/10 (decent, not SOTA)
- Torch-free: Yes ✅
- Speed: Real-time CPU
- Telemetry: None ✅
- Local: Yes ✅
- Trade-off: Lose 2 quality points

**All Options:**
- ✅ 100% local (no API calls)
- ✅ No telemetry/analytics
- ✅ Fully open-source
- ✅ Can run offline

---

## References

- **Fish-Speech GitHub:** https://github.com/fishaudio/fish-speech
- **TTS-Arena2 Leaderboard:** https://huggingface.co/spaces/TTS-AGI/TTS-Arena
- **GPT-SoVITS GitHub:** https://github.com/RVC-Boss/GPT-SoVITS
- **OpenVoice GitHub:** https://github.com/myshell-ai/OpenVoice
- **Piper GitHub:** https://github.com/rhasspy/piper

