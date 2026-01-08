"""
Compatibility wrapper: new public module name `voice_interface`.

This file re-exports the implementation from `app/XNAi_rag_app/voice_interface`.
It allows code and docs to import `voice_interface` during the rename transition.
"""

from app.XNAi_rag_app.voice_interface import *  # noqa: F401,F403

__all__ = [
    name for name in globals().keys() if not name.startswith("__")
]

print("Loaded voice_interface compatibility wrapper (use app/XNAi_rag_app/voice_interface.py for source)")
