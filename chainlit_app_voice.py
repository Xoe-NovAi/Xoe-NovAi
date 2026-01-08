"""
Compatibility wrapper: `chainlit_app_voice` -> re-export existing Chainlit app.

This module imports and re-exports the Chainlit app implemented in
`app/XNAi_rag_app/chainlit_app_enterprise_voice.py` so docs and run commands
may reference the shorter name.
"""

from app.XNAi_rag_app.chainlit_app_voice import *  # noqa: F401,F403

__all__ = [
    name for name in globals().keys() if not name.startswith("__")
]

print("Loaded chainlit_app_voice compatibility wrapper")
