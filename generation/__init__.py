"""
Generation module for Phase 3: Audio & Narration

This module contains components for:
- Script generation using multimodal LLM (Gemini 2.5 Flash)
- Audio synthesis using multi-provider TTS (ElevenLabs, OpenAI)
"""

from .script_generator import ScriptGenerator

__all__ = ['ScriptGenerator']