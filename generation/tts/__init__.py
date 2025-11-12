"""
TTS (Text-to-Speech) module with support for multiple providers.

This module provides a unified interface for different TTS providers
including ElevenLabs and OpenAI, with configuration-based provider selection.
"""

from .base import BaseTTSSynthesizer, AudioSegment, AudioResult
from .elevenlabs_provider import ElevenLabsTTSSynthesizer
from .openai_provider import OpenAITTSSynthesizer

__all__ = [
    "BaseTTSSynthesizer",
    "AudioSegment", 
    "AudioResult",
    "ElevenLabsTTSSynthesizer",
    "OpenAITTSSynthesizer",
]