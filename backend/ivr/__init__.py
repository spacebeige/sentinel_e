"""
IVR (Interactive Voice Response) System Module

This module provides components for a pan-India telephonic IVR system
with LLM-powered response generation.
"""

from .ivr_orchestrator import IVROrchestrator
from .ivr_client import stream_audio_to_pipeline, mock_llm_processing

__all__ = ['IVROrchestrator', 'stream_audio_to_pipeline', 'mock_llm_processing']
