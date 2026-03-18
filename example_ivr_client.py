#!/usr/bin/env python3
"""
Example IVR Client Script

This script demonstrates how to use the IVR telephonic system client
to stream audio to a Colab server and receive LLM-generated responses.

Based on the architecture described in the problem statement:
1. Caller -> Asterisk PBX (Raw Audio)
2. Colab Server -> Faster-Whisper (STT) + IndicBERT (Intent) + Wav2Vec (Accent)
3. IVR Client (This Script) -> LLM Orchestrator
4. LLM Response -> Regional TTS Engine
5. TTS Audio -> Asterisk PBX -> Caller
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ivr.ivr_client import stream_audio_to_pipeline, run_ivr_client
from backend.ivr.ivr_orchestrator import IVROrchestrator


async def test_orchestrator_standalone():
    """
    Test the IVR orchestrator without WebSocket connection.
    This demonstrates how the LLM processes pipeline data.
    """
    print("\n" + "="*80)
    print("TEST 1: IVR Orchestrator Standalone Test")
    print("="*80)

    orchestrator = IVROrchestrator()

    # Test Case 1: High-urgency network issue in Hindi (Mumbai)
    test_data_1 = {
        "text": "Mera network kaam nahi kar raha hai, please fix it.",
        "language": "hi",
        "detected_profile": {"lang": "hi_mumbai"},
        "intent_data": {
            "category": "Network_Issue",
            "urgency": "High"
        }
    }

    print("\n--- Test Case 1: Network Issue (Hindi - Mumbai) ---")
    print(f"Input: {test_data_1['text']}")
    response_1 = await orchestrator.process_pipeline_data(test_data_1)
    print(f"Response: {response_1['tts_script']}")
    print(f"Action: {response_1['suggested_action']}")
    print(f"Emotion: {response_1['tts_overrides']['emotion']}")

    # Test Case 2: Billing query in English (Bangalore)
    test_data_2 = {
        "text": "I want to check my bill for this month",
        "language": "en",
        "detected_profile": {"lang": "en_bangalore"},
        "intent_data": {
            "category": "Billing_Query",
            "urgency": "Medium"
        }
    }

    print("\n--- Test Case 2: Billing Query (English - Bangalore) ---")
    print(f"Input: {test_data_2['text']}")
    response_2 = await orchestrator.process_pipeline_data(test_data_2)
    print(f"Response: {response_2['tts_script']}")
    print(f"Action: {response_2['suggested_action']}")
    print(f"Emotion: {response_2['tts_overrides']['emotion']}")

    # Test Case 3: General inquiry in formal Hindi (Rural UP)
    test_data_3 = {
        "text": "Kya aap meri madad kar sakte hain?",
        "language": "hi",
        "detected_profile": {"lang": "hi_rural_up"},
        "intent_data": {
            "category": "General_Inquiry",
            "urgency": "Low"
        }
    }

    print("\n--- Test Case 3: General Inquiry (Hindi - Rural UP) ---")
    print(f"Input: {test_data_3['text']}")
    response_3 = await orchestrator.process_pipeline_data(test_data_3)
    print(f"Response: {response_3['tts_script']}")
    print(f"Action: {response_3['suggested_action']}")
    print(f"Emotion: {response_3['tts_overrides']['emotion']}")


def test_websocket_client():
    """
    Test the WebSocket client for audio streaming.
    NOTE: This requires a running Colab server with WebSocket endpoint.
    """
    print("\n" + "="*80)
    print("TEST 2: WebSocket Client Test")
    print("="*80)
    print("\nNOTE: This test requires:")
    print("  1. A Colab server running Faster-Whisper with WebSocket endpoint")
    print("  2. An ngrok URL for the Colab server")
    print("  3. A test audio file (16kHz, 16-bit mono WAV)")
    print("\nTo run this test:")
    print("  1. Update COLAB_WEBSOCKET_URL in backend/ivr/ivr_client.py")
    print("  2. Place a test audio file named 'test_hindi_complaint.wav' in the repo root")
    print("  3. Run: python example_ivr_client.py --websocket")
    print("\nSkipping WebSocket test (no server configured)...")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IVR Telephonic System Client Example"
    )
    parser.add_argument(
        '--websocket',
        action='store_true',
        help='Run WebSocket client test (requires Colab server)'
    )
    parser.add_argument(
        '--ws-url',
        type=str,
        help='WebSocket URL for Colab server'
    )
    parser.add_argument(
        '--audio-file',
        type=str,
        help='Path to test audio file (16kHz, 16-bit mono WAV)'
    )

    args = parser.parse_args()

    if args.websocket:
        # Run WebSocket client
        if not args.ws_url:
            print("ERROR: --ws-url required for WebSocket test")
            print("Example: python example_ivr_client.py --websocket --ws-url ws://your-ngrok-url.ngrok.app/stream")
            sys.exit(1)

        print(f"\nConnecting to WebSocket server: {args.ws_url}")
        if args.audio_file:
            print(f"Streaming audio file: {args.audio_file}")

        run_ivr_client(websocket_url=args.ws_url, audio_file=args.audio_file)
    else:
        # Run standalone orchestrator test
        asyncio.run(test_orchestrator_standalone())
        print("\n" + "="*80)
        print("All tests completed successfully!")
        print("="*80)


if __name__ == "__main__":
    main()
