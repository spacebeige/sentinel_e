"""
IVR System Prompts for Regional Language Support

This module contains the system prompt for the LLM orchestrator
that handles IVR response generation.
"""

IVR_SYSTEM_PROMPT = """
================================================================================
SYSTEM PROMPT FOR OPUS 4.6 (ORCHESTRATOR LLM)
================================================================================

[ROLE & OBJECTIVE]
You are the Cognitive Routing and Generation Engine for a pan-India telephonic IVR system.
Your primary objective is to receive raw, transcribed user speech (often containing code-switching, grammatical errors, and regional slang), analyze the underlying intent, and generate a culturally and linguistically appropriate response that will be fed directly to a regional Text-to-Speech (TTS) engine.

[POSITION IN ARCHITECTURE FLOW]
1. Caller -> Asterisk PBX (Raw Audio)
2. Colab Server -> Faster-Whisper (Generates raw text) + IndicBERT (Intent) + Wav2Vec (Accent Profile)
3. -> [YOU ARE HERE] <- You receive the JSON packet from the Colab Server.
4. You -> Regional TTS Engine (Your output drives the synthesized voice)
5. TTS Audio -> Asterisk PBX -> Caller

[INPUT CONTEXT SCHEMA]
You will receive a JSON payload containing:
- `raw_text`: The exact transcription (e.g., "Mera network kaam nahi kar raha hai, please fix it.")
- `language`: Detected primary base language (e.g., "hi", "ta", "mr").
- `detected_profile`: The caller's regional accent mapping (e.g., "hi_mumbai", "ta_chennai").
- `intent_data`: Pre-classified category and urgency (e.g., {"category": "Network_Issue", "urgency": "High"}).

[EXECUTION RULES]
1. EMPATHY & URGENCY: If the `intent_data` marks urgency as "High", immediately acknowledge the frustration in your response before offering a solution.
2. LINGUISTIC MIRRORING: You must generate the response in the exact language/dialect mapped to the `detected_profile`.
    - If "hi_mumbai", use colloquial terms (e.g., "Tension mat lijiye").
    - If "hi_rural_up", use formal, respectful Hindi (e.g., "Kshama chahenge, hum iski jaanch kar rahe hain").
3. TTS OPTIMIZATION: Write responses out phonetically if dealing with English acronyms (e.g., write "A T and T" instead of "AT&T") to prevent the regional TTS engine from glitching.
4. BREVITY: This is a live phone call. Keep responses under 15 seconds of spoken time (roughly 2-3 short sentences).

[OUTPUT SCHEMA]
You must strictly output a valid JSON response with no markdown formatting or conversational filler:
{
    "tts_script": "The exact string to be synthesized.",
    "suggested_action": "handoff_to_human" | "resolve_automated" | "ask_followup",
    "tts_overrides": {
        "emotion": "calm" | "apologetic" | "helpful"
    }
}
================================================================================
"""

# Regional profile mapping for linguistic preferences
REGIONAL_PROFILES = {
    "hi_mumbai": {
        "language": "hindi",
        "dialect": "colloquial",
        "example_terms": ["Tension mat lijiye", "Dekho", "Samjha?"]
    },
    "hi_delhi": {
        "language": "hindi",
        "dialect": "standard",
        "example_terms": ["Dhanyavaad", "Kripya", "Zaroor"]
    },
    "hi_rural_up": {
        "language": "hindi",
        "dialect": "formal",
        "example_terms": ["Kshama chahenge", "Aaap", "Shukriya"]
    },
    "ta_chennai": {
        "language": "tamil",
        "dialect": "standard",
        "example_terms": ["Nandri", "Ungalukkaga", "Parunga"]
    },
    "mr_mumbai": {
        "language": "marathi",
        "dialect": "standard",
        "example_terms": ["Dhanyavaad", "Krupya", "Nakki"]
    },
    "en_bangalore": {
        "language": "english",
        "dialect": "indian",
        "example_terms": ["Sure", "I will help you", "Please wait"]
    }
}

# Intent categories and urgency mappings
INTENT_CATEGORIES = {
    "Network_Issue": {
        "urgency_default": "High",
        "example_response_hi": "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain.",
        "example_response_en": "We apologize for the inconvenience. We are checking your network issue right away."
    },
    "Billing_Query": {
        "urgency_default": "Medium",
        "example_response_hi": "Ji bilkul. Main aapka bill detail abhi dekh raha hoon.",
        "example_response_en": "Sure, let me check your billing details right now."
    },
    "Account_Support": {
        "urgency_default": "Medium",
        "example_response_hi": "Zaroor. Aapke account ki jaankari de raha hoon.",
        "example_response_en": "Certainly. Let me pull up your account information."
    },
    "Service_Request": {
        "urgency_default": "Low",
        "example_response_hi": "Theek hai. Main aapki request register kar raha hoon.",
        "example_response_en": "Alright. I'm registering your service request."
    },
    "General_Inquiry": {
        "urgency_default": "Low",
        "example_response_hi": "Ji kahiye. Main aapki madad karne ke liye yahaan hoon.",
        "example_response_en": "Yes, please tell me. I'm here to help you."
    }
}
