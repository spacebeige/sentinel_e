import os
import json

# Canonical Cognitive Runtime Contracts
# Matches src/app/config/runtime.ts on the frontend

ORCHESTRATION_MODE_MAP = {
    "debate": {
        "endpoint": "/api/mco/run",
        "mode": "debate",
        "orchestration": True,
    },
    "glass": {
        "endpoint": "/api/mco/run",
        "mode": "glass",
        "orchestration": True,
    },
    "evidence": {
        "endpoint": "/api/mco/run",
        "mode": "evidence",
        "orchestration": True,
    },
    "synthesis": {
        "endpoint": "/api/mco/run",
        "mode": "synthesis",
        "orchestration": True,
    },
}

MODEL_RUNTIME_MAP = {
    "llama-3-3-70b": {
        "provider": "groq",
        "backendModel": "llama-3.3-70b-versatile",
        "runtime": "cloud",
    },
    "gemini-flash-2-0": {
        "provider": "google",
        "backendModel": "gemini-2.0-flash",
        "runtime": "cloud",
    },
    "deepseek": {
        "provider": "deepseek",
        "backendModel": "deepseek-chat",
        "runtime": "cloud",
    },
    "qwen3-32b": {
        "provider": "alibaba",
        "backendModel": "qwen3-32b",
        "runtime": "cloud",
    },
    "llama-4-scout-17b": {
        "provider": "meta",
        "backendModel": "llama-4-scout-17b",
        "runtime": "cloud",
    },
    "qwen-2-5-vl-7b": {
        "provider": "openrouter",
        "backendModel": "qwen/qwen-2.5-7b-instruct",
        "runtime": "cloud",
    },
    "llama-3-1-8b-instant": {
        "provider": "groq",
        "backendModel": "llama-3.1-8b-instant",
        "runtime": "cloud",
    },
    "mistral-large-3-675b": {
        "provider": "mistral",
        "backendModel": "mistral-large-3-675b",
        "runtime": "cloud",
    }
}
