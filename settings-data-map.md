# Settings Data Mapping

## Overview
This document specifies how the existing backend settings schema (`GET /api/user/settings` and `PUT /api/user/settings`) maps to the Figma UI `SettingsPage.tsx` controls.

## Current Settings Schema
Backend Schema (`SETTINGS_SCHEMA` in `endpoints_v2.py`):
```json
{
  "theme": "dark" | "light" | "system",
  "language": "en" | "es" | "fr" | "de" | "zh",
  "response_style": "concise" | "balanced" | "detailed",
  "default_mode": "standard" | "debate" | "evidence" | "glass" | "synthesis",
  "default_model": "llama-3-3-70b",
  "notifications_enabled": true | false,
  "debate_rounds": 1 - 10,
  "auto_save": true | false
}
```

## Component Mapping

| Figma Settings Control | Backend Payload Key | Supported | Action if Unsupported |
| --- | --- | --- | --- |
| **Theme Selector** | `theme` | Yes | N/A |
| **Default Engine (Model)** | `default_model` | Yes | N/A |
| **Orchestration Mode** | `default_mode` | Yes | N/A |
| **Response Style** | `response_style` | Yes | N/A |
| **Debate Depth** | `debate_rounds` | Yes | N/A |
| **Telemetry Opt-In** | N/A | No | Hide / Remove |
| **Analytics Opt-In** | N/A | No | Hide / Remove |
| **Feedback Opt-In** | N/A | No | Hide / Remove |
| **Export Data** | N/A | No | Show `alert('Not Implemented')` or disable |
| **Delete Account** | N/A | No | Show `alert('Not Implemented')` or disable |

## Implementation Strategy
1. The `SettingsPage` `useEffect` will fetch `/api/user/settings` using `api.get()` from `@services/api.js`.
2. The payload will update the React state (`preferences` and `advanced`).
3. Whenever a setting is modified in the UI, an `api.put('/api/user/settings', { key: value })` request will be dispatched.
4. Unsupported privacy toggles (telemetry, analytics) currently present in `SettingsPage.tsx` will be completely removed from the JSX to prevent UX confusion.
