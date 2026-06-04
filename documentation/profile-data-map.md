# Profile Data Mapping

## Overview
This document specifies how the existing backend payload from `GET /api/user` and Supabase auth maps into the Figma UI `ProfilePage.tsx` components.

## Current User Data Source
Payload from `GET /api/user` (via `frontend/src/services/api.js` interceptor unwrapping):
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "User Name",
  "stats": {
    "chat_count": 42,
    "message_count": 128,
    "memory_count": 5
  }
}
```

## Component Mapping

| Figma Profile Component | Backend Data Source | Fallback / Read-Only |
| --- | --- | --- |
| **Profile Header (Name)** | `response.data.name` or `user.user_metadata.full_name` | `user.email.split('@')[0]` |
| **Avatar Section** | `user.user_metadata.avatar_url` | Default placeholder / Initials |
| **Email Section** | `user.email` | N/A |
| **Conversations Stat** | `response.data.stats.chat_count` | `0` |
| **Messages Stat** | `response.data.stats.message_count` | `0` |
| **Favorite Mode Stat** | Not supported natively via `/api/user` | Placeholder / Hidden |
| **Favorite Model Stat** | Not supported natively via `/api/user` | Placeholder / Hidden |

## Displayed Sections (Final UI State)
1. **Header Block**: Avatar + Editable Name. (Avatar upload will be disabled).
2. **Account Information**: Email (Read-Only).
3. **Usage Statistics**: Conversations Count, Messages Count.
4. **Subscription Status**: `user.user_metadata.subscription` (Read-only badge).
