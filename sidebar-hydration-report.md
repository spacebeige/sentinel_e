# Sidebar Hydration Report

## The Problem
The sidebar was rendering "No chats yet" even when messages were present in the backend. 

Investigation into `ChatPage.tsx` revealed that the mapped `chatHistory` array was rigidly extracting `c.title` from the backend `chats` payload. However, the legacy backend and store sometimes persist the chat name as `name` or `chat_title`. This caused the array map to result in `undefined` values that the Figma UI sidebar components failed to render.

## The Fix
Expanded the mapping logic to securely resolve the title across all legacy naming conventions, providing a solid fallback:
```tsx
const chatHistory = chats.map((c: any) => ({
  id: c.id || c.chat_id,
  title: c.title || c.name || c.chat_title || "Untitled Chat",
  mode: c.mode || "standard",
  timestamp: c.updated_at ? new Date(c.updated_at) : (c.created_at ? new Date(c.created_at) : new Date())
}));
```
This ensures the legacy backend payloads correctly hydrate the new Figma sidebar.
