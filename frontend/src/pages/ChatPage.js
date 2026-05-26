console.log("OLD LEGACY CHATPAGE ACTIVE");
/**
 * ChatPage.js — Route: /chat
 * Thin wrapper that renders the ChatEngine (logic authority).
 * No logic duplication. No backend code.
 */
import React from 'react';
import ChatEngineV5 from '../components/ChatEngineV5';

export default function ChatPage() {
  return <ChatEngineV5 />;
}
