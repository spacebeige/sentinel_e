/**
 * ============================================================
 * Session Sidebar Component
 * ============================================================
 *
 * Displays:
 * - Current user sessions/chats
 * - New chat button
 * - Session switching
 * - Session deletion
 * - Search/filter sessions
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useAuthContext } from '../hooks/useAuthContext';
import { getHistory, createChat } from '../services/api';
import '../styles/SessionSidebar.css';

const SessionSidebar = ({ isOpen, onSelectSession }) => {
  const { user } = useAuthContext();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeSessionId, setActiveSessionId] = useState(null);

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      // Use API to fetch history (session/chat data)
      const historyResponse = await getHistory(50);
      const chats = historyResponse?.chats || [];
      setSessions(chats);
      if (chats.length > 0) {
        setActiveSessionId(chats[0].id);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user?.id) {
      loadSessions();
    }
  }, [user?.id, loadSessions]);

  const handleNewChat = async () => {
    try {
      const result = await createChat('New Chat', 'conversational');
      if (result?.id) {
        setActiveSessionId(result.id);
        loadSessions();
        onSelectSession?.(result.id);
      }
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
    onSelectSession?.(sessionId);
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    if (window.confirm('Delete this chat? This action cannot be undone.')) {
      try {
        // TODO: Add deleteChat API when available
        // For now, just refresh
        loadSessions();
        if (activeSessionId === sessionId) {
          setActiveSessionId(null);
        }
      } catch (error) {
        console.error('Error deleting chat:', error);
      }
    }
  };

  const filteredSessions = sessions.filter((session) =>
    session.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <aside className={`session-sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <button
          className="new-chat-button"
          onClick={handleNewChat}
          title="Start a new chat"
        >
          ✚ New Chat
        </button>
      </div>

      <div className="sidebar-search">
        <input
          type="text"
          placeholder="Search chats..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>

      <div className="sessions-list">
        {loading ? (
          <div className="loading">Loading chats...</div>
        ) : filteredSessions.length > 0 ? (
          filteredSessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${activeSessionId === session.id ? 'active' : ''}`}
              onClick={() => handleSelectSession(session.id)}
              title={session.title}
            >
              <div className="session-info">
                <div className="session-title">{session.title}</div>
                <div className="session-meta">
                  {session.metadata?.messageCount || 0} messages
                  {session.metadata?.feedbackScore && (
                    <span className="feedback-score">★ {session.metadata.feedbackScore}</span>
                  )}
                </div>
              </div>
              <button
                className="delete-button"
                onClick={(e) => handleDeleteSession(session.id, e)}
                title="Delete chat"
              >
                🗑️
              </button>
            </div>
          ))
        ) : (
          <div className="empty-state">
            No chats yet. Start a new conversation!
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <small>Sessions synced to cloud</small>
      </div>
    </aside>
  );
};

export default SessionSidebar;
