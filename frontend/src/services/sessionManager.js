/**
 * ============================================================
 * Session Manager Service
 * ============================================================
 *
 * Handles:
 * - Per-user session tracking
 * - Session history storage and retrieval
 * - Session state management
 * - User preferences and metadata
 * - Session switching/restoration
 */

import { db } from './firebaseAuth';
import {
  collection,
  doc,
  setDoc,
  getDoc,
  getDocs,
  query,
  where,
  orderBy,
  limit,
  updateDoc,
  deleteDoc,
} from 'firebase/firestore';

/**
 * Session object structure:
 * {
 *   id: string (auto-generated doc ID)
 *   userId: string
 *   title: string (auto-generated or user-defined)
 *   messages: Array<{role, content, timestamp}>
 *   mode: string (grievance, query, feedback, etc)
 *   subMode?: string
 *   detectedLanguage?: string
 *   phoneticPreference?: string
 *   createdAt: ISO timestamp
 *   updatedAt: ISO timestamp
 *   isActive: boolean
 *   metadata: {
 *     messageCount: number
 *     avgLatency: number
 *     hasUnresolvedQuery: boolean
 *     userFeedback?: string
 *     feedbackScore?: number (1-5)
 *   }
 * }
 */

/**
 * Create a new session
 * @param {string} userId - User ID
 * @param {string} mode - Session mode (grievance, query, feedback)
 * @param {string} subMode - Session sub-mode (optional)
 * @param {object} preferences - User preferences
 * @returns {Promise} - Session creation result with ID
 */
export async function createSession(userId, mode = 'query', subMode = '', preferences = {}) {
  try {
    const newSession = {
      userId,
      title: `Session - ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`,
      messages: [],
      mode,
      subMode,
      detectedLanguage: preferences.language || 'en',
      phoneticPreference: preferences.phoneticPreference || 'phonetic_roman',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isActive: true,
      metadata: {
        messageCount: 0,
        avgLatency: 0,
        hasUnresolvedQuery: false,
        userFeedback: null,
        feedbackScore: null,
      },
    };

    // Generate doc ID based on timestamp
    const sessionId = `${userId}-${Date.now()}`;
    await setDoc(doc(db, 'sessions', sessionId), newSession);

    // Return with session ID
    return {
      success: true,
      sessionId,
      session: newSession,
    };
  } catch (error) {
    console.error('Error creating session:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Add message to session
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID
 * @param {string} role - Message role (user/assistant)
 * @param {string} content - Message content
 * @param {number} latency - Response latency in ms
 * @returns {Promise} - Add message result
 */
export async function addMessageToSession(userId, sessionId, role, content, latency = 0) {
  try {
    const sessionDoc = doc(db, 'sessions', sessionId);
    const sessionSnapshot = await getDoc(sessionDoc);

    if (!sessionSnapshot.exists()) {
      return {
        success: false,
        error: 'Session not found',
      };
    }

    const sessionData = sessionSnapshot.data();
    const messages = sessionData.messages || [];
    const metadata = sessionData.metadata || {};

    // Add new message
    messages.push({
      role,
      content,
      timestamp: new Date().toISOString(),
      latency,
    });

    // Update metadata
    const totalLatency = metadata.avgLatency * metadata.messageCount + latency;
    metadata.messageCount = messages.length;
    metadata.avgLatency = totalLatency / messages.length;

    // Update session
    await updateDoc(sessionDoc, {
      messages,
      metadata,
      updatedAt: new Date().toISOString(),
    });

    return {
      success: true,
      messageCount: messages.length,
    };
  } catch (error) {
    console.error('Error adding message to session:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Get all sessions for a user
 * @param {string} userId - User ID
 * @param {number} maxSessions - Maximum sessions to retrieve
 * @returns {Promise} - Array of sessions
 */
export async function getUserSessions(userId, maxSessions = 20) {
  try {
    const sessionsRef = collection(db, 'sessions');
    const q = query(
      sessionsRef,
      where('userId', '==', userId),
      orderBy('updatedAt', 'desc'),
      limit(maxSessions)
    );

    const querySnapshot = await getDocs(q);
    const sessions = [];

    querySnapshot.forEach((doc) => {
      sessions.push({
        id: doc.id,
        ...doc.data(),
      });
    });

    return sessions;
  } catch (error) {
    console.error('Error fetching user sessions:', error);
    return [];
  }
}

/**
 * Get session by ID
 * @param {string} sessionId - Session ID
 * @returns {Promise} - Session data
 */
export async function getSession(sessionId) {
  try {
    const sessionDoc = await getDoc(doc(db, 'sessions', sessionId));
    if (sessionDoc.exists()) {
      return {
        id: sessionDoc.id,
        ...sessionDoc.data(),
      };
    }
    return null;
  } catch (error) {
    console.error('Error fetching session:', error);
    return null;
  }
}

/**
 * Update session metadata
 * @param {string} sessionId - Session ID
 * @param {object} updates - Updated fields
 * @returns {Promise} - Update result
 */
export async function updateSession(sessionId, updates) {
  try {
    await updateDoc(doc(db, 'sessions', sessionId), {
      ...updates,
      updatedAt: new Date().toISOString(),
    });

    return { success: true };
  } catch (error) {
    console.error('Error updating session:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Set session as inactive
 * @param {string} sessionId - Session ID
 * @returns {Promise} - Result
 */
export async function closeSession(sessionId) {
  try {
    await updateDoc(doc(db, 'sessions', sessionId), {
      isActive: false,
      updatedAt: new Date().toISOString(),
    });

    return { success: true };
  } catch (error) {
    console.error('Error closing session:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Delete session
 * @param {string} sessionId - Session ID
 * @returns {Promise} - Delete result
 */
export async function deleteSession(sessionId) {
  try {
    await deleteDoc(doc(db, 'sessions', sessionId));
    return { success: true };
  } catch (error) {
    console.error('Error deleting session:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Add feedback to session
 * @param {string} sessionId - Session ID
 * @param {string} feedback - Feedback text
 * @param {number} score - Feedback score (1-5)
 * @returns {Promise} - Result
 */
export async function addSessionFeedback(sessionId, feedback, score) {
  try {
    if (score < 1 || score > 5) {
      return {
        success: false,
        error: 'Score must be between 1 and 5',
      };
    }

    await updateDoc(doc(db, 'sessions', sessionId), {
      'metadata.userFeedback': feedback,
      'metadata.feedbackScore': score,
      'metadata.hasUnresolvedQuery': score < 3,
      updatedAt: new Date().toISOString(),
    });

    return { success: true };
  } catch (error) {
    console.error('Error adding session feedback:', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Get session statistics (Admin dashboard)
 * @param {string} userId - Optional: Get stats for specific user
 * @returns {Promise} - Statistics
 */
export async function getSessionStatistics(userId = null) {
  try {
    let q;

    if (userId) {
      q = query(collection(db, 'sessions'), where('userId', '==', userId));
    } else {
      q = query(collection(db, 'sessions'));
    }

    const querySnapshot = await getDocs(q);
    let totalSessions = 0;
    let totalMessages = 0;
    let totalLatency = 0;
    let avgScore = 0;
    let sessionCount = 0;

    querySnapshot.forEach((doc) => {
      const session = doc.data();
      totalSessions += 1;
      totalMessages += session.metadata?.messageCount || 0;
      totalLatency += session.metadata?.avgLatency || 0;

      if (session.metadata?.feedbackScore) {
        avgScore += session.metadata.feedbackScore;
        sessionCount += 1;
      }
    });

    const avgLatencyPerSession = totalSessions > 0 ? totalLatency / totalSessions : 0;
    const avgFeedbackScore = sessionCount > 0 ? avgScore / sessionCount : 0;

    return {
      totalSessions,
      totalMessages,
      avgLatency: Math.round(avgLatencyPerSession),
      avgFeedbackScore: avgFeedbackScore.toFixed(2),
      sessionsWithFeedback: sessionCount,
    };
  } catch (error) {
    console.error('Error fetching session statistics:', error);
    return null;
  }
}

const sessionManager = {
  createSession,
  addMessageToSession,
  getUserSessions,
  getSession,
  updateSession,
  closeSession,
  deleteSession,
  addSessionFeedback,
  getSessionStatistics,
};

export default sessionManager;
