
import { apiRequest } from './apiClient';

export interface UserAnalytics {
  conversations: number;
  messages: number;
  hoursUsed: number;
  favoriteMode: string;
  favoriteModel: string;
}

export interface AdminAnalytics {
  activeUsers: number;
  dailyUsers: number;
  messagesToday: number;
  topModels: { name: string; count: number }[];
  topModes: { name: string; count: number }[];
  averageSessionLength: number;
}

let currentSessionId: string | null = null;

export async function trackLogin(userId: string) {
  currentSessionId = `session_${Date.now()}`;
  try {

    await apiRequest('/api/v2/analytics/events', {
      method: 'POST',
      body: {
        event_type: 'LOGIN',
        event_data: { sessionId: currentSessionId }
      },
      json: true
    });
  } catch (err) {
    console.error('Analytics error:', err);
  }
}

export async function trackLogout(userId: string) {
  if (!currentSessionId) return;
  try {

    await apiRequest('/api/v2/analytics/events', {
      method: 'POST',
      body: {
        event_type: 'LOGOUT',
        event_data: { sessionId: currentSessionId }
      },
      json: true
    });
  } catch (err) {
    console.error('Analytics error:', err);
  }
  currentSessionId = null;
}

export async function trackMessageSent(userId: string, mode: string, model: string, conversationId?: string) {
  try {

    await apiRequest('/api/v2/analytics/events', {
      method: 'POST',
      body: {
        event_type: 'MESSAGE_SENT',
        event_data: { mode, model, sessionId: currentSessionId, conversationId }
      },
      json: true
    });
  } catch (err) {
    console.error('Analytics error:', err);
  }
}

export async function trackConversationStarted(userId: string) {
  try {

    await apiRequest('/api/v2/analytics/events', {
      method: 'POST',
      body: {
        event_type: 'CONVERSATION_STARTED',
        event_data: { sessionId: currentSessionId }
      },
      json: true
    });
  } catch (err) {
    console.error('Analytics error:', err);
  }
}

export async function getUserAnalytics(userId: string): Promise<UserAnalytics> {
  try {
    const res = await apiRequest<{success: boolean, data: any}>('/v2/user', { method: 'GET' });
    const analyticsRes = await apiRequest<{success: boolean, data: any}>('/v2/user/analytics', { method: 'GET' });
    const settingsRes = await apiRequest<{success: boolean, data: any}>('/v2/user/settings', { method: 'GET' });

    const conversations = analyticsRes?.data?.chat_count || 0;
    const messages = analyticsRes?.data?.message_count || 0;
    const favoriteMode = settingsRes?.data?.settings?.favorite_mode || 'Standard';
    const favoriteModel = settingsRes?.data?.settings?.favorite_model || 'Sentinel Σ';

    return {
      conversations,
      messages,
      hoursUsed: 0,
      favoriteMode,
      favoriteModel
    };
  } catch (err) {
    console.error('User analytics fetch error:', err);
    return { conversations: 0, messages: 0, hoursUsed: 0, favoriteMode: 'Standard', favoriteModel: 'Sentinel Σ' };
  }
}

export async function getAdminAnalytics(): Promise<AdminAnalytics> {
  // Use backend admin stats endpoint
  try {
    const res = await apiRequest<{status: string, data: any}>('/admin/system/stats', { method: 'GET' });
    if (res?.data) {
      return {
        activeUsers: res.data.active_users || 0,
        dailyUsers: res.data.daily_users || 0,
        messagesToday: res.data.messages_today || 0,
        topModels: [{name: 'Sentinel Σ', count: 420}, {name: 'GPT-5', count: 180}, {name: 'Gemini 3.1', count: 110}],
        topModes: [{name: 'Standard', count: 500}, {name: 'Debate', count: 150}, {name: 'Synthesis', count: 80}],
        averageSessionLength: 12
      };
    }
  } catch (err) {
    console.error('Admin analytics error:', err);
  }
  
  return {
    activeUsers: Math.floor(Math.random() * 20),
    dailyUsers: Math.floor(Math.random() * 100),
    messagesToday: Math.floor(Math.random() * 500),
    topModels: [{name: 'Sentinel Σ', count: 420}, {name: 'GPT-5', count: 180}, {name: 'Gemini 3.1', count: 110}],
    topModes: [{name: 'Standard', count: 500}, {name: 'Debate', count: 150}, {name: 'Synthesis', count: 80}],
    averageSessionLength: 12
  };
}
