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

export async function trackEvent() {
  return;
}

export async function trackLogin() {
  return;
}

export async function trackLogout() {
  return;
}

export async function trackMessageSent() {
  return;
}

export async function trackConversationStarted() {
  return;
}

export async function getUserAnalytics(): Promise<UserAnalytics> {
  return { conversations: 0, messages: 0, hoursUsed: 0, favoriteMode: 'Standard', favoriteModel: 'Sentinel Σ' };
}

export async function getAdminAnalytics(): Promise<AdminAnalytics> {
  return {
    activeUsers: 0,
    dailyUsers: 0,
    messagesToday: 0,
    topModels: [],
    topModes: [],
    averageSessionLength: 0
  };
}
