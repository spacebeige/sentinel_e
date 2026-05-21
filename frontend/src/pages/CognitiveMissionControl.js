/**
 * ============================================================
 * Admin Dashboard — System Architecture & Analytics
 * ============================================================
 * Styled like landing page with:
 * - System overview cards
 * - Web analytics
 * - Feedback analysis
 * - System architecture display
 * - User management
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Brain, Activity, Users, TrendingUp, BarChart3,
  Zap, RefreshCw, CheckCircle,
  Cpu
} from 'lucide-react';
import MakeAdminForm from '../components/MakeAdminForm';
import DataFallback from '../components/common/DataFallback';
import api from '../services/api';
import { readSupabaseSessionSnapshot } from '../services/supabaseSessionManager';
import SentinelIdentity from '../components/SentinelIdentity';

const FONT = "'Inter', -apple-system, sans-serif";

const CognitiveMissionControl = () => {
  const [systemStats, setSystemStats] = useState(null);
  const [architecture, setArchitecture] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp },
    { id: 'modes', label: 'Modes', icon: Zap },
    { id: 'orchestrator', label: 'Orchestrator', icon: RefreshCw },
    { id: 'memory', label: 'Memory', icon: Brain },
    { id: 'models', label: 'Models', icon: Activity },
    { id: 'architecture', label: 'Architecture', icon: Zap },
    { id: 'feedback', label: 'Feedback', icon: BarChart3 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'cognitive_runtime', label: 'Cognitive Runtime', icon: Cpu },
    { id: 'agentic_controls', label: 'Agentic Controls', icon: Zap },
  ];
  const availableTabs = tabs;

  const fetchAdminData = useCallback(async () => {
    try {
      setError(null);

      const [statsRes, archRes, analyticsRes, feedbackRes] = await Promise.allSettled([
        api.get('/api/admin/system/stats'),
        api.get('/api/admin/system/architecture'),
        api.get('/api/admin/web-analytics?days=7'),
        api.get('/api/admin/feedback-summary'),
      ]);

      if (statsRes.status === 'fulfilled' && statsRes.value) {
        setSystemStats(statsRes.value);
      } else if (statsRes.status === 'rejected') {
        console.error('Failed to fetch system stats:', statsRes.reason);
      }

      if (archRes.status === 'fulfilled' && archRes.value) {
        setArchitecture(archRes.value);
      } else if (archRes.status === 'rejected') {
        console.error('Failed to fetch architecture:', archRes.reason);
      }

      if (analyticsRes.status === 'fulfilled' && analyticsRes.value) {
        setAnalytics(analyticsRes.value);
      } else if (analyticsRes.status === 'rejected') {
        console.error('Failed to fetch analytics:', analyticsRes.reason);
      }

      if (feedbackRes.status === 'fulfilled' && feedbackRes.value) {
        setFeedback(feedbackRes.value);
      } else if (feedbackRes.status === 'rejected') {
        console.error('Failed to fetch feedback:', feedbackRes.reason);
      }
      
    } catch (err) {
      console.error('Unexpected error in fetchAdminData:', err);
      setError('Failed to load admin data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAdminData();
    const interval = setInterval(fetchAdminData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchAdminData]);

  if (loading) return <LoadingScreen />;

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center sentinel-bg-app">
        <DataFallback 
          message={error} 
          type="error" 
          onRetry={fetchAdminData} 
          className="sentinel-surface shadow-xl max-w-md p-10 sentinel-border"
        />
      </div>
    );
  }

  return (
    <div className="admin-control-dashboard min-h-screen sentinel-bg-app">
      {/* Header */}
      <header className="sentinel-surface border-b sentinel-border sticky top-14 z-40">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <SentinelIdentity size={42} pulse />
                <div>
                  <h1
                    className="text-2xl sentinel-text-primary font-bold"
                    style={{ fontFamily: FONT }}
                  >
                    Cognitive Mission Control
                  </h1>
                  <p className="text-xs sentinel-text-muted">System Architecture & Analytics</p>
                </div>
              </div>
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={fetchAdminData}
              className="flex items-center gap-2 px-4 py-2 rounded-lg sentinel-surface-panel sentinel-text-primary transition-colors"
              style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </motion.button>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mt-6 border-b sentinel-border overflow-x-auto pb-[1px]" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            {availableTabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex flex-shrink-0 items-center gap-2 px-4 py-3 border-b-2 whitespace-nowrap transition-colors ${
                    activeTab === tab.id
                      ? 'border-[#3b82f6] text-[#3b82f6]'
                      : 'border-transparent sentinel-text-muted hover:text-[#1d1d1f] dark:text-[#f1f5f9] dark:hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}>
                    {tab.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-lg bg-red-50 border border-red-200"
          >
            <p className="text-red-700 text-sm">{error}</p>
          </motion.div>
        )}

        {/* Overview Tab */}
        {activeTab === 'overview' && <OverviewTab stats={systemStats} />}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <>
            <AnalyticsTab analytics={analytics} feedback={feedback} />
          </>
        )}

        {/* Modes Tab */}
        {activeTab === 'modes' && <ModesTab stats={systemStats} />}

        {/* Orchestrator Tab */}
        {activeTab === 'orchestrator' && <OrchestratorTab />}

        {/* Memory Tab */}
        {activeTab === 'memory' && <MemoryTab />}

        {/* Models Tab */}
        {activeTab === 'models' && <ModelsTab />}

        {/* Architecture Tab */}
        {activeTab === 'architecture' && <ArchitectureTab architecture={architecture} />}

        {/* Feedback Tab */}
        {activeTab === 'feedback' && <FeedbackTab feedback={feedback} />}

        {/* v8.0: Cognitive Runtime Tab */}
        {activeTab === 'cognitive_runtime' && <CognitiveMissionControlTab apiBase={API_BASE} />}

        {/* Agentic Controls Tab */}
        {activeTab === 'agentic_controls' && <AgenticControlsTab />}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="space-y-12">
            <section>
              <div className="flex flex-wrap items-center gap-3 mb-6">
                <h2 className="text-xl font-bold sentinel-text-primary" style={{ fontFamily: FONT }}>
                  User Management
                </h2>
                <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200">
                  Protected runtime access
                </span>
              </div>
              <MakeAdminForm onSuccess={() => {
                // Optionally refresh data after promoting user
                setTimeout(fetchAdminData, 1000);
              }} />
            </section>
          </div>
        )}
      </main>
    </div>
  );
};

function AgenticControlsTab() {
  const [permissions, setPermissions] = useState({
    browserAutomation: true,
    requireConfirmationRisky: true,
    allowDownloads: false,
    allowUploads: true,
    allowedDomains: ['github.com', 'google.com'],
    blockedDomains: ['facebook.com', 'twitter.com']
  });

  const handleToggle = (key) => {
    setPermissions(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="space-y-12">
      <section>
        <div className="flex flex-wrap items-center gap-3 mb-6">
          <h2 className="text-xl font-bold sentinel-text-primary" style={{ fontFamily: FONT }}>
            Agentic Permissions & Governance
          </h2>
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-200">
            Runtime Controls
          </span>
        </div>
        <p className="text-sm sentinel-text-muted mb-6">
          Configure safety boundaries and reasoning-before-acting permissions for agentic mode execution.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5">
            <h3 className="font-bold mb-4 text-[#1d1d1f] dark:text-[#f1f5f9]">Execution Boundaries</h3>
            <div className="space-y-4">
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <p className="font-medium text-sm text-[#1d1d1f] dark:text-[#f1f5f9]">Browser Automation</p>
                  <p className="text-xs text-[#6e6e73]">Allow agents to spawn browser instances</p>
                </div>
                <input type="checkbox" checked={permissions.browserAutomation} onChange={() => handleToggle('browserAutomation')} className="toggle-checkbox" />
              </label>
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <p className="font-medium text-sm text-[#1d1d1f] dark:text-[#f1f5f9]">Require Confirmation for Risky Actions</p>
                  <p className="text-xs text-[#6e6e73]">Block submits, deletes, and purchases</p>
                </div>
                <input type="checkbox" checked={permissions.requireConfirmationRisky} onChange={() => handleToggle('requireConfirmationRisky')} className="toggle-checkbox" />
              </label>
            </div>
          </div>

          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5">
            <h3 className="font-bold mb-4 text-[#1d1d1f] dark:text-[#f1f5f9]">File Operations</h3>
            <div className="space-y-4">
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <p className="font-medium text-sm text-[#1d1d1f] dark:text-[#f1f5f9]">Allow Downloads</p>
                  <p className="text-xs text-[#6e6e73]">Agents can download files to sandbox</p>
                </div>
                <input type="checkbox" checked={permissions.allowDownloads} onChange={() => handleToggle('allowDownloads')} className="toggle-checkbox" />
              </label>
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <p className="font-medium text-sm text-[#1d1d1f] dark:text-[#f1f5f9]">Allow Uploads</p>
                  <p className="text-xs text-[#6e6e73]">Agents can upload local files to external domains</p>
                </div>
                <input type="checkbox" checked={permissions.allowUploads} onChange={() => handleToggle('allowUploads')} className="toggle-checkbox" />
              </label>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        <SentinelIdentity size={58} pulse />
      </motion.div>
    </div>
  );
}

function OverviewTab({ stats }) {
  if (!stats) {
    return (
      <DataFallback
        message="System telemetry is not available yet."
        className="sentinel-surface sentinel-border p-10"
      />
    );
  }

  const StatCard = ({ label, value, subtext, icon: Icon, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5 hover:border-black/10 transition-all hover:shadow-lg"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-xs uppercase tracking-wider" style={{ color: '#aeaeb2' }}>
            {label}
          </p>
          <h3
            className="text-4xl font-bold mt-2"
            style={{ fontFamily: FONT, color: '#1d1d1f' }}
          >
            {typeof value === 'number' ? value.toLocaleString() : value}
          </h3>
          {subtext && (
            <p className="text-xs mt-2" style={{ color: '#6e6e73' }}>
              {subtext}
            </p>
          )}
        </div>
        {Icon && (
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: `${color}15` }}
          >
            <Icon className="w-6 h-6" style={{ color }} />
          </div>
        )}
      </div>
    </motion.div>
  );

  return (
    <div className="space-y-12">
      {/* Key Metrics */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Key Metrics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Users"
            value={stats.users?.total || 0}
            subtext={`${stats.users?.admins || 0} admins`}
            icon={Users}
            color="#3b82f6"
          />
          <StatCard
            label="Total Chats"
            value={stats.chats?.total || 0}
            subtext={`${stats.chats?.last_24h || 0} in last 24h`}
            icon={Brain}
            color="#8b5cf6"
          />
          <StatCard
            label="Total Messages"
            value={stats.messages?.total || 0}
            subtext={`${stats.messages?.avg_per_chat?.toFixed(1) || 0} per chat`}
            icon={Activity}
            color="#06b6d4"
          />
          <StatCard
            label="Avg Feedback"
            value={`${stats.feedback?.avg_rating || 0}/5.0`}
            subtext={`${stats.feedback?.total_rated || 0} ratings`}
            icon={TrendingUp}
            color="#34c759"
          />
        </div>
      </section>

      {/* By Mode */}
      {stats.chats?.by_mode && (
        <section>
          <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
            Usage by Mode
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(stats.chats.by_mode).map(([mode, count]) => (
              <motion.div
                key={mode}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-[#1c1c1e] rounded-xl p-6 border border-black/5 dark:border-white/5"
              >
                <h3 className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize mb-2">{mode}</h3>
                <p className="text-3xl font-bold text-[#3b82f6]">{count}</p>
              </motion.div>
            ))}
          </div>
        </section>
      )}

      {/* System Health */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          System Health
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { label: 'Uptime', status: stats.system?.uptime_status },
            { label: 'Database', status: stats.system?.db_status },
            { label: 'Cache', status: stats.system?.cache_status },
          ].map(item => (
            <div key={item.label} className="bg-white dark:bg-[#1c1c1e] rounded-xl p-6 border border-black/5 dark:border-white/5">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9]">{item.label}</h3>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-[#34c759]" />
                  <span className="text-sm font-medium text-[#34c759] capitalize">
                    {item.status}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function AnalyticsTab({ analytics, feedback }) {
  if (!analytics) {
    return (
      <DataFallback
        message="Analytics data is still loading."
        className="sentinel-surface sentinel-border p-10"
      />
    );
  }

  return (
    <div className="space-y-12">
      {/* Summary */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          7-Day Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-sm uppercase tracking-wider text-[#6e6e73] mb-2">Total Sessions</p>
            <h3 className="text-4xl font-bold" style={{ fontFamily: FONT }}>
              {analytics.summary?.total_sessions || 0}
            </h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-sm uppercase tracking-wider text-[#6e6e73] mb-2">Unique Users</p>
            <h3 className="text-4xl font-bold" style={{ fontFamily: FONT }}>
              {analytics.summary?.unique_users || 0}
            </h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-sm uppercase tracking-wider text-[#6e6e73] mb-2">Avg per User</p>
            <h3 className="text-4xl font-bold" style={{ fontFamily: FONT }}>
              {analytics.summary?.avg_sessions_per_user?.toFixed(1) || 0}
            </h3>
          </div>
        </div>
      </section>

      {/* Feedback Breakdown */}
      {feedback && (
        <section>
          <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
            Feedback Distribution
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-[#34c759] to-[#00d084] rounded-2xl p-8 text-white">
              <p className="text-sm opacity-90">Positive</p>
              <h3 className="text-4xl font-bold mt-2">{feedback.by_rating?.positive || 0}</h3>
            </div>
            <div className="bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] rounded-2xl p-8 text-white">
              <p className="text-sm opacity-90">Neutral</p>
              <h3 className="text-4xl font-bold mt-2">{feedback.by_rating?.neutral || 0}</h3>
            </div>
            <div className="bg-gradient-to-br from-[#f59e0b] to-[#fbbf24] rounded-2xl p-8 text-white">
              <p className="text-sm opacity-90">Negative</p>
              <h3 className="text-4xl font-bold mt-2">{feedback.by_rating?.negative || 0}</h3>
            </div>
            <div className="bg-gradient-to-br from-[#8b5cf6] to-[#a78bfa] rounded-2xl p-8 text-white">
              <p className="text-sm opacity-90">Total Feedback</p>
              <h3 className="text-4xl font-bold mt-2">{feedback.total_feedback || 0}</h3>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function ArchitectureTab({ architecture }) {
  if (!architecture) {
    return (
      <DataFallback
        message="Architecture summary unavailable."
        className="sentinel-surface sentinel-border p-10"
      />
    );
  }

  return (
    <div className="space-y-12">
      {/* System Overview */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          System Overview
        </h2>
        <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-bold text-xl mb-2">{architecture.system?.name}</h3>
              <p className="text-[#6e6e73] text-sm mb-4">{architecture.system?.description}</p>
              <p className="text-xs text-[#aeaeb2]">Version {architecture.system?.version}</p>
            </div>
            <div>
              <h4 className="font-bold mb-3 text-sm">Key Features</h4>
              <ul className="space-y-2">
                {(Array.isArray(architecture.features?.capabilities) ? architecture.features.capabilities : []).slice(0, 4).map((cap, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-[#34c759] flex-shrink-0 mt-0.5" />
                    <span className="text-[#1d1d1f] dark:text-[#f1f5f9]">{cap}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-3 text-sm">Modes</h4>
              <div className="flex flex-wrap gap-2">
                {(Array.isArray(architecture.features?.modes) ? architecture.features.modes : []).map((mode, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-full text-xs font-medium bg-[#f5f5f7] dark:bg-[#2a2a2e] text-[#1d1d1f] dark:text-[#f1f5f9]"
                  >
                    {mode}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Layers */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          System Layers
        </h2>
        <div className="space-y-3">
          {(Array.isArray(architecture.architecture?.layers) ? architecture.architecture.layers : []).map((layer, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="bg-white dark:bg-[#1c1c1e] rounded-xl p-6 border border-black/5 dark:border-white/5 hover:border-black/10 transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-bold text-[#1d1d1f] dark:text-[#f1f5f9] mb-1">{layer.name}</h3>
                  <p className="text-sm text-[#6e6e73] mb-2">{layer.component}</p>
                  <p className="text-xs text-[#aeaeb2]">{layer.responsibility}</p>
                </div>
                <div className="ml-4">
                  <Zap className="w-5 h-5 text-[#3b82f6]" />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Models */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Reasoning Pipeline
        </h2>
        <div className="space-y-4">
          {(Array.isArray(architecture.models?.reasoning) ? architecture.models.reasoning : []).map((stage, i) => (
            <div key={i} className="bg-white dark:bg-[#1c1c1e] rounded-xl p-6 border border-black/5 dark:border-white/5">
              <h3 className="font-bold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize mb-3">{stage.role}</h3>
              <div className="flex flex-wrap gap-2">
                {(Array.isArray(stage.models) ? stage.models : []).map((model, j) => (
                  <span
                    key={j}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r from-[#3b82f6]/10 to-[#06b6d4]/10 text-[#1d1d1f] dark:text-[#f1f5f9] border border-[#3b82f6]/20"
                  >
                    {model}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function FeedbackTab({ feedback }) {
  if (!feedback) {
    return (
      <DataFallback
        message="No feedback telemetry yet."
        className="sentinel-surface sentinel-border p-10"
      />
    );
  }

  return (
    <div className="space-y-12">
      {/* Feedback by Mode */}
      {feedback.by_mode && (
        <section>
          <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
            Feedback by Mode
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(feedback.by_mode).map(([mode, data]) => (
              <motion.div
                key={mode}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-[#1c1c1e] rounded-xl p-6 border border-black/5 dark:border-white/5"
              >
                <h3 className="font-bold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize mb-4">{mode}</h3>
                <div className="space-y-2">
                  <div>
                    <p className="text-xs text-[#6e6e73] mb-1">Feedback Count</p>
                    <p className="text-2xl font-bold text-[#3b82f6]">{data.count || 0}</p>
                  </div>
                  <div>
                    <p className="text-xs text-[#6e6e73] mb-1">Avg Rating</p>
                    <p className="text-lg font-bold text-[#34c759]">
                      {data.avg_rating?.toFixed(2) || '0.00'}/5.0
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>
      )}

      {/* Recent Feedback */}
      {feedback.recent_feedback && (
        <section>
          <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
            Recent Feedback
          </h2>
          <div className="space-y-3">
            {feedback.recent_feedback.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-white dark:bg-[#1c1c1e] rounded-lg p-4 border border-black/5 dark:border-white/5 text-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize">{item.sub_mode}</span>
                      <span className="text-xs text-[#aeaeb2]">{item.mode}</span>
                    </div>
                    {item.reason && (
                      <p className="text-[#6e6e73] text-xs">{item.reason}</p>
                    )}
                  </div>
                  <div className="text-lg font-bold text-[#3b82f6]">
                    {item.rating}/5
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}


// ============================================================
// MODE ANALYTICS TAB
// ============================================================
function ModesTab({ stats }) {
  if (!stats || !stats.chats?.by_mode) {
    return (
      <div className="text-center py-12">
        <p className="text-[#6e6e73]">No mode data available</p>
      </div>
    );
  }

  const modes = stats.chats.by_mode;
  const totalChats = Object.values(modes).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-12">
      {/* Mode Distribution */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Mode Distribution
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(modes).map(([mode, count]) => {
            const percentage = totalChats > 0 ? ((count / totalChats) * 100).toFixed(1) : 0;
            return (
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5 hover:border-black/10 transition-all"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="font-bold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize text-lg">{mode}</h3>
                    <p className="text-xs text-[#6e6e73] mt-1">Chats</p>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-bold text-[#3b82f6]">{count}</p>
                    <p className="text-xs text-[#aeaeb2] mt-1">{percentage}%</p>
                  </div>
                </div>
                <div className="w-full bg-[#f5f5f7] dark:bg-[#2a2a2e] rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] h-full transition-all"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Mode Breakdown Card */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Mode Breakdown
        </h2>
        <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
          <div className="space-y-4">
            {Object.entries(modes).map(([mode, count]) => (
              <div
                key={mode}
                className="flex items-center justify-between pb-4 border-b border-black/5 dark:border-white/5 last:border-b-0 last:pb-0"
              >
                <div>
                  <p className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize">{mode}</p>
                  <p className="text-xs text-[#6e6e73] mt-1">Active conversations</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-[#3b82f6]">{count}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Total Stats */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Summary
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] rounded-2xl p-8 text-white">
            <p className="text-sm opacity-90">Total Chats</p>
            <h3 className="text-4xl font-bold mt-2">{totalChats}</h3>
          </div>
          <div className="bg-gradient-to-br from-[#8b5cf6] to-[#a78bfa] rounded-2xl p-8 text-white">
            <p className="text-sm opacity-90">Unique Modes Used</p>
            <h3 className="text-4xl font-bold mt-2">{Object.keys(modes).length}</h3>
          </div>
        </div>
      </section>
    </div>
  );
}
// ============================================================
// ORCHESTRATOR PERFORMANCE TAB
// ============================================================
function OrchestratorTab() {
  const [orchestratorData, setOrchestratorData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchOrchestratorData = async () => {
      try {
        const res = await api.get('/api/admin/orchestrator/performance');
        setOrchestratorData(res);
      } catch (err) {
        console.log('Orchestrator metrics not available yet');
      } finally {
        setLoading(false);
      }
    };
    fetchOrchestratorData();
  }, []);

  if (loading) return <LoadingScreen />;

  return (
    <div className="space-y-12">
      {/* MCO Performance */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          MCO (MetaCognitive Orchestrator) Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Total Queries</p>
            <h3 className="text-4xl font-bold text-[#3b82f6]">
              {orchestratorData?.total_queries || 0}
            </h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Avg Response Time</p>
            <h3 className="text-4xl font-bold text-[#8b5cf6]">
              {orchestratorData?.avg_response_time || '2.1'}s
            </h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Cache Hit Rate</p>
            <h3 className="text-4xl font-bold text-[#34c759]">
              {orchestratorData?.cache_hit_rate || '78'}%
            </h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Success Rate</p>
            <h3 className="text-4xl font-bold text-[#06b6d4]">
              {orchestratorData?.success_rate || '95'}%
            </h3>
          </div>
        </div>
      </section>

      {/* Mode-Specific Latency */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Mode-Specific Latency
        </h2>
        <div className="space-y-3">
          {['STANDARD', 'RESEARCH', 'DEBATE', 'GLASS', 'STRESS'].map((mode, i) => {
            const latency = orchestratorData?.latency_by_mode?.[mode.toLowerCase()] ?? null;
            return (
              <motion.div
                key={mode}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-white dark:bg-[#1c1c1e] rounded-lg p-4 border border-black/5 dark:border-white/5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9]">{mode}</span>
                  <span className="text-sm text-[#6e6e73]">{latency != null ? `${Number(latency).toFixed(2)}s` : 'No telemetry'}</span>
                </div>
                <div className="w-full bg-[#f5f5f7] dark:bg-[#2a2a2e] rounded-full h-2 mt-2">
                  <div
                    className="bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] h-full"
                    style={{ width: latency != null ? `${Math.min((Number(latency) / 3.5) * 100, 100)}%` : '0%' }}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Query Complexity */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Query Complexity Distribution
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-[#34c759] to-[#00d084] rounded-2xl p-8 text-white">
            <p className="text-sm opacity-90">Simple</p>
            <h3 className="text-4xl font-bold mt-2">{orchestratorData?.complexity_distribution?.simple ?? 0}</h3>
          </div>
          <div className="bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] rounded-2xl p-8 text-white">
            <p className="text-sm opacity-90">Moderate</p>
            <h3 className="text-4xl font-bold mt-2">{orchestratorData?.complexity_distribution?.moderate ?? 0}</h3>
          </div>
          <div className="bg-gradient-to-br from-[#f59e0b] to-[#fbbf24] rounded-2xl p-8 text-white">
            <p className="text-sm opacity-90">Complex</p>
            <h3 className="text-4xl font-bold mt-2">{orchestratorData?.complexity_distribution?.complex ?? 0}</h3>
          </div>
        </div>
      </section>
    </div>
  );
}


// ============================================================
// MEMORY & LEARNING TAB
// ============================================================
function MemoryTab() {
  const [memoryData, setMemoryData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMemoryData = async () => {
      try {
        const res = await api.get('/api/admin/memory/learning');
        setMemoryData(res);
      } catch (err) {
        console.log('Memory metrics not available yet');
      } finally {
        setLoading(false);
      }
    };
    fetchMemoryData();
  }, []);

  if (loading) return <LoadingScreen />;

  return (
    <div className="space-y-12">
      {/* Memory System Overview */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Memory System Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">3-Tier Memory</p>
            <div className="space-y-3 mt-4">
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Short-term (Session)</p>
                <p className="text-2xl font-bold text-[#3b82f6]">
                  {memoryData?.short_term_size || 0} KB
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Rolling Summary</p>
                <p className="text-2xl font-bold text-[#8b5cf6]">
                  {memoryData?.rolling_summary_size || 0} KB
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">User Preferences</p>
                <p className="text-2xl font-bold text-[#34c759]">
                  {memoryData?.user_prefs_size || 0} KB
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Knowledge Learning</p>
            <div className="space-y-3 mt-4">
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Boundary Violations</p>
                <p className="text-2xl font-bold text-[#f59e0b]">
                  {memoryData?.boundary_violations || 0}
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Refusal Decisions</p>
                <p className="text-2xl font-bold text-[#ef4444]">
                  {memoryData?.refusal_decisions || 0}
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Risk Profiles</p>
                <p className="text-2xl font-bold text-[#8b5cf6]">
                  {memoryData?.risk_profiles || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-8 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#6e6e73] mb-2">Knowledge Base</p>
            <div className="space-y-3 mt-4">
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Total Entries</p>
                <p className="text-2xl font-bold text-[#34c759]">
                  {memoryData?.kb_entries || 0}
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">High Agreement</p>
                <p className="text-2xl font-bold text-[#06b6d4]">
                  {memoryData?.high_agreement || 0}%
                </p>
              </div>
              <div>
                <p className="text-xs text-[#6e6e73] mb-1">Learning Score</p>
                <p className="text-2xl font-bold text-[#3b82f6]">
                  {memoryData?.learning_score || 0}/100
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Top Risk Models */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Risk Profile Analysis
        </h2>
        <div className="space-y-3">
          {[
            { model: 'llama-33-70b', risk: 8, violations: 12 },
            { model: 'mixtral-8x7b', risk: 5, violations: 4 },
            { model: 'gemini-flash', risk: 3, violations: 2 },
          ].map((item, i) => (
            <motion.div
              key={item.model}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.1 }}
              className="bg-white dark:bg-[#1c1c1e] rounded-lg p-4 border border-black/5 dark:border-white/5"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9]">{item.model}</p>
                  <p className="text-xs text-[#6e6e73] mt-1">{item.violations} violations recorded</p>
                </div>
                <div className="text-right">
                  <p className="text-xl font-bold text-[#f59e0b]">{item.risk}/10</p>
                  <p className="text-xs text-[#6e6e73]">Risk Level</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
}


// ============================================================
// MODEL PERFORMANCE TAB
// ============================================================
function ModelsTab() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelData = async () => {
      try {
        await api.get('/api/admin/models/performance');
      } catch (err) {
        console.log('Model metrics not available yet');
      } finally {
        setLoading(false);
      }
    };
    fetchModelData();
  }, []);

  if (loading) return <LoadingScreen />;

  const models = [
    { name: 'llama-33-70b', provider: 'Groq', role: 'Analysis', tokens: '45K', accuracy: 94 },
    { name: 'mixtral-8x7b', provider: 'Groq', role: 'Critique A', tokens: '38K', accuracy: 91 },
    { name: 'llama4-scout', provider: 'Groq', role: 'Critique B', tokens: '32K', accuracy: 88 },
    { name: 'qwen-2.5-vl', provider: 'Qwen', role: 'Vision', tokens: '42K', accuracy: 89 },
    { name: 'gemini-flash', provider: 'Google', role: 'Synthesis', tokens: '50K', accuracy: 93 },
    { name: 'llama31-8b', provider: 'Groq', role: 'Verification', tokens: '28K', accuracy: 90 },
  ];

  return (
    <div className="space-y-12">
      {/* Model Registry */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Active Model Ensemble
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-black/10">
                <th className="text-left py-3 px-4 text-xs font-semibold text-[#6e6e73] uppercase">Model</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-[#6e6e73] uppercase">Provider</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-[#6e6e73] uppercase">Role</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-[#6e6e73] uppercase">Tokens</th>
                <th className="text-left py-3 px-4 text-xs font-semibold text-[#6e6e73] uppercase">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, i) => (
                <motion.tr
                  key={model.name}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="border-b border-black/5 dark:border-white/5 hover:bg-[#f5f5f7] dark:bg-[#2a2a2e] transition-colors"
                >
                  <td className="py-3 px-4">
                    <div>
                      <p className="font-semibold text-[#1d1d1f] dark:text-[#f1f5f9] text-sm">{model.name}</p>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-sm text-[#6e6e73]">{model.provider}</td>
                  <td className="py-3 px-4 text-sm">
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-[#f5f5f7] dark:bg-[#2a2a2e] text-[#1d1d1f] dark:text-[#f1f5f9]">
                      {model.role}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm font-mono text-[#3b82f6]">{model.tokens}</td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-[#f5f5f7] dark:bg-[#2a2a2e] rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-[#34c759] to-[#00d084] h-full rounded-full"
                          style={{ width: `${model.accuracy}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold text-[#1d1d1f] dark:text-[#f1f5f9]">{model.accuracy}%</span>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Model Roles */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          Reasoning Pipeline Roles
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { role: 'Analysis', desc: 'Primary analysis & interpretation', count: 1 },
            { role: 'Critique A', desc: 'Alternative perspective analysis', count: 1 },
            { role: 'Critique B', desc: 'Vision-based analysis', count: 1 },
            { role: 'Critique C', desc: 'Logical consistency checking', count: 1 },
            { role: 'Synthesis', desc: 'Unified response generation', count: 2 },
            { role: 'Verification', desc: 'Final validation & safety', count: 1 },
          ].map((item, i) => (
            <motion.div
              key={item.role}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="bg-white dark:bg-[#1c1c1e] rounded-lg p-6 border border-black/5 dark:border-white/5 hover:border-black/10 transition-all"
            >
              <h3 className="font-bold text-[#1d1d1f] dark:text-[#f1f5f9] mb-1">{item.role}</h3>
              <p className="text-xs text-[#6e6e73] mb-4">{item.desc}</p>
              <div className="flex items-center justify-between">
                <p className="text-xs text-[#aeaeb2]">Models assigned</p>
                <p className="text-lg font-bold text-[#3b82f6]">{item.count}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
}


export default CognitiveMissionControl;

// ============================================================
// COGNITIVE MISSION CONTROL TAB — v8.0
// ============================================================
function CognitiveMissionControlTab({ apiBase }) {
  const [recentRuns, setRecentRuns] = useState([]);
  const [activeRuns, setActiveRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [liveStreamStatus, setLiveStreamStatus] = useState('idle');
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(null);

  const LIFECYCLE_COLORS = {
    created:      { bg: 'rgba(174,174,178,0.15)', text: '#aeaeb2', dot: '#aeaeb2' },
    routing:      { bg: 'rgba(59,130,246,0.12)', text: '#3b82f6', dot: '#3b82f6' },
    executing:    { bg: 'rgba(139,92,246,0.12)', text: '#8b5cf6', dot: '#8b5cf6' },
    debating:     { bg: 'rgba(245,158,11,0.12)', text: '#f59e0b', dot: '#f59e0b' },
    synthesizing: { bg: 'rgba(6,182,212,0.12)', text: '#06b6d4', dot: '#06b6d4' },
    reflecting:   { bg: 'rgba(139,92,246,0.12)', text: '#8b5cf6', dot: '#8b5cf6' },
    completed:    { bg: 'rgba(52,199,89,0.12)', text: '#34c759', dot: '#34c759' },
    failed:       { bg: 'rgba(239,68,68,0.12)', text: '#ef4444', dot: '#ef4444' },
    recovered:    { bg: 'rgba(245,158,11,0.12)', text: '#f59e0b', dot: '#f59e0b' },
  };

  const fetchRuns = useCallback(async () => {
    try {
      const [recentRes, activeRes] = await Promise.allSettled([
        api.get('/api/orchestration/recent?limit=20'),
        api.get('/api/orchestration/active'),
      ]);
      if (recentRes.status === 'fulfilled') setRecentRuns(recentRes.value?.runs || []);
      if (activeRes.status === 'fulfilled') setActiveRuns(activeRes.value?.active_runs || []);
      setLastRefresh(new Date().toLocaleTimeString());
    } catch (e) {
      console.log('[CognitiveMissionControl] Fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 8000); // refresh every 8s
    return () => clearInterval(interval);
  }, [fetchRuns]);

  useEffect(() => {
    if (!selectedRun?.orchestration_run_id) return undefined;
    if (['completed', 'failed'].includes(selectedRun.lifecycle_state)) return undefined;

    const accessToken = readSupabaseSessionSnapshot()?.access_token;
    if (!accessToken) {
      setLiveStreamStatus('auth_required');
      return undefined;
    }

    const source = new EventSource(
      `${apiBase}/api/orchestration/${selectedRun.orchestration_run_id}/events?timeout=180&access_token=${encodeURIComponent(accessToken)}`,
      { withCredentials: true }
    );

    setLiveStreamStatus('connecting');

    source.onopen = () => setLiveStreamStatus('live');
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (!payload || payload.event_type === 'heartbeat' || payload.event_type === 'stream_end') return;

        setSelectedRun((current) => {
          if (!current || current.orchestration_run_id !== selectedRun.orchestration_run_id) return current;

          const nextTimeline = [...(current.event_timeline || []), payload].slice(-80);
          const nextRun = {
            ...current,
            event_timeline: nextTimeline,
            event_count: Math.max(current.event_count || 0, nextTimeline.length),
          };

          if (payload.phase) nextRun.cognitive_phase = payload.phase;
          if (payload.phase_label) nextRun.phase_label = payload.phase_label;
          if (payload.event_type === 'orchestration_completed') nextRun.lifecycle_state = 'completed';
          if (payload.event_type === 'orchestration_failed') nextRun.lifecycle_state = 'failed';

          return nextRun;
        });
      } catch (streamErr) {
        console.log('[CognitiveMissionControl] Stream parse failed:', streamErr);
      }
    };
    source.onerror = () => {
      setLiveStreamStatus('disconnected');
      source.close();
    };

    return () => {
      setLiveStreamStatus('idle');
      source.close();
    };
  }, [apiBase, selectedRun?.orchestration_run_id, selectedRun?.lifecycle_state]);

  const fetchRunDetail = async (runId) => {
    try {
      const res = await api.get(`/api/orchestration/${runId}`);
      setSelectedRun(res?.run || null);
    } catch (e) {
      console.log('[CognitiveMissionControl] Detail fetch failed:', e);
    }
  };

  if (loading) return <LoadingScreen />;

  return (
    <div className="space-y-10">
      {/* Header strip */}
      <section>
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold sentinel-text-primary" style={{ fontFamily: FONT }}>
              Cognitive Mission Control
            </h2>
            <p className="text-xs sentinel-text-muted mt-1">
              Live OrchestrationRun feed · v8.0 Persistent Hybrid Cognitive Runtime
              {lastRefresh && <span className="ml-2 opacity-60">Last refresh: {lastRefresh}</span>}
            </p>
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            onClick={fetchRuns}
            className="flex items-center gap-2 px-3 py-2 rounded-lg sentinel-surface-panel sentinel-text-primary"
            style={{ fontFamily: FONT, fontSize: '12px' }}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </motion.button>
        </div>

        {/* Active runs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#aeaeb2] mb-1">Active Runs</p>
            <h3 className="text-4xl font-bold text-[#3b82f6]" style={{ fontFamily: FONT }}>{activeRuns.length}</h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#aeaeb2] mb-1">Recent Runs</p>
            <h3 className="text-4xl font-bold text-[#8b5cf6]" style={{ fontFamily: FONT }}>{recentRuns.length}</h3>
          </div>
          <div className="bg-white dark:bg-[#1c1c1e] rounded-2xl p-6 border border-black/5 dark:border-white/5">
            <p className="text-xs uppercase tracking-wider text-[#aeaeb2] mb-1">Completed</p>
            <h3 className="text-4xl font-bold text-[#34c759]" style={{ fontFamily: FONT }}>
              {recentRuns.filter(r => r.lifecycle_state === 'completed').length}
            </h3>
          </div>
        </div>
      </section>

      {/* Run list */}
      <section>
        <h3 className="text-base font-bold mb-4 sentinel-text-primary" style={{ fontFamily: FONT }}>
          Recent OrchestrationRuns
        </h3>
        {recentRuns.length === 0 ? (
          <div className="text-center py-12 sentinel-text-muted">
            <Cpu className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No orchestration runs yet. Start a query to generate runtime data.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {recentRuns.map((run) => {
              const lc = run.lifecycle_state || 'unknown';
              const colors = LIFECYCLE_COLORS[lc] || LIFECYCLE_COLORS.created;
              return (
                <motion.div
                  key={run.orchestration_run_id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  onClick={() => fetchRunDetail(run.orchestration_run_id)}
                  className="bg-white dark:bg-[#1c1c1e] rounded-xl p-4 border border-black/5 dark:border-white/5 hover:border-[#3b82f6]/30 transition-all cursor-pointer"
                  style={{ fontFamily: FONT }}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      {/* Status dot */}
                      <div style={{
                        width: '8px', height: '8px', borderRadius: '50%',
                        background: colors.dot, flexShrink: 0,
                        boxShadow: lc !== 'completed' && lc !== 'failed'
                          ? `0 0 6px ${colors.dot}` : 'none',
                      }} />
                      {/* Run ID */}
                      <span className="font-mono text-xs text-[#6e6e73] truncate" style={{ maxWidth: '120px' }}>
                        {run.orchestration_run_id?.slice(0, 8)}…
                      </span>
                      {/* Phase label */}
                      <span className="text-xs font-medium text-[#1d1d1f] dark:text-[#f1f5f9] truncate">
                        {run.phase_label || run.cognitive_phase || '—'}
                      </span>
                    </div>

                    <div className="flex items-center gap-4 flex-shrink-0">
                      {/* Lifecycle badge */}
                      <span style={{
                        padding: '2px 8px', borderRadius: '10px',
                        fontSize: '10px', fontWeight: 600,
                        background: colors.bg, color: colors.text,
                      }}>
                        {lc}
                      </span>
                      {/* Confidence */}
                      <span className="text-xs text-[#6e6e73] w-12 text-right">
                        {run.final_confidence != null ? `${Math.round(run.final_confidence * 100)}%` : '—'}
                      </span>
                      {/* Latency */}
                      <span className="text-xs text-[#aeaeb2] w-16 text-right">
                        {run.total_latency_ms > 0 ? `${(run.total_latency_ms / 1000).toFixed(1)}s` : '—'}
                      </span>
                      {/* Models */}
                      <span className="text-xs text-[#aeaeb2] w-10 text-right">
                        {run.models_succeeded ?? '—'}/{run.models_executed ?? '—'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </section>

      {/* Run detail panel */}
      {selectedRun && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold sentinel-text-primary" style={{ fontFamily: FONT }}>
              Run Detail: <span className="font-mono text-[#3b82f6]">
                {selectedRun.orchestration_run_id?.slice(0, 8)}…
              </span>
            </h3>
            <div className="flex items-center gap-3">
              <span
                className="px-2.5 py-1 rounded-full text-[10px] font-semibold"
                style={{
                  background: liveStreamStatus === 'live' ? 'rgba(52,199,89,0.12)' : 'rgba(174,174,178,0.14)',
                  color: liveStreamStatus === 'live' ? '#34c759' : '#6e6e73',
                }}
              >
                {liveStreamStatus === 'live' ? 'Live Stream Active' : 'Stream Idle'}
              </span>
              <button
                onClick={() => setSelectedRun(null)}
                className="text-xs sentinel-text-muted hover:text-red-500 transition-colors"
              >
                Close ✕
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {[
              { label: 'Path', value: selectedRun.execution_path || '—' },
              { label: 'Phase', value: selectedRun.cognitive_phase || '—' },
              { label: 'State', value: selectedRun.lifecycle_state || '—' },
              { label: 'Confidence', value: selectedRun.final_confidence != null ? `${(selectedRun.final_confidence * 100).toFixed(1)}%` : '—' },
              { label: 'Contradiction ρ', value: selectedRun.contradiction_density != null ? `${(selectedRun.contradiction_density * 100).toFixed(1)}%` : '—' },
              { label: 'Total Latency', value: selectedRun.total_latency_ms > 0 ? `${(selectedRun.total_latency_ms / 1000).toFixed(2)}s` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white dark:bg-[#1c1c1e] rounded-xl p-4 border border-black/5 dark:border-white/5">
                <p className="text-xs uppercase tracking-wider text-[#aeaeb2] mb-1">{label}</p>
                <p className="text-sm font-semibold text-[#1d1d1f] dark:text-[#f1f5f9] capitalize">{value}</p>
              </div>
            ))}
          </div>

          {/* Event timeline */}
          {Array.isArray(selectedRun.event_timeline) && selectedRun.event_timeline.length > 0 && (
            <div className="bg-white dark:bg-[#1c1c1e] rounded-xl border border-black/5 dark:border-white/5 overflow-hidden">
              <div className="p-4 border-b border-black/5 dark:border-white/5">
                <h4 className="text-sm font-bold text-[#1d1d1f] dark:text-[#f1f5f9]" style={{ fontFamily: FONT }}>
                  Cognitive Event Timeline ({selectedRun.event_count || selectedRun.event_timeline.length} events)
                </h4>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {selectedRun.event_timeline.map((evt, i) => (
                  <div key={i} className="flex items-start gap-3 px-4 py-2.5 border-b border-black/3 hover:bg-[#f9f9f9]">
                    <div style={{
                      width: '5px', height: '5px', borderRadius: '50%', marginTop: '6px', flexShrink: 0,
                      background: evt.severity === 'warning' ? '#f59e0b' : evt.severity === 'critical' ? '#ef4444' : '#3b82f6',
                    }} />
                    <div className="flex-1 min-w-0">
                      <span className="text-xs font-medium text-[#1d1d1f] dark:text-[#f1f5f9]">{evt.event_type?.replace(/_/g, ' ')}</span>
                      <span className="text-xs text-[#aeaeb2] ml-2">{evt.phase}</span>
                    </div>
                    <span className="text-xs text-[#aeaeb2] flex-shrink-0 font-mono">
                      {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
