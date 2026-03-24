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
  Sigma
} from 'lucide-react';
import axios from 'axios';
import MakeAdminForm from '../components/MakeAdminForm';

const FONT = "'Inter', -apple-system, sans-serif";

const AdminDashboard = () => {
  const [systemStats, setSystemStats] = useState(null);
  const [architecture, setArchitecture] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchAdminData = useCallback(async () => {
    try {
      setError(null);
      const token = localStorage.getItem('access_token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      const [statsRes, archRes, analyticsRes, feedbackRes] = await Promise.all([
        axios.get(`${API_BASE}/api/admin/system/stats`, { headers }).catch(e => ({ status: 500 })),
        axios.get(`${API_BASE}/api/admin/system/architecture`, { headers }).catch(e => ({ status: 500 })),
        axios.get(`${API_BASE}/api/admin/web-analytics?days=7`, { headers }).catch(e => ({ status: 500 })),
        axios.get(`${API_BASE}/api/admin/feedback-summary`, { headers }).catch(e => ({ status: 500 })),
      ]);

      if (statsRes.data) setSystemStats(statsRes.data);
      if (archRes.data) setArchitecture(archRes.data);
      if (analyticsRes.data) setAnalytics(analyticsRes.data);
      if (feedbackRes.data) setFeedback(feedbackRes.data);
    } catch (err) {
      setError('Failed to load admin data');
    } finally {
      setLoading(false);
    }
  }, [API_BASE]);

  useEffect(() => {
    fetchAdminData();
    const interval = setInterval(fetchAdminData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchAdminData]);

  if (loading) return <LoadingScreen />;

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#f5f5f7' }}>
      {/* Header */}
      <header className="bg-white border-b border-black/5 sticky top-14 z-40">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div
                  className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center"
                >
                  <Sigma className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1
                    className="text-2xl text-[#1d1d1f] font-bold"
                    style={{ fontFamily: FONT }}
                  >
                    Admin Control Center
                  </h1>
                  <p className="text-xs text-[#aeaeb2]">System Architecture & Analytics</p>
                </div>
              </div>
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={fetchAdminData}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#f5f5f7] hover:bg-[#e8e8ed] transition-colors"
              style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </motion.button>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mt-6 border-b border-black/5">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'analytics', label: 'Analytics', icon: TrendingUp },
              { id: 'architecture', label: 'Architecture', icon: Zap },
              { id: 'feedback', label: 'Feedback', icon: BarChart3 },
              { id: 'users', label: 'Users', icon: Users },
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-[#3b82f6] text-[#3b82f6]'
                      : 'border-transparent text-[#6e6e73] hover:text-[#1d1d1f]'
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

        {/* Architecture Tab */}
        {activeTab === 'architecture' && <ArchitectureTab architecture={architecture} />}

        {/* Feedback Tab */}
        {activeTab === 'feedback' && <FeedbackTab feedback={feedback} />}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="space-y-12">
            <section>
              <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
                User Management
              </h2>
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

function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        <Sigma className="w-12 h-12 text-[#3b82f6]" />
      </motion.div>
    </div>
  );
}

function OverviewTab({ stats }) {
  if (!stats) return <div>No data</div>;

  const StatCard = ({ label, value, subtext, icon: Icon, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl p-6 border border-black/5 hover:border-black/10 transition-all hover:shadow-lg"
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
                className="bg-white rounded-xl p-6 border border-black/5"
              >
                <h3 className="font-semibold text-[#1d1d1f] capitalize mb-2">{mode}</h3>
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
            <div key={item.label} className="bg-white rounded-xl p-6 border border-black/5">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-[#1d1d1f]">{item.label}</h3>
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
  if (!analytics) return <div>No analytics data</div>;

  return (
    <div className="space-y-12">
      {/* Summary */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          7-Day Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-2xl p-8 border border-black/5">
            <p className="text-sm uppercase tracking-wider text-[#6e6e73] mb-2">Total Sessions</p>
            <h3 className="text-4xl font-bold" style={{ fontFamily: FONT }}>
              {analytics.summary?.total_sessions || 0}
            </h3>
          </div>
          <div className="bg-white rounded-2xl p-8 border border-black/5">
            <p className="text-sm uppercase tracking-wider text-[#6e6e73] mb-2">Unique Users</p>
            <h3 className="text-4xl font-bold" style={{ fontFamily: FONT }}>
              {analytics.summary?.unique_users || 0}
            </h3>
          </div>
          <div className="bg-white rounded-2xl p-8 border border-black/5">
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
  if (!architecture) return <div>No architecture data</div>;

  return (
    <div className="space-y-12">
      {/* System Overview */}
      <section>
        <h2 className="text-xl font-bold mb-6" style={{ fontFamily: FONT }}>
          System Overview
        </h2>
        <div className="bg-white rounded-2xl p-8 border border-black/5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-bold text-xl mb-2">{architecture.system?.name}</h3>
              <p className="text-[#6e6e73] text-sm mb-4">{architecture.system?.description}</p>
              <p className="text-xs text-[#aeaeb2]">Version {architecture.system?.version}</p>
            </div>
            <div>
              <h4 className="font-bold mb-3 text-sm">Key Features</h4>
              <ul className="space-y-2">
                {architecture.features?.capabilities?.slice(0, 4).map((cap, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-[#34c759] flex-shrink-0 mt-0.5" />
                    <span className="text-[#1d1d1f]">{cap}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-3 text-sm">Modes</h4>
              <div className="flex flex-wrap gap-2">
                {architecture.features?.modes?.map((mode, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-full text-xs font-medium bg-[#f5f5f7] text-[#1d1d1f]"
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
          {architecture.architecture?.layers?.map((layer, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="bg-white rounded-xl p-6 border border-black/5 hover:border-black/10 transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-bold text-[#1d1d1f] mb-1">{layer.name}</h3>
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
          {architecture.models?.reasoning?.map((stage, i) => (
            <div key={i} className="bg-white rounded-xl p-6 border border-black/5">
              <h3 className="font-bold text-[#1d1d1f] capitalize mb-3">{stage.role}</h3>
              <div className="flex flex-wrap gap-2">
                {stage.models?.map((model, j) => (
                  <span
                    key={j}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r from-[#3b82f6]/10 to-[#06b6d4]/10 text-[#1d1d1f] border border-[#3b82f6]/20"
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
  if (!feedback) return <div>No feedback data</div>;

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
                className="bg-white rounded-xl p-6 border border-black/5"
              >
                <h3 className="font-bold text-[#1d1d1f] capitalize mb-4">{mode}</h3>
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
                className="bg-white rounded-lg p-4 border border-black/5 text-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-[#1d1d1f] capitalize">{item.sub_mode}</span>
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

export default AdminDashboard;
