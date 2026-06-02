import React, { useState, useEffect } from 'react';
import { useAuthContext } from '../providers/AuthProvider';
import { Shield, Activity, Users, Database, MessageSquare, Clock, Cpu, BarChart3 } from 'lucide-react';
import { getAdminAnalytics, AdminAnalytics } from '../services/analyticsService';
import { getAdminRequestStatus } from '../api';
import { Navigate, Link } from 'react-router';

const AdminPage: React.FC = () => {
  const { user, isAdmin, role } = useAuthContext();
  const [analytics, setAnalytics] = useState<AdminAnalytics | null>(null);
  const [requestStatus, setRequestStatus] = useState<'pending' | 'approved' | 'rejected' | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  useEffect(() => {
    document.title = "Admin • Sentinel-E";
    let mounted = true;

    if (role === 'admin' || role === 'owner') {
      setStatusLoading(false);
      const fetchAnalytics = async () => {
        const data = await getAdminAnalytics();
        if (mounted) setAnalytics(data);
      };
      fetchAnalytics();
      const interval = setInterval(fetchAnalytics, 10000);
      return () => {
        mounted = false;
        clearInterval(interval);
      };
    } else {
      const fetchStatus = async () => {
        if (user?.email) {
          const status = await getAdminRequestStatus(user.email);
          if (mounted) setRequestStatus(status);
        }
        if (mounted) setStatusLoading(false);
      };
      fetchStatus();
    }

    return () => {
      mounted = false;
    };
  }, [role, user?.email]);

  // Redundant email check removed: authorization is handled by ProtectedRoute



  if (statusLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#09090b]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-[#8b5cf6]"></div>
          <span className="text-sm font-medium text-zinc-400">Verifying authorization...</span>
        </div>
      </div>
    );
  }

  if (role !== 'admin' && role !== 'owner') {
    return (
      <div className="flex h-screen w-full flex-col items-center justify-center bg-[#09090b] text-white">
        <Link to="/" className="fixed top-8 left-8 z-50 transition-transform hover:scale-105">
          <img src="/logo.png" alt="Sentinel-E" className="h-7 w-auto" />
        </Link>
        <div className="w-full max-w-md p-8 rounded-2xl shadow-2xl text-center" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
          <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-white/5 border border-white/10">
            <Shield className="h-8 w-8 text-[#8b5cf6]" />
          </div>
          <h2 className="text-xl font-bold mb-3 text-[#f5f5f7]">Access Denied</h2>
          <p className="text-sm text-zinc-400 mb-8">
            {requestStatus === 'pending' 
              ? "Your admin request is pending approval."
              : "Admin access has not been approved."}
          </p>
          <Link to="/chat" className="inline-flex w-full items-center justify-center rounded-xl py-3 text-[14px] font-medium bg-[#f5f5f7] text-[#1d1d1f] transition-all hover:scale-[1.01] active:scale-[0.99]">
            Return to Chat
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-y-auto bg-[#09090b] text-white">
      <div className="mx-auto w-full max-w-7xl p-8">
        <div className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center">
              <Shield className="mr-3 h-8 w-8 text-[#8b5cf6]" />
              Sentinel-E Admin Portal
            </h1>
            <p className="mt-2 text-sm text-zinc-400">
              Welcome back, {user?.email}
            </p>
          </div>
          <div className="rounded-full bg-emerald-500/10 px-4 py-1 text-sm font-medium text-emerald-400 border border-emerald-500/20">
            System Online
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-4 mb-6">
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Activity className="h-5 w-5 text-indigo-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Daily Active Users</h3>
            </div>
            <p className="text-3xl font-bold text-white">{analytics?.dailyUsers || 0}</p>
            <p className="mt-2 text-sm text-zinc-500">Unique logins today</p>
          </div>
          
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Users className="h-5 w-5 text-emerald-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Active Sessions</h3>
            </div>
            <p className="text-3xl font-bold text-white">{analytics?.activeUsers || 0}</p>
            <p className="mt-2 text-sm text-zinc-500">Currently connected</p>
          </div>
          
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <MessageSquare className="h-5 w-5 text-sky-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Messages Today</h3>
            </div>
            <p className="text-3xl font-bold text-white">{analytics?.messagesToday || 0}</p>
            <p className="mt-2 text-sm text-zinc-500">Total volume across users</p>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Clock className="h-5 w-5 text-amber-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Avg Session Length</h3>
            </div>
            <p className="text-3xl font-bold text-white">{analytics?.averageSessionLength || 0}m</p>
            <p className="mt-2 text-sm text-zinc-500">Per user session duration</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-6">
              <Cpu className="h-5 w-5 text-purple-400 mr-2" />
              <h2 className="text-xl font-medium text-white">Top Models Utilized</h2>
            </div>
            <div className="space-y-4">
              {analytics?.topModels.map((model, i) => (
                <div key={i} className="flex items-center justify-between border-b border-white/5 pb-4 last:border-0 last:pb-0">
                  <div className="flex items-center">
                    <div className="mr-4 h-2 w-2 rounded-full bg-purple-500"></div>
                    <div>
                      <p className="text-sm font-medium text-white">{model.name}</p>
                    </div>
                  </div>
                  <span className="text-sm font-bold text-white">{model.count} msgs</span>
                </div>
              ))}
              {(!analytics?.topModels || analytics.topModels.length === 0) && (
                <p className="text-sm text-zinc-500">No data available yet</p>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-6">
              <BarChart3 className="h-5 w-5 text-pink-400 mr-2" />
              <h2 className="text-xl font-medium text-white">Top Orchestration Modes</h2>
            </div>
            <div className="space-y-4">
              {analytics?.topModes.map((mode, i) => (
                <div key={i} className="flex items-center justify-between border-b border-white/5 pb-4 last:border-0 last:pb-0">
                  <div className="flex items-center">
                    <div className="mr-4 h-2 w-2 rounded-full bg-pink-500"></div>
                    <div>
                      <p className="text-sm font-medium text-white">{mode.name}</p>
                    </div>
                  </div>
                  <span className="text-sm font-bold text-white">{mode.count} msgs</span>
                </div>
              ))}
              {(!analytics?.topModes || analytics.topModes.length === 0) && (
                <p className="text-sm text-zinc-500">No data available yet</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminPage;
