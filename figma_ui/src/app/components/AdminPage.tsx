import React from 'react';
import { useAuthContext } from '../providers/AuthProvider';
import { Shield, Activity, Users, Database } from 'lucide-react';

const AdminPage: React.FC = () => {
  const { user } = useAuthContext();

  return (
    <div className="flex h-screen w-screen flex-col overflow-y-auto bg-[#09090b] text-white">
      <div className="mx-auto w-full max-w-7xl p-8">
        <div className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center">
              <Shield className="mr-3 h-8 w-8 text-[#8b5cf6]" />
              Sentinel-E Admin Portal
            </h1>
            <p className="mt-2 text-sm text-zinc-400">
              Welcome back, Commander {user?.email}
            </p>
          </div>
          <div className="rounded-full bg-green-500/10 px-4 py-1 text-sm font-medium text-green-400 border border-green-500/20">
            System Online
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3 mb-10">
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Activity className="h-5 w-5 text-zinc-400 mr-2" />
              <h3 className="text-lg font-medium text-white">System Status</h3>
            </div>
            <p className="text-3xl font-bold text-white">Optimal</p>
            <p className="mt-2 text-sm text-zinc-500">All cognitive engines operational</p>
          </div>
          
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Users className="h-5 w-5 text-zinc-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Active Sessions</h3>
            </div>
            <p className="text-3xl font-bold text-white">24</p>
            <p className="mt-2 text-sm text-zinc-500">Authorized personnel connected</p>
          </div>
          
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center mb-4">
              <Database className="h-5 w-5 text-zinc-400 mr-2" />
              <h3 className="text-lg font-medium text-white">Database Health</h3>
            </div>
            <p className="text-3xl font-bold text-white">99.9%</p>
            <p className="mt-2 text-sm text-zinc-500">Neon Postgres responding</p>
          </div>
        </div>

        <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
          <h2 className="text-xl font-medium text-white mb-6">Recent Security Events</h2>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between border-b border-white/5 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center">
                  <div className="mr-4 h-2 w-2 rounded-full bg-[#8b5cf6]"></div>
                  <div>
                    <p className="text-sm font-medium text-white">Admin access granted</p>
                    <p className="text-xs text-zinc-500">{user?.email}</p>
                  </div>
                </div>
                <span className="text-xs text-zinc-500">Just now</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminPage;
