import React, { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { Shield, Check, X, Loader2, AlertCircle } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';
import { useAuthContext } from '../providers/AuthProvider';

interface AdminRequest {
  id: string;
  name: string;
  email: string;
  organization: string;
  reason: string;
  status: 'pending' | 'approved' | 'rejected';
  submitted_at: string;
}

export default function AdminRequestsPage() {
  const [requests, setRequests] = useState<AdminRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  
  const { theme } = useTheme();
  const { user } = useAuthContext();
  const isDark = theme === "dark";

  useEffect(() => {
    fetchRequests();
  }, []);

  const fetchRequests = async () => {
    try {
      const { data, error } = await supabase
        .from('admin_requests')
        .select('*')
        .order('submitted_at', { ascending: false });

      if (error) throw error;
      setRequests(data || []);
    } catch (err: any) {
      console.error('Error fetching admin requests:', err);
      setError('Failed to load admin requests. Ensure you have owner permissions.');
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (requestId: string, email: string, action: 'approve' | 'reject') => {
    setActionLoading(requestId);
    try {
      if (action === 'approve') {
        // 1. Insert into admin_users
        const { error: insertError } = await supabase.from('admin_users').insert({
          email: email,
          role: 'admin',
          created_by: user?.id
        });
        
        if (insertError) throw insertError;
        
        // 2. Update profiles to admin if profile exists
        // This is optional but good for syncing roles
        await supabase.from('profiles').update({ role: 'admin' }).eq('email', email);
      }
      
      // 3. Update request status
      const { error: updateError } = await supabase
        .from('admin_requests')
        .update({ status: action === 'approve' ? 'approved' : 'rejected' })
        .eq('id', requestId);

      if (updateError) throw updateError;
      
      // 4. Update local state
      setRequests(prev => prev.map(req => 
        req.id === requestId 
          ? { ...req, status: action === 'approve' ? 'approved' : 'rejected' } 
          : req
      ));
      
    } catch (err: any) {
      console.error(`Error ${action}ing request:`, err);
      alert(`Failed to ${action} request: ${err.message}`);
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="animate-spin text-indigo-500 w-8 h-8" />
      </div>
    );
  }

  return (
    <div className={`min-h-screen p-8 pt-24 ${isDark ? 'text-white' : 'text-black'}`}>
      <div className="max-w-5xl mx-auto space-y-8">
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-indigo-500" />
          <h1 className="text-3xl font-bold">Admin Requests</h1>
        </div>

        {error ? (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 flex items-center gap-3">
            <AlertCircle size={20} />
            {error}
          </div>
        ) : (
          <div className="space-y-4">
            {requests.length === 0 ? (
              <p className="text-sm opacity-60">No admin requests found.</p>
            ) : (
              requests.map(req => (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  key={req.id} 
                  className="p-6 rounded-2xl flex flex-col md:flex-row gap-6 md:items-center justify-between"
                  style={{
                    background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
                    border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                  }}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-lg">{req.name}</h3>
                      <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        req.status === 'pending' ? 'bg-amber-500/20 text-amber-500 border border-amber-500/20' :
                        req.status === 'approved' ? 'bg-emerald-500/20 text-emerald-500 border border-emerald-500/20' :
                        'bg-red-500/20 text-red-500 border border-red-500/20'
                      }`}>
                        {req.status.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-sm opacity-70">{req.email} {req.organization ? `• ${req.organization}` : ''}</p>
                    <p className="text-sm opacity-90 mt-2">"{req.reason}"</p>
                    <p className="text-xs opacity-50 mt-2">Submitted: {new Date(req.submitted_at).toLocaleString()}</p>
                  </div>
                  
                  {req.status === 'pending' && (
                    <div className="flex gap-2">
                      <button
                        disabled={actionLoading === req.id}
                        onClick={() => handleAction(req.id, req.email, 'approve')}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 transition-colors border border-emerald-500/20 disabled:opacity-50"
                      >
                        {actionLoading === req.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                        Approve
                      </button>
                      <button
                        disabled={actionLoading === req.id}
                        onClick={() => handleAction(req.id, req.email, 'reject')}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-colors border border-red-500/20 disabled:opacity-50"
                      >
                        {actionLoading === req.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <X className="w-4 h-4" />}
                        Reject
                      </button>
                    </div>
                  )}
                </motion.div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
