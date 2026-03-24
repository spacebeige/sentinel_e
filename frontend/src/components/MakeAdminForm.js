/**
 * ============================================================
 * Make Admin Component — Promote User to Admin Role
 * ============================================================
 * Allows admins to promote other users to admin status
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import axios from 'axios';

const FONT = "'Inter', -apple-system, sans-serif";

export function MakeAdminForm({ onSuccess }) {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatus(null);

    try {
      const token = localStorage.getItem('access_token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      const response = await axios.post(
        `${API_BASE}/api/admin/users/make-admin`,
        { email },
        { headers }
      );

      setStatus({
        type: 'success',
        message: `✓ ${email} is now an admin`,
        data: response.data,
      });
      setEmail('');

      if (onSuccess) onSuccess(response.data);
    } catch (error) {
      setStatus({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to promote user',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl p-8 border border-black/5 max-w-2xl"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center">
          <Shield className="w-5 h-5 text-purple-600" />
        </div>
        <h2 className="text-2xl font-bold text-[#1d1d1f]" style={{ fontFamily: FONT }}>
          Promote to Admin
        </h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-[#1d1d1f] mb-2">
            Email Address
          </label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="user@example.com"
            required
            disabled={loading}
            className="w-full px-4 py-2.5 rounded-lg border border-black/10 bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ fontFamily: FONT }}
          />
          <p className="text-xs text-[#6e6e73] mt-1">
            The user's email must exist in the system
          </p>
        </div>

        {status && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`p-4 rounded-lg border flex items-start gap-3 ${
              status.type === 'success'
                ? 'bg-green-50 border-green-200'
                : 'bg-red-50 border-red-200'
            }`}
          >
            {status.type === 'success' ? (
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            )}
            <div>
              <p
                style={{ fontFamily: FONT }}
                className={status.type === 'success' ? 'text-green-700 font-medium' : 'text-red-700 font-medium'}
              >
                {status.message}
              </p>
              {status.data && (
                <p className="text-xs mt-1 text-[#6e6e73]">
                  Status: {status.data.status} • Role: {status.data.role}
                </p>
              )}
            </div>
          </motion.div>
        )}

        <button
          type="submit"
          disabled={!email || loading}
          className="w-full px-4 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-purple-700 text-white font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
          style={{ fontFamily: FONT }}
        >
          {loading ? (
            <>
              <Loader className="w-4 h-4 animate-spin" />
              Promoting...
            </>
          ) : (
            <>
              <Shield className="w-4 h-4" />
              Make Admin
            </>
          )}
        </button>
      </form>

      <div className="mt-6 p-4 rounded-lg bg-purple-50 border border-purple-200">
        <p className="text-sm text-purple-900" style={{ fontFamily: FONT }}>
          <strong>Admin permissions:</strong> Access to system analytics, user management, architecture overview, and feedback analysis.
        </p>
      </div>
    </motion.div>
  );
}

export default MakeAdminForm;
