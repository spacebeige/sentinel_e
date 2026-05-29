import React, { useState, useEffect } from 'react';
import { useSupabaseAuth } from '../hooks/useSupabaseAuth';
import { supabase } from '../lib/supabase';
import { User, Mail, Calendar, Edit2, Shield, Crown } from 'lucide-react';
import { useTheme } from 'next-themes';

export default function ProfilePage() {
  const { user, isAdmin } = useSupabaseAuth();
  const [customName, setCustomName] = useState('');
  const [isEditingName, setIsEditingName] = useState(false);
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  useEffect(() => {
    if (user) {
      setCustomName(user.user_metadata?.custom_name || user.user_metadata?.full_name || 'Sentinel User');
    }
  }, [user]);

  const handleSaveName = async () => {
    if (!user) return;
    try {
      await supabase.auth.updateUser({
        data: { custom_name: customName }
      });
      setIsEditingName(false);
    } catch (error) {
      console.error('Failed to update name', error);
    }
  };

  if (!user) return null;

  const surfaceClass = isDark ? 'bg-white/5 border-white/10' : 'bg-black/5 border-black/5';
  const textClass = isDark ? 'text-white' : 'text-black';
  const textMutedClass = isDark ? 'text-white/60' : 'text-black/60';

  const subscription = user.user_metadata?.subscription || 'standard';

  return (
    <div className={`min-h-screen ${isDark ? 'bg-[#09090b]' : 'bg-[#f5f5f7]'} p-8 md:p-12 overflow-y-auto`}>
      <div className="max-w-3xl mx-auto space-y-8">
        <h1 className={`text-3xl font-bold flex items-center gap-3 ${textClass}`}>
          <User className="w-8 h-8 text-indigo-500" />
          Commander Profile
        </h1>

        <div className={`p-8 rounded-3xl border ${surfaceClass}`}>
          <div className="flex flex-col md:flex-row items-center md:items-start gap-8">
            <div className="w-32 h-32 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-5xl font-bold shrink-0 shadow-lg">
              {customName.charAt(0).toUpperCase()}
            </div>
            
            <div className="flex-1 w-full space-y-6">
              <div>
                <label className={`text-xs font-semibold uppercase tracking-wider ${textMutedClass} mb-2 block`}>Display Name</label>
                {isEditingName ? (
                  <div className="flex items-center gap-3 w-full">
                    <input 
                      type="text" 
                      value={customName}
                      onChange={(e) => setCustomName(e.target.value)}
                      className={`flex-1 bg-transparent border-b-2 border-indigo-500 outline-none ${textClass} font-semibold text-2xl pb-1`}
                      autoFocus
                    />
                    <button onClick={handleSaveName} className="bg-indigo-500 hover:bg-indigo-600 transition-colors text-white px-4 py-2 rounded-xl font-medium">Save</button>
                    <button onClick={() => setIsEditingName(false)} className={`px-4 py-2 rounded-xl font-medium ${textMutedClass} hover:text-red-400 transition-colors`}>Cancel</button>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <h3 className={`text-2xl font-semibold ${textClass}`}>{customName}</h3>
                    <button onClick={() => setIsEditingName(true)} className={`${textMutedClass} hover:text-indigo-500 transition-colors bg-white/5 p-2 rounded-full`}>
                      <Edit2 className="w-4 h-4" />
                    </button>
                  </div>
                )}
                <p className={`text-sm mt-2 ${textMutedClass}`}>Used across all Sentinel-E cognitive interfaces.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/5">
                <div>
                  <label className={`text-xs font-semibold uppercase tracking-wider ${textMutedClass} mb-1 block`}>Primary Email</label>
                  <div className={`flex items-center gap-2 font-medium ${textClass}`}>
                    <Mail className={`w-4 h-4 ${textMutedClass}`} /> {user.email}
                  </div>
                </div>
                
                <div>
                  <label className={`text-xs font-semibold uppercase tracking-wider ${textMutedClass} mb-1 block`}>Account Created</label>
                  <div className={`flex items-center gap-2 font-medium ${textClass}`}>
                    <Calendar className={`w-4 h-4 ${textMutedClass}`} /> {new Date(user.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className={`p-6 rounded-3xl border ${surfaceClass} flex items-center gap-4`}>
            <div className={`w-14 h-14 rounded-full flex items-center justify-center ${isAdmin ? 'bg-emerald-500/10 text-emerald-500' : 'bg-white/5 text-white/50'}`}>
              <Shield className="w-7 h-7" />
            </div>
            <div>
              <h4 className={`text-sm font-semibold uppercase tracking-wider ${textMutedClass}`}>System Role</h4>
              <p className={`text-xl font-bold ${isAdmin ? 'text-emerald-500' : textClass}`}>
                {isAdmin ? 'Administrator' : 'Standard Operator'}
              </p>
            </div>
          </div>
          
          <div className={`p-6 rounded-3xl border ${surfaceClass} flex items-center gap-4`}>
            <div className={`w-14 h-14 rounded-full flex items-center justify-center ${subscription === 'pro' ? 'bg-indigo-500/10 text-indigo-400' : 'bg-white/5 text-white/50'}`}>
              <Crown className="w-7 h-7" />
            </div>
            <div>
              <h4 className={`text-sm font-semibold uppercase tracking-wider ${textMutedClass}`}>Clearance Tier</h4>
              <p className={`text-xl font-bold capitalize ${subscription === 'pro' ? 'text-indigo-400' : textClass}`}>
                {subscription} License
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
