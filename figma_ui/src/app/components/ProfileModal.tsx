import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, User, Mail, Calendar, Activity, Edit2, Shield, Crown, Sparkles, MessageSquare, Clock } from 'lucide-react';
import { useSupabaseAuth } from '@hooks/useSupabaseAuth';
import { supabase } from '../lib/supabase';
import { getUserAnalytics, UserAnalytics } from '../services/analyticsService';

interface ProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  isDark: boolean;
}

export function ProfileModal({ isOpen, onClose, isDark }: ProfileModalProps) {
  const { user, isAdmin } = useSupabaseAuth();
  const [customName, setCustomName] = useState('');
  const [isEditingName, setIsEditingName] = useState(false);
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null);
  
  // Fake subscription for demonstration since we don't have a backend payments flow
  const [subscription, setSubscription] = useState<'standard' | 'pro'>('standard');

  useEffect(() => {
    if (user) {
      setCustomName(user.user_metadata?.custom_name || user.user_metadata?.full_name || 'Sentinel User');
      setSubscription(user.user_metadata?.subscription || 'standard');
      setAnalytics(getUserAnalytics(user.id));
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

  const toggleSubscription = async () => {
    if (!user) return;
    const newSub = subscription === 'standard' ? 'pro' : 'standard';
    setSubscription(newSub);
    try {
      await supabase.auth.updateUser({
        data: { subscription: newSub }
      });
    } catch (error) {
      console.error('Failed to update subscription', error);
    }
  };

  if (!isOpen || !user) return null;

  const bgClass = isDark ? 'bg-[#121216]' : 'bg-[#f5f5f7]';
  const surfaceClass = isDark ? 'bg-white/5 border-white/10' : 'bg-black/5 border-black/5';
  const textClass = isDark ? 'text-white' : 'text-black';
  const textMutedClass = isDark ? 'text-white/60' : 'text-black/60';

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
        <motion.div 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }} 
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        />
        
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className={`relative w-full max-w-lg overflow-hidden rounded-3xl border shadow-2xl ${bgClass} border-white/10`}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-white/10">
            <h2 className={`text-xl font-semibold flex items-center gap-2 ${textClass}`}>
              <User className="w-5 h-5" /> Profile & Entitlements
            </h2>
            <button onClick={onClose} className={`p-2 rounded-full hover:${surfaceClass} transition-colors ${textMutedClass}`}>
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
            
            {/* Identity Section */}
            <div className={`p-5 rounded-2xl border ${surfaceClass}`}>
              <div className="flex items-start gap-4">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold shrink-0">
                  {customName.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    {isEditingName ? (
                      <div className="flex items-center gap-2 w-full">
                        <input 
                          type="text" 
                          value={customName}
                          onChange={(e) => setCustomName(e.target.value)}
                          className={`flex-1 bg-transparent border-b border-indigo-500 outline-none ${textClass} font-semibold text-lg`}
                          autoFocus
                        />
                        <button onClick={handleSaveName} className="text-xs bg-indigo-500 text-white px-2 py-1 rounded-md">Save</button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <h3 className={`text-xl font-semibold truncate ${textClass}`}>{customName}</h3>
                        <button onClick={() => setIsEditingName(true)} className={`${textMutedClass} hover:text-indigo-500 transition-colors`}>
                          <Edit2 className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </div>
                  
                  <div className={`flex items-center gap-2 mt-1 text-sm ${textMutedClass}`}>
                    <Mail className="w-4 h-4" /> <span className="truncate">{user.email}</span>
                  </div>
                  
                  <div className={`flex items-center gap-2 mt-1 text-sm ${textMutedClass}`}>
                    <Calendar className="w-4 h-4" /> <span>Joined {new Date(user.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Roles & Entitlements */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-4 rounded-2xl border ${surfaceClass} flex flex-col items-center justify-center text-center`}>
                <Shield className={`w-8 h-8 mb-2 ${isAdmin ? 'text-emerald-500' : textMutedClass}`} />
                <span className={`text-sm font-medium ${textMutedClass}`}>System Role</span>
                <span className={`text-lg font-bold ${isAdmin ? 'text-emerald-500' : textClass}`}>
                  {isAdmin ? 'Administrator' : 'Standard User'}
                </span>
              </div>
              
              <div 
                className={`p-4 rounded-2xl border ${surfaceClass} flex flex-col items-center justify-center text-center cursor-pointer transition-transform hover:scale-[1.02] active:scale-[0.98] relative overflow-hidden`}
                onClick={toggleSubscription}
              >
                {subscription === 'pro' && (
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 pointer-events-none" />
                )}
                <Crown className={`w-8 h-8 mb-2 ${subscription === 'pro' ? 'text-indigo-400' : textMutedClass}`} />
                <span className={`text-sm font-medium ${textMutedClass}`}>Subscription</span>
                <span className={`text-lg font-bold ${subscription === 'pro' ? 'text-indigo-400' : textClass} capitalize`}>
                  {subscription} Tier
                </span>
                <span className={`text-[10px] mt-1 ${textMutedClass}`}>(Click to Toggle)</span>
              </div>
            </div>

            {/* Analytics */}
            <div>
              <h3 className={`text-sm font-bold uppercase tracking-wider mb-3 ${textMutedClass} flex items-center gap-2`}>
                <Activity className="w-4 h-4" /> Usage Analytics
              </h3>
              
              <div className="grid grid-cols-2 gap-3">
                <div className={`p-4 rounded-2xl border ${surfaceClass}`}>
                  <div className={`flex items-center gap-2 text-sm mb-1 ${textMutedClass}`}>
                    <MessageSquare className="w-4 h-4" /> Messages
                  </div>
                  <div className={`text-2xl font-bold ${textClass}`}>{analytics?.messages || 0}</div>
                </div>
                
                <div className={`p-4 rounded-2xl border ${surfaceClass}`}>
                  <div className={`flex items-center gap-2 text-sm mb-1 ${textMutedClass}`}>
                    <Clock className="w-4 h-4" /> Hours Used
                  </div>
                  <div className={`text-2xl font-bold ${textClass}`}>{analytics?.hoursUsed || 0}</div>
                </div>

                <div className={`p-4 rounded-2xl border ${surfaceClass} col-span-2 flex items-center justify-between`}>
                  <div>
                    <div className={`text-sm mb-1 ${textMutedClass}`}>Favorite Model</div>
                    <div className={`font-semibold ${textClass}`}>{analytics?.favoriteModel || 'N/A'}</div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm mb-1 ${textMutedClass}`}>Top Mode</div>
                    <div className={`font-semibold ${textClass} flex items-center gap-1 justify-end`}>
                      <Sparkles className="w-3 h-3 text-indigo-400" /> {analytics?.favoriteMode || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
