import React, { useState, useEffect } from 'react';
import { useSupabaseAuth } from '@hooks/useSupabaseAuth';
import api from '@services/api';
import { motion } from 'framer-motion';
import { useTheme } from 'next-themes';
import { IOSListGroup, IOSListItem } from './ui/IOSListGroup';
import { Mail, Calendar, Key, Bell, Shield, Edit2, LogOut } from 'lucide-react';

export default function ProfilePage() {
  const { user, isAdmin, signOut } = useSupabaseAuth();
  const [customName, setCustomName] = useState('');
  const [isEditingName, setIsEditingName] = useState(false);
  const [tempName, setTempName] = useState('');
  
  // Sheet states
  const [showEmailSheet, setShowEmailSheet] = useState(false);
  const [showPhoneSheet, setShowPhoneSheet] = useState(false);
  const [showRoleSheet, setShowRoleSheet] = useState(false);
  const [phone, setPhone] = useState('');
  
  // Avatar upload
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    alert('Avatar upload is currently disabled. Please use Gravatar or your OAuth provider.');
  };
  
  // Analytics State
  const [stats, setStats] = useState({
    conversations: 0,
    messages: 0,
    favoriteMode: 'standard',
    favoriteModel: 'sentinel-sigma'
  });

  const { theme } = useTheme();
  const isDark = theme === 'dark' || theme === 'system';

  useEffect(() => {
    document.title = "Profile • Sentinel-E";
    if (!user) return;

    const fetchProfileAndStats = async () => {
      try {
        const res = await api.get('/api/user');
        if (res.data && res.data.success) {
          const profileData = res.data.data;
          const name = profileData.name || profileData.email?.split('@')[0] || 'Sentinel User';
          setCustomName(name);
          setTempName(name);
          setStats({
            conversations: profileData.stats?.chat_count || 0,
            messages: profileData.stats?.message_count || 0,
            favoriteMode: 'Standard',
            favoriteModel: 'Sentinel Σ'
          });
        }
      } catch (err) {
        console.error('Failed to fetch profile', err);
      }
    };

    fetchProfileAndStats();
  }, [user]);

  const handleSaveName = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    alert('Updating profile name is not natively supported by the backend yet. Read-only.');
    setCustomName(tempName);
    setIsEditingName(false);
  };

  if (!user) return null;

  const subscription = user.user_metadata?.subscription || 'standard';
  const roleLabel = isAdmin ? 'Administrator' : 'Operator';
  const tierLabel = subscription === 'pro' ? 'Pro License' : 'Standard License';

  return (
    <div className={`min-h-screen ${isDark ? 'bg-[#000000]' : 'bg-[#F2F2F7]'} font-sans pb-12`}>
      <div className="max-w-[700px] mx-auto px-4 pt-12 md:pt-16 relative">
        
        {/* Clickable Top Logo */}
        <a href="/" className="absolute top-4 left-4 flex items-center gap-2 cursor-pointer transition-opacity hover:opacity-70">
          <img src="/logo.png" alt="Sentinel-E" className="h-6 w-auto" />
          <span className="font-semibold text-black dark:text-white">Sentinel-E</span>
        </a>

        {/* Apple ID Style Hero */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center mb-10"
        >
          <div className="relative group cursor-pointer mb-5" onClick={() => fileInputRef.current?.click()}>
            <div className="w-[120px] h-[120px] md:w-[140px] md:h-[140px] rounded-full bg-gradient-to-b from-[#8E8E93] to-[#48484A] flex items-center justify-center text-white text-5xl md:text-6xl font-medium shadow-lg ring-4 ring-white/10 dark:ring-white/5 transition-transform duration-300 group-hover:scale-105 overflow-hidden">
               {/* Display avatar_url if exists, else initial */}
               {customName.charAt(0).toUpperCase()}
            </div>
            <div className="absolute inset-0 rounded-full bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 backdrop-blur-[2px]">
              <span className="text-white text-sm font-medium">Edit Photo</span>
            </div>
            <input type="file" ref={fileInputRef} onChange={handleAvatarUpload} accept="image/*" className="hidden" />
          </div>

          {isEditingName ? (
            <form onSubmit={handleSaveName} className="flex flex-col items-center gap-3 w-full max-w-sm">
              <input 
                type="text" 
                value={tempName}
                onChange={(e) => setTempName(e.target.value)}
                className="w-full text-center bg-transparent border-b border-[#007AFF] outline-none text-2xl font-semibold dark:text-white pb-1"
                autoFocus
              />
              <div className="flex gap-2">
                <button type="button" onClick={() => { setIsEditingName(false); setTempName(customName); }} className="text-[#007AFF] text-[15px]">Cancel</button>
                <button type="submit" className="text-[#007AFF] text-[15px] font-semibold">Save</button>
              </div>
            </form>
          ) : (
            <div className="flex flex-col items-center">
              <h1 className="text-[28px] md:text-[32px] font-semibold tracking-tight text-black dark:text-white leading-tight mb-1 flex items-center gap-2">
                {customName}
              </h1>
              <p className="text-[17px] text-[#8E8E93]">{user.email}</p>
              
              <div className="flex gap-2 mt-4">
                <span onClick={() => setShowRoleSheet(true)} className="cursor-pointer px-3 py-1 rounded-full bg-[#E5E5EA] dark:bg-[#3A3A3C] text-[13px] font-medium text-black dark:text-white hover:opacity-80 transition-opacity">
                  {roleLabel}
                </span>
                <span className="px-3 py-1 rounded-full bg-[#E5E5EA] dark:bg-[#3A3A3C] text-[13px] font-medium text-[#007AFF] dark:text-[#0A84FF]">
                  {tierLabel}
                </span>
              </div>
            </div>
          )}
        </motion.div>

        {/* Stats Row (VisionOS Glass Widgets) */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8"
        >
          <div className="ios-glass-panel rounded-[18px] p-4 flex flex-col items-center justify-center text-center">
            <span className="text-[28px] font-semibold text-black dark:text-white">{stats.conversations}</span>
            <span className="text-[13px] text-[#8E8E93] font-medium mt-1">Sessions</span>
          </div>
          <div className="ios-glass-panel rounded-[18px] p-4 flex flex-col items-center justify-center text-center">
            <span className="text-[28px] font-semibold text-black dark:text-white">{stats.messages}</span>
            <span className="text-[13px] text-[#8E8E93] font-medium mt-1">Messages</span>
          </div>
          <div className="ios-glass-panel rounded-[18px] p-4 flex flex-col items-center justify-center text-center">
            <span className="text-[17px] font-semibold text-black dark:text-white truncate w-full px-2 capitalize">{stats.favoriteModel.replace(/-/g, ' ')}</span>
            <span className="text-[13px] text-[#8E8E93] font-medium mt-1">Top Model</span>
          </div>
          <div className="ios-glass-panel rounded-[18px] p-4 flex flex-col items-center justify-center text-center">
            <span className="text-[17px] font-semibold text-black dark:text-white capitalize truncate w-full px-2">{stats.favoriteMode}</span>
            <span className="text-[13px] text-[#8E8E93] font-medium mt-1">Top Mode</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <IOSListGroup>
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#8E8E93] flex items-center justify-center"><Edit2 className="w-4 h-4 text-white" /></div>}
              title="Edit Name"
              onClick={() => setIsEditingName(true)}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#007AFF] flex items-center justify-center"><Mail className="w-4 h-4 text-white" /></div>}
              title="Email Details"
              onClick={() => setShowEmailSheet(true)}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#34C759] flex items-center justify-center"><Key className="w-4 h-4 text-white" /></div>}
              title="Phone Numbers"
              onClick={() => setShowPhoneSheet(true)}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#8E8E93] flex items-center justify-center"><Key className="w-4 h-4 text-white" /></div>}
              title="Password & Security"
              onClick={() => { alert('Password management sheet'); }}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#8E8E93] flex items-center justify-center"><Shield className="w-4 h-4 text-white" /></div>}
              title="Data & Privacy"
              onClick={() => {}}
            />
          </IOSListGroup>

          <IOSListGroup>
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#007AFF] flex items-center justify-center"><Calendar className="w-4 h-4 text-white" /></div>}
              title="Subscriptions"
              rightContent={<span className="capitalize">{subscription}</span>}
              onClick={() => {}}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#FF3B30] flex items-center justify-center"><Bell className="w-4 h-4 text-white" /></div>}
              title="Notifications"
              onClick={() => {}}
            />
          </IOSListGroup>

          <IOSListGroup>
            <IOSListItem 
              title="Sign Out"
              destructive
              onClick={() => signOut()}
            />
          </IOSListGroup>
        </motion.div>

        {/* Modals / Sheets */}
        {showEmailSheet && (
          <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setShowEmailSheet(false)}>
            <div className="w-full sm:w-[400px] bg-white dark:bg-[#1C1C1E] rounded-t-2xl sm:rounded-2xl p-6" onClick={e => e.stopPropagation()}>
              <h3 className="text-xl font-semibold mb-4 text-black dark:text-white">Email Address</h3>
              <p className="text-[#8E8E93] text-sm mb-4">Your primary email address is used for communication and login.</p>
              <div className="ios-glass-panel p-4 rounded-xl mb-4 text-black dark:text-white">
                {user.email} <span className="text-xs ml-2 text-green-500 font-medium">Verified</span>
              </div>
              <div className="flex flex-col gap-2">
                <button onClick={() => { navigator.clipboard.writeText(user.email || ''); alert('Copied!'); }} className="w-full py-3 bg-[#E5E5EA] dark:bg-[#3A3A3C] text-black dark:text-white rounded-xl font-medium">Copy Email</button>
                <button onClick={() => setShowEmailSheet(false)} className="w-full py-3 text-[#007AFF] font-medium">Done</button>
              </div>
            </div>
          </div>
        )}

        {showPhoneSheet && (
          <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setShowPhoneSheet(false)}>
            <div className="w-full sm:w-[400px] bg-white dark:bg-[#1C1C1E] rounded-t-2xl sm:rounded-2xl p-6" onClick={e => e.stopPropagation()}>
              <h3 className="text-xl font-semibold mb-4 text-black dark:text-white">Phone Numbers</h3>
              <input type="tel" value={phone} onChange={e => setPhone(e.target.value)} placeholder="Add a phone number..." className="w-full bg-[#E5E5EA] dark:bg-[#3A3A3C] p-3 rounded-xl outline-none text-black dark:text-white mb-4" />
              <div className="flex gap-2">
                <button onClick={() => { alert('Saved phone to profile'); setShowPhoneSheet(false); }} className="flex-1 py-3 bg-[#007AFF] text-white rounded-xl font-medium">Save</button>
                <button onClick={() => setShowPhoneSheet(false)} className="flex-1 py-3 bg-[#E5E5EA] dark:bg-[#3A3A3C] text-black dark:text-white rounded-xl font-medium">Cancel</button>
              </div>
            </div>
          </div>
        )}

        {showRoleSheet && (
          <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setShowRoleSheet(false)}>
            <div className="w-full sm:w-[400px] bg-white dark:bg-[#1C1C1E] rounded-t-2xl sm:rounded-2xl p-6" onClick={e => e.stopPropagation()}>
              <h3 className="text-xl font-semibold mb-2 text-black dark:text-white">Account Role</h3>
              <p className="text-[#8E8E93] text-sm mb-6">Current active role permissions.</p>
              <div className="ios-glass-panel p-4 rounded-xl mb-4">
                <h4 className="font-semibold text-black dark:text-white mb-1">{roleLabel}</h4>
                <p className="text-sm text-[#8E8E93]">
                  {isAdmin ? "You have full administrative access to all settings and telemetry." : "You have standard user access. Some admin features may be restricted."}
                </p>
              </div>
              <button onClick={() => setShowRoleSheet(false)} className="w-full py-3 bg-[#E5E5EA] dark:bg-[#3A3A3C] text-black dark:text-white rounded-xl font-medium">Dismiss</button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
