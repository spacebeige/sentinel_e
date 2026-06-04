import React, { useState, useEffect } from 'react';
import { useSupabaseAuth } from '@hooks/useSupabaseAuth';
import api from '@services/api';
import { motion } from 'framer-motion';
import { useTheme } from 'next-themes';
import { IOSListGroup, IOSListItem } from './ui/IOSListGroup';
import { IOSToggle } from './ui/IOSToggle';
import { IOSContextMenu } from './ui/IOSContextMenu';
import { Settings, User, Monitor, Key, Shield, Download, Trash2, Cpu, Activity, MessageSquare } from 'lucide-react';
import { MODELS as AVAILABLE_MODELS } from "../config/runtime";

export default function SettingsPage() {
  const { user } = useSupabaseAuth();
  const { theme, setTheme } = useTheme();
  const isDark = theme === 'dark' || theme === 'system';
  
  const [preferences, setPreferences] = useState({
    defaultMode: 'standard',
    defaultModel: 'llama-3-3-70b',
  });

  

  const [advanced, setAdvanced] = useState({
    responseStyle: 'balanced',
    debateDepth: '6',
  });

  // Privacy Actions Modals
  const handleExportData = () => {
    alert("Export Data is not yet supported by the backend.");
  };

  const handleDeleteAccount = () => {
    alert("Account deletion is not yet supported by the backend.");
  };

  useEffect(() => {
    document.title = "Settings • Sentinel-E";
    if (!user) return;
    const fetchSettings = async () => {
      try {
        const res = await api.get('/api/user/settings');
        if (res.data && res.data.success) {
          const settings = res.data.data.settings;
          setPreferences({
            defaultMode: settings.default_mode || 'standard',
            defaultModel: settings.default_model || 'llama-3-3-70b',
          });
          setAdvanced({
            responseStyle: settings.response_style || 'balanced',
            debateDepth: String(settings.debate_rounds || '3'),
          });
          if (settings.theme) {
            setTheme(settings.theme);
          }
        }
      } catch (err) {
        console.error('Failed to load settings', err);
      }
    };
    fetchSettings();
  }, [user, setTheme]);

  const savePreferences = async (newPrefs: any) => {
    setPreferences(newPrefs);
    try {
      await api.put('/api/user/settings', {
        default_mode: newPrefs.defaultMode,
        default_model: newPrefs.defaultModel,
        response_style: newPrefs.responseStyle ?? advanced.responseStyle,
        debate_rounds: Number(newPrefs.debateDepth ?? advanced.debateDepth)
      });
    } catch (err) {
      console.error('Failed to save preferences', err);
    }
  };

  

  const handleThemeChange = async (newTheme: string) => {
    setTheme(newTheme);
    try {
      await api.put('/api/user/settings', { theme: newTheme });
    } catch (err) {
      console.error('Failed to save theme', err);
    }
  };

  if (!user) return null;

  return (
    <div className={`min-h-screen ${isDark ? 'bg-[#000000]' : 'bg-[#F2F2F7]'} font-sans pb-12`}>
      <div className="max-w-[700px] mx-auto px-4 pt-12 md:pt-16 relative">
        
        {/* Clickable Top Logo */}
        <a href="/" className="absolute top-4 left-4 flex items-center gap-2 cursor-pointer transition-opacity hover:opacity-70">
          <img src="/logo.png" alt="Sentinel-E" className="h-6 w-auto" />
          <span className="font-semibold text-black dark:text-white">Sentinel-E</span>
        </a>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8"
        >
          <h1 className="text-[32px] font-semibold tracking-tight text-black dark:text-white leading-tight">
            Settings
          </h1>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Account Group */}
          <IOSListGroup title="Account">
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#8E8E93] flex items-center justify-center"><User className="w-4 h-4 text-white" /></div>}
              title="Profile"
              onClick={() => {}}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#007AFF] flex items-center justify-center"><Key className="w-4 h-4 text-white" /></div>}
              title="Email"
              rightContent={<span className="text-[15px]">{user.email}</span>}
              onClick={() => {}}
            />
          </IOSListGroup>

          {/* AI Preferences Group */}
          <IOSListGroup title="AI Preferences">
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#AF52DE] flex items-center justify-center"><Cpu className="w-4 h-4 text-white" /></div>}
              title="Default Engine"
              rightContent={
                <IOSContextMenu 
                  value={preferences.defaultModel}
                  onChange={(val) => savePreferences({ ...preferences, defaultModel: val })}
                  options={AVAILABLE_MODELS.map(m => ({ label: m.name, value: m.id }))}
                />
              }
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#5856D6] flex items-center justify-center"><MessageSquare className="w-4 h-4 text-white" /></div>}
              title="Orchestration Mode"
              rightContent={
                <IOSContextMenu 
                  value={preferences.defaultMode}
                  onChange={(val) => savePreferences({ ...preferences, defaultMode: val })}
                  options={[
                    { label: 'Standard', value: 'standard' },
                    { label: 'Pro Orchestration', value: 'pro' }
                  ]}
                />
              }
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#FF2D55] flex items-center justify-center"><MessageSquare className="w-4 h-4 text-white" /></div>}
              title="Response Style"
              rightContent={
                <IOSContextMenu 
                  value={advanced.responseStyle}
                  onChange={(val) => { setAdvanced({ ...advanced, responseStyle: val }); savePreferences({ ...preferences, responseStyle: val }); }}
                  options={[
                    { label: 'Analytical', value: 'analytical' },
                    { label: 'Balanced', value: 'balanced' },
                    { label: 'Executive', value: 'executive' },
                    { label: 'Technical', value: 'technical' }
                  ]}
                />
              }
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#8E8E93] flex items-center justify-center"><Cpu className="w-4 h-4 text-white" /></div>}
              title="Debate Depth"
              rightContent={
                <IOSContextMenu 
                  value={advanced.debateDepth}
                  onChange={(val) => { setAdvanced({ ...advanced, debateDepth: val }); savePreferences({ ...preferences, debateDepth: val }); }}
                  options={[
                    { label: '4 Rounds', value: '4' },
                    { label: '6 Rounds', value: '6' },
                    { label: '8 Rounds', value: '8' }
                  ]}
                />
              }
            />
          </IOSListGroup>

          {/* Appearance Group */}
          <IOSListGroup title="Appearance">
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#FF9500] flex items-center justify-center"><Monitor className="w-4 h-4 text-white" /></div>}
              title="Theme"
              rightContent={
                <IOSContextMenu 
                  value={theme || 'system'}
                  onChange={(val) => handleThemeChange(val)}
                  options={[
                    { label: 'System', value: 'system' },
                    { label: 'Dark', value: 'dark' },
                    { label: 'Light', value: 'light' }
                  ]}
                />
              }
            />
          </IOSListGroup>

          

          {/* Data Controls */}
          <IOSListGroup title="Data Controls">
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#007AFF] flex items-center justify-center"><Download className="w-4 h-4 text-white" /></div>}
              title="Export Data"
              onClick={handleExportData}
            />
            <IOSListItem 
              icon={<div className="w-[28px] h-[28px] rounded-lg bg-[#FF3B30] flex items-center justify-center"><Trash2 className="w-4 h-4 text-white" /></div>}
              title="Delete Account"
              destructive
              onClick={handleDeleteAccount}
            />
          </IOSListGroup>

        </motion.div>
      </div>
    </div>
  );
}
