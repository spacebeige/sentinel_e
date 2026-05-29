import React, { useState, useEffect } from 'react';
import { useSupabaseAuth } from '../hooks/useSupabaseAuth';
import { Settings, Moon, Sun, Monitor, Bell, Shield, Download, Trash2, Cpu, Save } from 'lucide-react';
import { useTheme } from 'next-themes';

export default function SettingsPage() {
  const { user } = useSupabaseAuth();
  const { theme, setTheme } = useTheme();
  const isDark = theme === 'dark' || theme === 'system';
  
  const [preferences, setPreferences] = useState({
    defaultMode: 'standard',
    defaultModel: 'sentinel-sigma',
    autoSaveChats: true,
    conversationHistory: true,
  });

  const [privacy, setPrivacy] = useState({
    analyticsOptIn: true,
    storeConversations: true,
  });

  useEffect(() => {
    const savedPrefs = localStorage.getItem('sentinel_preferences');
    if (savedPrefs) setPreferences(JSON.parse(savedPrefs));
    
    const savedPrivacy = localStorage.getItem('sentinel_privacy');
    if (savedPrivacy) setPrivacy(JSON.parse(savedPrivacy));
  }, []);

  const savePreferences = (newPrefs: any) => {
    setPreferences(newPrefs);
    localStorage.setItem('sentinel_preferences', JSON.stringify(newPrefs));
  };

  const savePrivacy = (newPrivacy: any) => {
    setPrivacy(newPrivacy);
    localStorage.setItem('sentinel_privacy', JSON.stringify(newPrivacy));
  };

  if (!user) return null;

  const surfaceClass = isDark ? 'bg-white/5 border-white/10' : 'bg-black/5 border-black/5';
  const textClass = isDark ? 'text-white' : 'text-black';
  const textMutedClass = isDark ? 'text-white/60' : 'text-black/60';

  const Toggle = ({ checked, onChange }: { checked: boolean, onChange: (c: boolean) => void }) => (
    <button 
      onClick={() => onChange(!checked)}
      className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${checked ? 'bg-indigo-500' : 'bg-zinc-600'}`}
    >
      <div className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform duration-300 ${checked ? 'transform translate-x-6' : ''}`} />
    </button>
  );

  return (
    <div className={`min-h-screen ${isDark ? 'bg-[#09090b]' : 'bg-[#f5f5f7]'} p-8 md:p-12 overflow-y-auto`}>
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className={`text-3xl font-bold flex items-center gap-3 ${textClass}`}>
          <Settings className="w-8 h-8 text-indigo-500" />
          Settings Configuration
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column: Navigation/Sections */}
          <div className="space-y-6 lg:col-span-1">
            <div className={`p-6 rounded-3xl border ${surfaceClass} space-y-4`}>
              <div className={`flex items-center gap-3 font-semibold ${textClass} pb-2 border-b border-white/10`}>
                <Monitor className="w-5 h-5 text-indigo-400" /> Appearance
              </div>
              <div className={`flex items-center gap-3 font-semibold ${textClass} pb-2 border-b border-white/10`}>
                <Cpu className="w-5 h-5 text-indigo-400" /> Preferences
              </div>
              <div className={`flex items-center gap-3 font-semibold ${textClass} pb-2 border-b border-white/10`}>
                <Shield className="w-5 h-5 text-indigo-400" /> Privacy & Data
              </div>
              <div className={`flex items-center gap-3 font-semibold ${textClass}`}>
                <User className="w-5 h-5 text-indigo-400" /> Account
              </div>
            </div>
          </div>

          {/* Right Column: Settings Content */}
          <div className="space-y-8 lg:col-span-2">
            
            {/* Appearance */}
            <section className={`p-8 rounded-3xl border ${surfaceClass}`}>
              <h2 className={`text-xl font-bold mb-6 ${textClass}`}>Appearance</h2>
              <div className="grid grid-cols-3 gap-4">
                <button 
                  onClick={() => setTheme('light')}
                  className={`flex flex-col items-center p-4 rounded-2xl border transition-all ${theme === 'light' ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:bg-white/5'}`}
                >
                  <Sun className={`w-8 h-8 mb-2 ${theme === 'light' ? 'text-indigo-400' : textMutedClass}`} />
                  <span className={`font-medium ${theme === 'light' ? textClass : textMutedClass}`}>Light</span>
                </button>
                <button 
                  onClick={() => setTheme('dark')}
                  className={`flex flex-col items-center p-4 rounded-2xl border transition-all ${theme === 'dark' ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:bg-white/5'}`}
                >
                  <Moon className={`w-8 h-8 mb-2 ${theme === 'dark' ? 'text-indigo-400' : textMutedClass}`} />
                  <span className={`font-medium ${theme === 'dark' ? textClass : textMutedClass}`}>Dark</span>
                </button>
                <button 
                  onClick={() => setTheme('system')}
                  className={`flex flex-col items-center p-4 rounded-2xl border transition-all ${theme === 'system' ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:bg-white/5'}`}
                >
                  <Monitor className={`w-8 h-8 mb-2 ${theme === 'system' ? 'text-indigo-400' : textMutedClass}`} />
                  <span className={`font-medium ${theme === 'system' ? textClass : textMutedClass}`}>System</span>
                </button>
              </div>
            </section>

            {/* Preferences */}
            <section className={`p-8 rounded-3xl border ${surfaceClass}`}>
              <h2 className={`text-xl font-bold mb-6 ${textClass}`}>Preferences</h2>
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className={`font-semibold ${textClass}`}>Default Mode</h3>
                    <p className={`text-sm ${textMutedClass}`}>Select your starting orchestration mode</p>
                  </div>
                  <select 
                    value={preferences.defaultMode}
                    onChange={(e) => savePreferences({ ...preferences, defaultMode: e.target.value })}
                    className={`bg-black/20 border border-white/10 rounded-lg px-3 py-2 outline-none ${textClass}`}
                  >
                    <option value="standard">Standard</option>
                    <option value="pro">Pro Orchestration</option>
                  </select>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className={`font-semibold ${textClass}`}>Default Model</h3>
                    <p className={`text-sm ${textMutedClass}`}>Your primary cognitive engine</p>
                  </div>
                  <select 
                    value={preferences.defaultModel}
                    onChange={(e) => savePreferences({ ...preferences, defaultModel: e.target.value })}
                    className={`bg-black/20 border border-white/10 rounded-lg px-3 py-2 outline-none ${textClass}`}
                  >
                    <option value="sentinel-sigma">Sentinel Σ</option>
                    <option value="gpt4">GPT-4</option>
                    <option value="gemini">Gemini</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className={`font-semibold ${textClass}`}>Auto-Save Chats</h3>
                    <p className={`text-sm ${textMutedClass}`}>Automatically persist conversations</p>
                  </div>
                  <Toggle 
                    checked={preferences.autoSaveChats} 
                    onChange={(c) => savePreferences({ ...preferences, autoSaveChats: c })} 
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className={`font-semibold ${textClass}`}>Conversation History</h3>
                    <p className={`text-sm ${textMutedClass}`}>Display past chats in the sidebar</p>
                  </div>
                  <Toggle 
                    checked={preferences.conversationHistory} 
                    onChange={(c) => savePreferences({ ...preferences, conversationHistory: c })} 
                  />
                </div>
              </div>
            </section>

            {/* Privacy & Account */}
            <section className={`p-8 rounded-3xl border ${surfaceClass}`}>
              <h2 className={`text-xl font-bold mb-6 ${textClass}`}>Privacy & Account</h2>
              <div className="space-y-6">
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className={`font-semibold ${textClass}`}>Analytics Opt-In</h3>
                    <p className={`text-sm ${textMutedClass}`}>Help improve Sentinel-E with anonymous usage data</p>
                  </div>
                  <Toggle 
                    checked={privacy.analyticsOptIn} 
                    onChange={(c) => savePrivacy({ ...privacy, analyticsOptIn: c })} 
                  />
                </div>

                <div className="flex items-center justify-between pb-6 border-b border-white/10">
                  <div>
                    <h3 className={`font-semibold flex items-center gap-2 ${textClass}`}><Download className="w-4 h-4" /> Export Data</h3>
                    <p className={`text-sm ${textMutedClass}`}>Download all your conversations as JSON</p>
                  </div>
                  <button className="px-4 py-2 bg-white/10 hover:bg-white/20 transition-colors rounded-lg font-medium text-white text-sm">
                    Export
                  </button>
                </div>

                <div className="flex items-center justify-between pt-2">
                  <div>
                    <h3 className="font-semibold text-red-500">Danger Zone</h3>
                    <p className={`text-sm ${textMutedClass}`}>Permanently delete your account and all data</p>
                  </div>
                  <button className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 transition-colors rounded-lg font-medium text-sm flex items-center gap-2">
                    <Trash2 className="w-4 h-4" /> Delete Account
                  </button>
                </div>

              </div>
            </section>

          </div>
        </div>
      </div>
    </div>
  );
}
