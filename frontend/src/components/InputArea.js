import React, { useState, useRef } from 'react';
import { Mic, Paperclip, X, CornerDownLeft } from 'lucide-react';

const InputArea = ({ onSend, loading, mode }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const fileInputRef = useRef(null);

  const handleSend = () => {
    if (!text && !file) return;
    onSend({ text, file });
    setText('');
    setFile(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const toggleVoice = () => {
    if (!isListening) {
        setIsListening(true);
        // Simulation
        setTimeout(() => {
             setIsListening(false);
             setText(prev => prev + " [Voice Input Probe] ");
        }, 1500);
    } else {
        setIsListening(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto mb-6">
      <div 
        className={`
            relative bg-white dark:bg-gpt-surface rounded-2xl border transition-all duration-300 ease-out
            ${isFocused 
                ? 'border-poly-accent/50 shadow-glow ring-1 ring-poly-accent/20' 
                : 'border-slate-200 dark:border-white/10 shadow-lg'
            }
        `}
      >
        {/* File Preview */}
        {file && (
            <div className="px-4 pt-3 flex items-center">
                <div className="flex items-center space-x-2 text-xs text-slate-700 dark:text-white bg-slate-100 dark:bg-white/10 px-3 py-1.5 rounded-full border border-slate-200 dark:border-white/10">
                    <Paperclip className="w-3 h-3 text-poly-accent" />
                    <span className="truncate max-w-[200px]">{file.name}</span>
                    <button onClick={() => setFile(null)} className="ml-2 hover:text-red-400 transition-colors">
                        <X className="w-3 h-3" />
                    </button>
                </div>
            </div>
        )}

        {/* Text Area */}
        <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
            }}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyDown={handleKeyPress}
            placeholder={`Message Sentinel-${mode === 'standard' ? 'E' : 'Σ'}...`}
            className="w-full bg-transparent text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none resize-none px-4 py-3.5 min-h-[56px] max-h-[200px] text-sm leading-relaxed scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/10"
            rows={1}
            style={{ height: '56px' }}
        />

        {/* Toolbar */}
        <div className="flex items-center justify-between px-2 pb-2">
            
            {/* Left Actions */}
            <div className="flex items-center space-x-1">
                <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-white/5 transition-colors"
                >
                    <Paperclip className="w-5 h-5" />
                </button>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
                
                <button 
                   onClick={toggleVoice}
                   className={`p-2 rounded-lg transition-colors ${isListening ? 'text-red-400 animate-pulse' : 'text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-white/5'}`}
                >
                    <Mic className="w-5 h-5" />
                </button>
            </div>

            {/* Right Logic (Send) */}
            <button 
                onClick={handleSend}
                disabled={loading || (!text && !file)}
                className={`
                    p-2 rounded-lg transition-all duration-200 flex items-center justify-center
                    ${(text || file) 
                        ? 'bg-emerald-600 text-white shadow-lg hover:bg-emerald-500' 
                        : 'bg-white/5 text-slate-600 cursor-not-allowed'
                    }
                `}
            >
                {loading ? (
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                    <CornerDownLeft className="w-5 h-5" />
                )}
            </button>
        </div>
      </div>
      
      {/* Disclaimer */}
      <div className="text-center mt-2">
         <p className="text-[10px] text-slate-500 dark:text-slate-600">Sentinel-E may produce inaccurate results. Verify critical information independently.</p>
      </div>
    </div>
  );
};

export default InputArea;
