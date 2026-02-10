import React, { useState, useRef } from 'react';
import { Send, Mic, Paperclip, X } from 'lucide-react';

const InputArea = ({ onSend, loading, mode }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [isListening, setIsListening] = useState(false); // Placeholder for voice
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
    // Basic placeholder for voice
    if (!isListening) {
        setIsListening(true);
        // Start recording logic here if implemented
        alert("Voice recording started (Simulation) - Speak now...");
        setTimeout(() => {
             setIsListening(false);
             setText(text + " [Transcribed Voice Input] ");
        }, 2000);
    } else {
        setIsListening(false);
    }
  };

  return (
    <div className="p-4 border-t border-slate-700 bg-slate-800">
      {file && (
        <div className="flex items-center space-x-2 text-xs text-slate-300 mb-2 bg-slate-700 p-2 rounded max-w-max">
           <Paperclip className="w-3 h-3" />
           <span>{file.name}</span>
           <button onClick={() => setFile(null)}><X className="w-3 h-3 hover:text-red-400" /></button>
        </div>
      )}

      <div className="flex items-end space-x-2">
        <button 
          onClick={toggleVoice}
          className={`p-3 rounded-full transition-colors ${isListening ? 'bg-red-500 animate-pulse text-white' : 'bg-slate-700 text-slate-400 hover:text-white'}`}
          title="Voice Input"
        >
          <Mic className="w-5 h-5" />
        </button>

        <button 
          onClick={() => fileInputRef.current?.click()}
          className="p-3 rounded-full bg-slate-700 text-slate-400 hover:text-white transition-colors"
          title="Attach File"
        >
          <Paperclip className="w-5 h-5" />
        </button>
        <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
        />

        <div className="flex-1 bg-slate-700 rounded-2xl p-2 focus-within:ring-2 ring-emerald-500/50">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={`Message Sentinel (${mode} mode)...`}
            className="w-full bg-transparent text-slate-100 placeholder-slate-400 focus:outline-none resize-none px-2 py-1 max-h-32 min-h-[44px]"
            rows="1"
          />
        </div>

        <button 
          onClick={handleSend}
          disabled={loading || (!text && !file)}
          className={`p-3 rounded-full ${loading ? 'bg-slate-600' : 'bg-emerald-600 hover:bg-emerald-500'} text-white transition-colors shadow-lg shadow-emerald-900/20`}
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default InputArea;
