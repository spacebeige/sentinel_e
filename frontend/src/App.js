import React, { useState } from 'react';
import axios from 'axios';
import { Shield, FlaskConical, LayoutDashboard } from 'lucide-react';
import Sidebar from './components/Sidebar';
import InputArea from './components/InputArea';
import ResponseViewer from './components/ResponseViewer';

function App() {
  const [mode, setMode] = useState('standard'); // 'standard' | 'experimental'
  const [history, setHistory] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleModeSwitch = (newMode) => {
    setMode(newMode);
    setCurrentResult(null);
  };

  const handleSend = async ({ text, file }) => {
    setLoading(true);
    setCurrentResult(null);

    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('file', file);

    const endpoint = mode === 'standard' 
        ? 'http://localhost:8001/run/standard' 
        : 'http://localhost:8001/run/experimental';

    try {
      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const result = response.data;
      
      // Update State
      setCurrentResult(result);
      
      // Add to sidebar history
      const newEntry = {
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          mode: mode,
          summary: text ? text.substring(0, 30) + "..." : "[File Upload]",
          data: result
      };
      setHistory(prev => [newEntry, ...prev]);

    } catch (error) {
      console.error(error);
      alert("Error processing request: " + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = (run) => {
      setMode(run.mode);
      setCurrentResult(run.data);
  };

  return (
    <div className="flex h-screen bg-slate-950 font-sans text-slate-200">
      
      {/* Sidebar */}
      <Sidebar history={history} onSelectRun={handleSelectRun} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        
        {/* Header */}
        <header className="px-6 py-4 flex items-center justify-between border-b border-slate-800 bg-slate-900/50 backdrop-blur">
            <div className="flex items-center space-x-4">
                <LayoutDashboard className="w-5 h-5 text-slate-400" />
                <h1 className="text-lg font-semibold text-white">Sentinel Dashboard</h1>
            </div>

            {/* Mode Toggle */}
            <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                    onClick={() => handleModeSwitch('standard')}
                    className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${
                        mode === 'standard' 
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                >
                    <Shield className="w-4 h-4 mr-2" />
                    Standard
                </button>
                <button
                    onClick={() => handleModeSwitch('experimental')}
                    className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${
                        mode === 'experimental' 
                        ? 'bg-amber-700 text-white shadow-lg shadow-amber-900/20' 
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                >
                    <FlaskConical className="w-4 h-4 mr-2" />
                    Experimental
                </button>
            </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-slate-700">
            {!currentResult && !loading && (
                <div className="h-full flex flex-col items-center justify-center text-slate-600">
                    <div className={`p-4 rounded-full mb-4 ${mode === 'standard' ? 'bg-blue-900/20 text-blue-500' : 'bg-amber-900/20 text-amber-500'}`}>
                         {mode === 'standard' ? <Shield className="w-12 h-12" /> : <FlaskConical className="w-12 h-12" />}
                    </div>
                    <p className="text-lg">Ready for {mode} analysis</p>
                    <p className="text-sm opacity-60 mt-2">Enter text, upload a file, or use voice.</p>
                </div>
            )}

            {loading && (
                <div className="flex items-center justify-center h-full">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
                </div>
            )}

            {currentResult && (
                <div className="max-w-4xl mx-auto">
                    <ResponseViewer data={currentResult} mode={mode} />
                </div>
            )}
        </div>

        {/* Input Area */}
        <InputArea onSend={handleSend} loading={loading} mode={mode} />

      </div>
    </div>
  );
}

export default App;
