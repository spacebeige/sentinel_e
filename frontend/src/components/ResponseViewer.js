import React from 'react';
import { AlertCircle, CheckCircle2, Terminal } from 'lucide-react';

const ResponseViewer = ({ data, mode }) => {
  if (!data) return null;

  // STANDARD MODE UI
  if (mode === 'standard') {
      const resultText = data.result || "No output generated.";
      return (
        <div className="bg-white dark:bg-slate-700 p-6 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600">
          <div className="prose dark:prose-invert max-w-none whitespace-pre-wrap font-sans text-slate-800 dark:text-slate-100">
            {resultText}
          </div>
          <div className="mt-4 flex items-center text-xs text-green-600 dark:text-green-400 font-medium">
             <CheckCircle2 className="w-3 h-3 mr-1" />
             Safe Aggregated Response
          </div>
        </div>
      );
  }

  // EXPERIMENTAL MODE UI
  return (
    <div className="space-y-4">
       {/* Metrics Cards */}
       <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800 p-3 rounded border border-slate-700">
             <div className="text-xs text-slate-500 uppercase">Input Mode</div>
             <div className="text-sm font-mono text-emerald-400">Multi-Modal</div>
          </div>
           <div className="bg-slate-800 p-3 rounded border border-slate-700">
             <div className="text-xs text-slate-500 uppercase">HFI Score</div>
             <div className="text-xl font-bold text-amber-500">{data.metrics?.HFI?.toFixed(2) || 'N/A'}</div>
          </div>
           <div className="bg-slate-800 p-3 rounded border border-slate-700">
             <div className="text-xs text-slate-500 uppercase">Rounds</div>
             <div className="text-xl font-bold text-white">{data.rounds_executed || 0}</div>
          </div>
       </div>

       {/* Intersection Graph (JSON view for start) */}
       <div className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
          <div className="bg-slate-800 px-4 py-2 border-b border-slate-700 flex items-center justify-between">
             <h4 className="text-sm text-slate-300 font-semibold flex items-center">
                <Terminal className="w-4 h-4 mr-2" />
                Structural Analysis
             </h4>
             <span className="text-xs bg-red-900/50 text-red-400 px-2 py-0.5 rounded border border-red-800">Experimental</span>
          </div>
          <div className="p-4 overflow-x-auto">
             <pre className="text-xs font-mono text-emerald-300/90 whitespace-pre-wrap break-all">
                {JSON.stringify(data.intersection, null, 2)}
             </pre>
          </div>
       </div>

       {/* Full Raw Data */}
        <details className="group">
            <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-300 list-none flex items-center">
                 <span className="mr-2">▶</span> View Full Forensic Log
            </summary>
            <div className="mt-2 text-xs font-mono text-slate-500 bg-black p-4 rounded overflow-auto max-h-96">
                <pre>{JSON.stringify(data, null, 2)}</pre>
            </div>
        </details>
    </div>
  );
};

export default ResponseViewer;
