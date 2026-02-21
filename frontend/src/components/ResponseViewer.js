import React from 'react';
import { CheckCircle2, Terminal, Copy, Check } from 'lucide-react';
import { FeedbackButton } from './FeedbackButton';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ResponseViewer = ({ data, mode, onFeedback }) => {
  const [copied, setCopied] = React.useState(false);

  if (!data) return null;

  // STANDARD MODE UI
  if (mode === 'standard') {
      const resultText = data.data?.priority_answer || data.priority_answer || data.result || "No output generated.";
      const runId = data.chat_id || data.run_id || data.id;

      const handleCopy = () => {
        navigator.clipboard.writeText(resultText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      };

      return (
        <div className="bg-white dark:bg-white/5 p-6 rounded-xl border border-slate-200 dark:border-white/10 shadow-lg backdrop-blur-sm">
          {/* Clean Markdown Content */}
          <div className="prose prose-slate dark:prose-invert max-w-none text-slate-900 dark:text-slate-200 leading-relaxed">
            <style>{`
              .dark .prose-invert h1 { font-size: 1.875rem; margin-bottom: 1rem; }
              .dark .prose-invert h2 { font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
              .dark .prose-invert p { margin-bottom: 1rem; }
              .dark .prose-invert ul, .dark .prose-invert ol { margin-left: 1.5rem; margin-bottom: 1rem; }
              .dark .prose-invert li { margin-bottom: 0.5rem; }
              .dark .prose-invert code { background: rgba(100, 200, 100, 0.1); padding: 0.2em 0.4em; border-radius: 0.25rem; color: #10b981; }
              .dark .prose-invert pre { background: rgba(0, 0, 0, 0.5); padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
              .dark .prose-invert a { color: #3b82f6; text-decoration: underline; }
              .dark .prose-invert table { border-collapse: collapse; width: 100%; }
              .dark .prose-invert th, .dark .prose-invert td { border: 1px solid rgba(255, 255, 255, 0.1); padding: 0.75rem; text-align: left; }
            `}</style>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {resultText}
            </ReactMarkdown>
          </div>

          {/* ChatGPT-Style Footer */}
          <div className="mt-6 pt-4 border-t border-slate-200 dark:border-white/10 flex items-center justify-between gap-4">
            <div className="flex items-center text-xs text-emerald-600 dark:text-emerald-400 font-medium bg-emerald-100 dark:bg-emerald-500/10 rounded-full px-3 py-1 w-fit">
                <CheckCircle2 className="w-3 h-3 mr-2" />
                Safe Response
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleCopy}
                className="p-2 hover:bg-slate-100 dark:hover:bg-white/10 rounded-lg transition-colors text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                title="Copy response"
              >
                {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
              </button>
              {runId && <FeedbackButton runId={runId} onFeedbackSent={onFeedback} />}
            </div>
          </div>
        </div>
      );
  }

  // EXPERIMENTAL MODE UI
  const runId = data.chat_id || data.run_id || (data.metadata ? data.metadata.run_id : null);
  
  // Support both new 'SentinelResponse' structure and legacy formats
  const strictData = data.data || {};
  const strictMetadata = data.metadata || {};
  
  const humanText = strictData.priority_answer || data.priority_answer || data.human_layer || data.result;
  
  // Map machine metrics from new Metadata or fallback to legacy
  const machineData = {
    ...strictMetadata,
    ...strictData, // merge data for easy access to model_positions etc
    ...(data.machine_layer || {})
  };

  // Extract Metrics (Shim for V4/Legacy compatibility)
  const signals = machineData.signals || machineData.machine_metadata || {};
  const decision = machineData.decision || {};
  const analysis = machineData.analysis || {
     models_used: strictMetadata.models_used || [],
     rounds: strictMetadata.rounds_executed || 0
  };
  const risk = machineData.risk_layer || {};

  return (
    <div className="space-y-8">
       
       {/* 1. Human Layer (Chat) */}
       {humanText && (
        <div className="bg-white/5 p-6 rounded-xl border border-white/10 shadow-lg backdrop-blur-sm relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-2 opacity-50 text-[10px] uppercase font-bold tracking-widest text-slate-500 pointer-events-none group-hover:opacity-100 transition-opacity">
                Human Layer
            </div>
            <div className="prose prose-invert max-w-none text-slate-200 leading-relaxed text-sm">
                <style>{`
                  .prose-invert h1 { font-size: 1.5rem; margin-bottom: 0.75rem; color: #fff; }
                  .prose-invert h2 { font-size: 1.25rem; margin-top: 1.25rem; margin-bottom: 0.5rem; color: #e2e8f0; }
                  .prose-invert p { margin-bottom: 0.75rem; }
                  .prose-invert ul { margin-left: 1.25rem; margin-bottom: 0.75rem; list-style-type: disc; }
                  .prose-invert li { margin-bottom: 0.25rem; }
                  .prose-invert strong { color: #facc15; font-weight: 600; }
                `}</style>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {humanText}
                </ReactMarkdown>
            </div>
        </div>
       )}

       {/* 2. Machine Layer (Metrics Dashboard) */}
       {Object.keys(machineData).length > 0 && (
       <div className="space-y-4">
           {/* Section Header */}
           <div className="flex items-center space-x-2 border-b border-white/5 pb-2">
               <Terminal className="w-4 h-4 text-emerald-500" />
               <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400">Machine Layer Interception</h3>
           </div>

           {/* Cards Grid */}
           <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
               {/* Verdict */}
               <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                   <div className="text-[9px] text-slate-500 uppercase font-bold">Verdict</div>
                   <div className={`text-lg font-bold mt-1 ${
                       decision.verdict === 'CRITICAL' ? 'text-red-500' : 
                       decision.verdict === 'FRAGILE' ? 'text-amber-500' : 
                       decision.verdict === 'JUSTIFIED' ? 'text-emerald-500' : 'text-slate-300'
                   }`}>
                       {decision.verdict || 'PENDING'}
                   </div>
               </div>

               {/* Confidence */}
               <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                   <div className="text-[9px] text-slate-500 uppercase font-bold">Confidence</div>
                   <div className="text-lg font-bold text-white mt-1">
                       {decision.confidence || '-'}
                   </div>
               </div>

               {/* Variance */}
               <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                   <div className="text-[9px] text-slate-500 uppercase font-bold">Variance</div>
                   <div className="text-lg font-bold text-blue-400 mt-1">
                       {signals.variance_score != null ? signals.variance_score.toFixed(2) : '-'}
                   </div>
               </div>

               {/* Risk Level */}
               <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                   <div className="text-[9px] text-slate-500 uppercase font-bold">Boundary Risk</div>
                   <div className={`text-lg font-bold mt-1 ${
                       risk.boundary_severity === 'CRITICAL' || risk.boundary_severity === 'HIGH' ? 'text-red-500 animate-pulse' : 'text-emerald-400'
                   }`}>
                       {risk.boundary_severity || 'LOW'}
                   </div>
               </div>
           </div>

           {/* Analysis Details - Key Assumptions & Findings */}
           <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               {/* Key Assumptions */}
               <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
                   <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
                       <span className="w-1.5 h-1.5 bg-amber-500 rounded-full mr-2"></span>
                       Key Assumptions Map
                   </h4>
                   <ul className="space-y-2">
                       {analysis.key_assumptions?.length > 0 ? (
                           analysis.key_assumptions.slice(0, 5).map((item, idx) => (
                               <li key={idx} className="text-xs text-slate-300 flex items-start">
                                   <span className="text-slate-600 mr-2 font-mono">{idx + 1}.</span>
                                   {item}
                                </li>
                           ))
                       ) : (
                           <li className="text-xs text-slate-600 italic">No assumptions extracted.</li>
                       )}
                   </ul>
               </div>

               {/* Divergence Points */}
               <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
                   <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
                       <span className="w-1.5 h-1.5 bg-purple-500 rounded-full mr-2"></span>
                       Structural Divergence
                   </h4>
                   <ul className="space-y-2">
                        {analysis.divergence_points?.length > 0 ? (
                           analysis.divergence_points.slice(0, 5).map((item, idx) => (
                               <li key={idx} className="text-xs text-slate-300 flex items-start">
                                   <span className="text-slate-600 mr-2 font-mono">⚠</span>
                                   {item}
                                </li>
                           ))
                       ) : (
                           <li className="text-xs text-slate-600 italic">No structural divergence detected.</li>
                       )}
                   </ul>
               </div>
           </div>

           {/* JSON Raw Data Toggle */}
           <details className="group">
               <summary className="cursor-pointer text-[10px] text-slate-600 hover:text-slate-400 uppercase tracking-widest font-bold py-2 flex items-center select-none transition-colors">
                   <span className="mr-2 transform group-open:rotate-90 transition-transform">▶</span> View Raw Machine Protocol
               </summary>
               <div className="mt-2 bg-black rounded-lg border border-white/10 p-4 max-h-60 overflow-auto shadow-inner">
                   <pre className="text-[10px] font-mono text-emerald-500/80 whitespace-pre-wrap break-all">
                       {JSON.stringify(machineData, null, 2)}
                   </pre>
               </div>
           </details>
       </div>
       )}

       {/* Footer Actions */}
       {runId && (
           <div className="flex justify-end pt-4 border-t border-white/5">
                <FeedbackButton runId={runId} />
           </div>
       )}

    </div>
  );
};

export default ResponseViewer;
