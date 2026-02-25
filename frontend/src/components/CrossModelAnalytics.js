import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Shield, Activity, Brain, AlertTriangle, Eye, Skull,
  ChevronDown, ChevronUp, Loader2, RefreshCw, Users,
  ArrowRight, CheckCircle2, XCircle, Clock
} from 'lucide-react';

/* ─── Helpers ───────────────────────────────────────────────────── */
const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const fixed2 = v => v != null ? Number(v).toFixed(2) : '—';
import { API_BASE } from '../config';

const riskColor = (level) => {
  const l = (level || '').toUpperCase();
  if (l === 'CRITICAL') return 'var(--accent-red)';
  if (l === 'HIGH') return 'var(--accent-red)';
  if (l === 'MEDIUM') return 'var(--accent-yellow)';
  return 'var(--accent-green)';
};

const scoreColor = (v) => {
  if (v == null) return 'var(--text-tertiary)';
  if (v >= 0.7) return 'var(--accent-red)';
  if (v >= 0.4) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
};

const MODEL_COLORS = {
  groq: '#3b82f6',
  llama70b: '#6366f1',
  qwen: '#8b5cf6',
  qwenvl: '#06b6d4',
};

const MODEL_LABELS = {
  groq: 'Groq (LLaMA 3.1)',
  llama70b: 'Llama 3.3 70B',
  qwen: 'Qwen 2.5',
  qwenvl: 'QwenVL',
};

/* ─── Score Bar ─────────────────────────────────────────────────── */
const ScoreBar = ({ label, score, compact = false }) => {
  const v = Math.min((score || 0) * 100, 100);
  const color = scoreColor(score);
  return (
    <div className={compact ? 'space-y-0.5' : 'space-y-1'}>
      <div className="flex justify-between items-center">
        <span className="text-[9px] uppercase font-semibold tracking-wider" style={{ color: 'var(--text-tertiary)' }}>
          {label}
        </span>
        <span className="text-[9px] font-mono" style={{ color }}>{fixed2(score)}</span>
      </div>
      <div className="progress-bar" style={{ height: compact ? '3px' : '4px' }}>
        <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

/* ─── Step Badge ────────────────────────────────────────────────── */
const StepBadge = ({ step }) => {
  const isConsensus = step.type === 'consensus';
  const succeeded = step.status === 'success';

  return (
    <div className="p-2.5 rounded-lg transition-all" style={{
      backgroundColor: 'var(--bg-tertiary)',
      border: `1px solid ${succeeded ? 'var(--border-secondary)' : 'rgba(239,68,68,0.3)'}`,
    }}>
      <div className="flex items-center gap-2 mb-1.5">
        <div className="flex items-center gap-1">
          {succeeded
            ? <CheckCircle2 className="w-3 h-3" style={{ color: 'var(--accent-green)' }} />
            : <XCircle className="w-3 h-3" style={{ color: 'var(--accent-red)' }} />
          }
          <span className="text-[9px] font-bold" style={{ color: 'var(--text-secondary)' }}>
            Step {step.step}
          </span>
        </div>
        <span className={`text-[8px] px-1.5 py-0.5 rounded-full font-semibold uppercase ${
          isConsensus ? '' : ''
        }`} style={{
          backgroundColor: isConsensus ? 'rgba(139,92,246,0.15)' : 'rgba(59,130,246,0.15)',
          color: isConsensus ? 'var(--accent-purple)' : 'var(--accent-blue)',
        }}>
          {isConsensus ? 'CONSENSUS' : 'INDIVIDUAL'}
        </span>
      </div>

      <div className="flex items-center gap-1 mb-1.5 flex-wrap">
        {isConsensus ? (
          <>
            <span className="text-[8px] font-medium" style={{ color: MODEL_COLORS[step.analyzer_ids?.[0]] || 'var(--text-tertiary)' }}>
              {step.analyzers?.[0] || '?'}
            </span>
            <span className="text-[8px]" style={{ color: 'var(--text-tertiary)' }}>+</span>
            <span className="text-[8px] font-medium" style={{ color: MODEL_COLORS[step.analyzer_ids?.[1]] || 'var(--text-tertiary)' }}>
              {step.analyzers?.[1] || '?'}
            </span>
          </>
        ) : (
          <span className="text-[8px] font-medium" style={{ color: MODEL_COLORS[step.analyzer_id] || 'var(--text-tertiary)' }}>
            {step.analyzer || '?'}
          </span>
        )}
        <ArrowRight className="w-2.5 h-2.5" style={{ color: 'var(--text-tertiary)' }} />
        <span className="text-[8px] font-bold" style={{ color: MODEL_COLORS[step.subject_id] || 'var(--text-primary)' }}>
          {step.subject}
        </span>
      </div>

      {succeeded && step.scores && (
        <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 mt-1">
          <MiniScore label="Manip" value={step.scores.manipulation_level} />
          <MiniScore label="Risk" value={step.scores.risk_level} />
          <MiniScore label="Threat" value={step.scores.threat_level} />
        </div>
      )}
    </div>
  );
};

const MiniScore = ({ label, value }) => (
  <div className="flex items-center justify-between">
    <span className="text-[7px] uppercase" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
    <span className="text-[8px] font-mono font-bold" style={{ color: scoreColor(value) }}>
      {fixed2(value)}
    </span>
  </div>
);

/* ─── Model Profile Card ───────────────────────────────────────── */
const ModelProfileCard = ({ modelId, profile, expanded, onToggle }) => {
  if (!profile || profile.status === 'no_data') return null;

  const scores = profile.scores || {};
  const color = profile.color || MODEL_COLORS[modelId] || 'var(--accent-blue)';

  return (
    <div className="rounded-lg overflow-hidden" style={{
      backgroundColor: 'var(--bg-tertiary)',
      border: `1px solid var(--border-secondary)`,
      borderLeft: `3px solid ${color}`,
    }}>
      <button onClick={onToggle}
        className="w-full flex items-center justify-between p-3 text-left transition-colors"
        style={{ backgroundColor: 'transparent' }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
          <span className="text-[11px] font-bold truncate" style={{ color: 'var(--text-primary)' }}>
            {profile.name}
          </span>
          <span className="text-[8px] px-1.5 py-0.5 rounded-full font-bold uppercase flex-shrink-0"
            style={{ backgroundColor: `${riskColor(profile.overall_risk)}15`, color: riskColor(profile.overall_risk) }}>
            {profile.overall_risk || 'N/A'}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[8px] font-mono" style={{ color: 'var(--text-tertiary)' }}>
            {profile.step_count} steps
          </span>
          {expanded ? <ChevronUp className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} /> 
                    : <ChevronDown className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />}
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-2" style={{ borderTop: '1px solid var(--border-secondary)' }}>
          <div className="pt-2 space-y-1.5">
            <ScoreBar label="Manipulation" score={scores.manipulation_level} compact />
            <ScoreBar label="Risk Level" score={scores.risk_level} compact />
            <ScoreBar label="Self-Preservation" score={scores.self_preservation} compact />
            <ScoreBar label="Evasion" score={scores.evasion_index} compact />
            <ScoreBar label="Conf. Inflation" score={scores.confidence_inflation} compact />
            <ScoreBar label="Threat Level" score={scores.threat_level} compact />
          </div>
          {profile.key_signals && profile.key_signals.length > 0 && (
            <div className="pt-1">
              <span className="text-[8px] font-semibold uppercase" style={{ color: 'var(--text-tertiary)' }}>
                Key Signals
              </span>
              <div className="flex flex-wrap gap-1 mt-1">
                {profile.key_signals.slice(0, 5).map((sig, i) => (
                  <span key={i} className="text-[7px] px-1.5 py-0.5 rounded-full"
                    style={{ backgroundColor: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>
                    {sig}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/* ─── Main Component ────────────────────────────────────────────── */
const CrossModelAnalytics = ({ chatId, lastResponse, query, autoTrigger = false }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedModels, setExpandedModels] = useState({});
  const [showSteps, setShowSteps] = useState(false);
  const [hasTriggered, setHasTriggered] = useState(false);

  const runAnalysis = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_BASE}/api/cross-analysis`, {
        chat_id: chatId || null,
        query: query || '',
        llm_response: lastResponse || '',
      });
      setData(res.data);
    } catch (err) {
      console.error('Cross-analysis error:', err);
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [chatId, lastResponse, query, loading]);

  // Auto-trigger on first response in glass mode
  useEffect(() => {
    if (autoTrigger && lastResponse && !hasTriggered && !loading && !data) {
      setHasTriggered(true);
      runAnalysis();
    }
  }, [autoTrigger, lastResponse, hasTriggered, loading, data, runAnalysis]);

  const toggleModel = (modelId) => {
    setExpandedModels(prev => ({ ...prev, [modelId]: !prev[modelId] }));
  };

  const overall = data?.overall_risk || {};
  const profiles = data?.model_profiles || {};
  const steps = data?.steps || [];
  const analyzedModels = data?.analyzed_models || [];

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Eye className="w-3.5 h-3.5" style={{ color: 'var(--accent-purple)' }} />
          <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
            Cross-Model Analysis
          </span>
        </div>
        <button onClick={runAnalysis} disabled={loading}
          className="flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-semibold transition-all"
          style={{
            backgroundColor: loading ? 'var(--bg-tertiary)' : 'rgba(139,92,246,0.15)',
            color: loading ? 'var(--text-tertiary)' : 'var(--accent-purple)',
            border: '1px solid rgba(139,92,246,0.2)',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}>
          {loading
            ? <><Loader2 className="w-3 h-3 animate-spin" /> Running...</>
            : <><RefreshCw className="w-3 h-3" /> {data ? 'Re-run' : 'Analyze'}</>
          }
        </button>
      </div>

      {/* Loading State */}
      {loading && !data && (
        <div className="p-4 rounded-lg text-center space-y-2" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
          <Loader2 className="w-6 h-6 animate-spin mx-auto" style={{ color: 'var(--accent-purple)' }} />
          <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
            Running 8-step cross-model behavioral analysis...
          </p>
          <p className="text-[8px]" style={{ color: 'var(--text-tertiary)' }}>
            3 models × 8 analysis steps
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-3 rounded-lg flex items-center gap-2" style={{
          backgroundColor: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.2)',
        }}>
          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--accent-red)' }} />
          <span className="text-[10px]" style={{ color: 'var(--accent-red)' }}>{error}</span>
        </div>
      )}

      {/* Results */}
      {data && (
        <>
          {/* Overall Risk Banner */}
          <div className="p-3 rounded-lg" style={{
            backgroundColor: `${riskColor(overall.level)}08`,
            border: `1px solid ${riskColor(overall.level)}30`,
          }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4" style={{ color: riskColor(overall.level) }} />
                <span className="text-[11px] font-bold" style={{ color: 'var(--text-primary)' }}>
                  Overall Behavioral Risk
                </span>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded-full font-bold uppercase"
                style={{ backgroundColor: `${riskColor(overall.level)}20`, color: riskColor(overall.level) }}>
                {overall.level || 'N/A'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <MiniMetric label="Avg Threat" value={fixed2(overall.average_threat)} color={scoreColor(overall.average_threat)} />
              <MiniMetric label="Avg Manip" value={fixed2(overall.average_manipulation)} color={scoreColor(overall.average_manipulation)} />
              <MiniMetric label="Avg Risk" value={fixed2(overall.average_risk)} color={scoreColor(overall.average_risk)} />
              <MiniMetric label="Max Threat" value={fixed2(overall.max_threat)} color={scoreColor(overall.max_threat)} />
            </div>
          </div>

          {/* Pipeline Stats */}
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Clock className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />
              <span className="text-[9px] font-mono" style={{ color: 'var(--text-tertiary)' }}>
                {data.steps_completed}/{data.steps_total} steps · {data.elapsed_seconds}s
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Users className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />
              <span className="text-[9px] font-mono" style={{ color: 'var(--text-tertiary)' }}>
                {overall.models_analyzed || 0} models analyzed
              </span>
            </div>
          </div>

          {/* Models Being Analyzed */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 px-0.5">
              <Brain className="w-3 h-3" style={{ color: 'var(--accent-blue)' }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                Models Analyzed
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {analyzedModels.map(m => (
                <span key={m.id} className="text-[8px] px-2 py-1 rounded-full font-bold uppercase"
                  style={{ backgroundColor: `${m.color}18`, color: m.color, border: `1px solid ${m.color}30` }}>
                  {m.name}
                </span>
              ))}
            </div>
          </div>

          {/* Per-Model Profiles */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider px-0.5" style={{ color: 'var(--text-secondary)' }}>
              Behavioral Profiles
            </span>
            {Object.entries(profiles).map(([modelId, profile]) => (
              <ModelProfileCard
                key={modelId}
                modelId={modelId}
                profile={profile}
                expanded={expandedModels[modelId]}
                onToggle={() => toggleModel(modelId)}
              />
            ))}
          </div>

          {/* Analysis Steps (collapsible) */}
          <div>
            <button onClick={() => setShowSteps(!showSteps)}
              className="w-full flex items-center justify-between p-2 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}>
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                Analysis Pipeline ({steps.length} steps)
              </span>
              {showSteps ? <ChevronUp className="w-3.5 h-3.5" style={{ color: 'var(--text-tertiary)' }} />
                         : <ChevronDown className="w-3.5 h-3.5" style={{ color: 'var(--text-tertiary)' }} />}
            </button>
            {showSteps && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                {steps.map((step, i) => <StepBadge key={i} step={step} />)}
              </div>
            )}
          </div>
        </>
      )}

      {/* Empty state */}
      {!data && !loading && !error && (
        <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
          <Eye className="w-6 h-6 mx-auto mb-2" style={{ color: 'var(--text-tertiary)', opacity: 0.5 }} />
          <p className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
            Cross-model behavioral analysis will run automatically when a response is generated in Glass mode.
          </p>
          <p className="text-[8px] mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Analyzes: Groq (LLaMA 3.1) · Llama 3.3 70B · Qwen 2.5 · QwenVL
          </p>
        </div>
      )}
    </div>
  );
};

/* ─── Mini Metric ───────────────────────────────────────────────── */
const MiniMetric = ({ label, value, color }) => (
  <div className="text-center p-1.5 rounded" style={{ backgroundColor: 'rgba(255,255,255,0.03)' }}>
    <div className="text-[8px] uppercase font-semibold" style={{ color: 'var(--text-tertiary)' }}>{label}</div>
    <div className="text-[11px] font-bold font-mono" style={{ color }}>{value}</div>
  </div>
);

export default CrossModelAnalytics;
