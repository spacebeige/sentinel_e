import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
    Activity, Shield, AlertTriangle, Brain, TrendingUp, 
    RefreshCw, ChevronRight, BarChart2
} from 'lucide-react';

const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const fixed2 = v => v != null ? Number(v).toFixed(2) : '—';

const MetricCard = ({ label, value, color, icon: Icon, sub }) => (
  <div className="p-4 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
    <div className="flex items-center justify-between mb-2">
      <span className="text-[9px] uppercase font-bold tracking-wider" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
      {Icon && <Icon className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />}
    </div>
    <div className="text-2xl font-bold" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
    {sub && <div className="text-[10px] mt-1" style={{ color: 'var(--text-tertiary)' }}>{sub}</div>}
  </div>
);

const RiskBar = ({ label, score, maxScore = 1 }) => {
  const v = Math.min((score / maxScore) * 100, 100);
  const color = score >= 0.7 ? 'var(--accent-red)' : score >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="text-[10px] w-24 truncate font-mono" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
      <div className="flex-1 progress-bar">
        <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono w-12 text-right" style={{ color: 'var(--text-tertiary)' }}>{fixed2(score)}</span>
    </div>
  );
};

const LearningDashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');

    const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await axios.get(`${API_BASE_URL}/api/learning`);
            setData(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to load learning data');
        } finally {
            setLoading(false);
        }
    }, [API_BASE_URL]);

    useEffect(() => { fetchData(); }, [fetchData]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64" style={{ color: 'var(--text-tertiary)' }}>
                <Activity className="w-5 h-5 animate-spin mr-2" />
                <span className="text-sm">Loading learning matrix...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center p-8 rounded-lg"
              style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--accent-red)' }}>
                <AlertTriangle className="w-10 h-10 mb-3" style={{ color: 'var(--accent-red)' }} />
                <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Learning Error</h3>
                <p className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>{error}</p>
                <button onClick={fetchData} className="mt-4 text-xs px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-colors"
                  style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-primary)', color: 'var(--text-secondary)' }}>
                    <RefreshCw className="w-3 h-3" /> Retry
                </button>
            </div>
        );
    }

    if (!data || data.status === 'disabled') {
        return (
            <div className="flex items-center justify-center p-8 rounded-lg"
              style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--accent-yellow)' }}>
                <div className="text-center">
                    <AlertTriangle className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--accent-yellow)' }} />
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Learning Not Active</h3>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>{data?.message || 'KnowledgeLearner not initialized.'}</p>
                </div>
            </div>
        );
    }

    const summary = data.summary || {};
    const suggestions = data.threshold_suggestions || {};
    const riskProfiles = data.risk_profiles || {};
    const claimRisks = data.claim_type_risks || {};

    const tabs = [
        { id: 'overview', label: 'Overview', icon: BarChart2 },
        { id: 'models', label: 'Model Risks', icon: Brain },
        { id: 'claims', label: 'Claim Types', icon: Shield },
        { id: 'suggestions', label: 'Thresholds', icon: TrendingUp },
    ];

    return (
        <div className="flex-1 overflow-y-auto p-6 scrollbar-thin" style={{ color: 'var(--text-primary)' }}>
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5" style={{ color: 'var(--accent-green)' }} />
                    <h2 className="text-lg font-bold tracking-tight">Knowledge Learning Matrix</h2>
                    <span className="text-[9px] font-mono px-2 py-0.5 rounded-full"
                      style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-tertiary)' }}>v4.5</span>
                </div>
                <button onClick={fetchData} className="p-2 rounded-lg transition-colors"
                  style={{ color: 'var(--text-tertiary)' }} title="Refresh">
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* Tab Navigation */}
            <div className="flex items-center gap-1 mb-6 p-1 rounded-lg"
              style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                {tabs.map(tab => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] uppercase font-bold tracking-wider rounded-md transition-all"
                        style={{
                            backgroundColor: activeTab === tab.id ? 'rgba(34,197,94,0.1)' : 'transparent',
                            color: activeTab === tab.id ? 'var(--accent-green)' : 'var(--text-tertiary)',
                            border: activeTab === tab.id ? '1px solid rgba(34,197,94,0.2)' : '1px solid transparent',
                        }}>
                        <tab.icon className="w-3 h-3" />
                        <span>{tab.label}</span>
                    </button>
                ))}
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <MetricCard label="Violations" value={summary.total_violations_recorded || 0}
                            color="var(--accent-red)" icon={AlertTriangle} sub="boundary violations recorded" />
                        <MetricCard label="Refusals" value={summary.total_refusals_issued || 0}
                            color="var(--accent-yellow)" icon={Shield} sub="responses refused" />
                        <MetricCard label="Models Tracked" value={summary.models_tracked || 0}
                            color="var(--accent-blue)" icon={Brain} />
                        <MetricCard label="Feedback" value={summary.feedback_collected || summary.total_feedback || 0}
                            color="var(--accent-green)" icon={Activity} sub="user ratings collected" />
                    </div>
                    {summary.avg_satisfaction != null && (
                        <div className="p-4 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                            <h4 className="text-[10px] uppercase font-bold mb-3" style={{ color: 'var(--text-tertiary)' }}>User Satisfaction</h4>
                            <div className="flex items-center gap-4">
                                <span className="text-3xl font-bold" style={{ color: 'var(--accent-green)' }}>{pct(summary.avg_satisfaction)}</span>
                                <div className="flex-1 progress-bar" style={{ height: '10px' }}>
                                    <div className="progress-bar-fill" style={{ width: `${(summary.avg_satisfaction || 0) * 100}%`, backgroundColor: 'var(--accent-green)' }} />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Model Risks Tab */}
            {activeTab === 'models' && (
                <div className="space-y-4">
                    {Object.keys(riskProfiles).length === 0 ? (
                        <div className="text-center py-8 text-sm" style={{ color: 'var(--text-tertiary)' }}>No model risk data yet.</div>
                    ) : (
                        Object.entries(riskProfiles).map(([model, profile]) => (
                            <div key={model} className="p-4 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="text-xs font-bold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
                                        <Brain className="w-3 h-3" style={{ color: 'var(--accent-blue)' }} />
                                        <span>{model}</span>
                                    </h4>
                                    <span className={`risk-badge--${(profile.risk_level || 'low').toLowerCase()} text-[10px]`}>
                                        {profile.risk_level || 'LOW'}
                                    </span>
                                </div>
                                <div className="grid grid-cols-2 gap-3 text-xs">
                                    <div><span style={{ color: 'var(--text-tertiary)' }}>Violations:</span> <span style={{ color: 'var(--text-primary)' }}>{profile.total_violations || 0}</span></div>
                                    <div><span style={{ color: 'var(--text-tertiary)' }}>Avg Severity:</span> <span style={{ color: 'var(--text-primary)' }}>{fixed2(profile.avg_severity)}</span></div>
                                    <div><span style={{ color: 'var(--text-tertiary)' }}>Refusal Rate:</span> <span style={{ color: 'var(--text-primary)' }}>{pct(profile.refusal_rate)}</span></div>
                                    <div><span style={{ color: 'var(--text-tertiary)' }}>Trend:</span> <span style={{ color: 'var(--text-primary)' }}>{profile.trend || '—'}</span></div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}

            {/* Claim Types Tab */}
            {activeTab === 'claims' && (
                <div className="space-y-4">
                    {Object.keys(claimRisks).length === 0 ? (
                        <div className="text-center py-8 text-sm" style={{ color: 'var(--text-tertiary)' }}>No claim type data yet.</div>
                    ) : (
                        <div className="p-4 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                            <h4 className="text-[10px] uppercase font-bold mb-4 flex items-center gap-2"
                              style={{ color: 'var(--text-secondary)' }}>
                                <Shield className="w-3 h-3" style={{ color: 'var(--accent-yellow)' }} />
                                Claim Type Risk Distribution
                            </h4>
                            <div className="space-y-1">
                                {Object.entries(claimRisks).map(([claim, risk]) => (
                                    <RiskBar key={claim} label={claim} score={typeof risk === 'number' ? risk : risk.avg_severity || 0} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Threshold Suggestions Tab */}
            {activeTab === 'suggestions' && (
                <div className="space-y-4">
                    {Object.keys(suggestions).length === 0 ? (
                        <div className="text-center py-8 text-sm" style={{ color: 'var(--text-tertiary)' }}>No threshold suggestions yet.</div>
                    ) : (
                        Object.entries(suggestions).map(([model, suggestion]) => (
                            <div key={model} className="p-4 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                                <div className="flex items-center gap-2 mb-3">
                                    <TrendingUp className="w-3 h-3" style={{ color: 'var(--accent-green)' }} />
                                    <h4 className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{model}</h4>
                                </div>
                                {typeof suggestion === 'object' ? (
                                    <div className="space-y-2">
                                        {Object.entries(suggestion).map(([key, val]) => (
                                            <div key={key} className="flex items-center justify-between text-xs">
                                                <span style={{ color: 'var(--text-tertiary)' }}>{key}:</span>
                                                <span className="font-mono flex items-center" style={{ color: 'var(--text-primary)' }}>
                                                    <ChevronRight className="w-3 h-3 mr-1" style={{ color: 'var(--accent-green)' }} />
                                                    {typeof val === 'number' ? fixed2(val) : String(val)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{String(suggestion)}</p>
                                )}
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};

export default LearningDashboard;
