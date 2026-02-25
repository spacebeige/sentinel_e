/**
 * ModelsPage — Dynamic Model Registry Page
 * Fetches from GET /api/mco/models. Shows all registered MCO models
 * with provider, role, enabled status, and specs. Falls back to
 * static list if backend is offline.
 */
import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight, Check, XCircle, AlertCircle, Zap, Star } from 'lucide-react';
import useModels from '../hooks/useModels';

const FONT = "'Inter', -apple-system, sans-serif";

const ROLE_BADGES = {
  code: { label: 'Code Specialist', color: '#10b981' },
  vision: { label: 'Multimodal', color: '#06b6d4' },
  baseline: { label: 'Reasoning', color: '#6366f1' },
  conceptual: { label: 'Conceptual', color: '#8b5cf6' },
  longctx: { label: 'Long Context', color: '#f59e0b' },
};

const ROLE_DESCRIPTIONS = {
  code: 'Specialized in code generation, review, debugging, and multi-language architecture reasoning.',
  vision: 'Multimodal reasoning with visual document parsing, image understanding, and multilingual support.',
  baseline: 'General-purpose reasoning engine with balanced performance and broad knowledge coverage.',
  conceptual: 'Deep conceptual analysis with large context windows and efficient inference capabilities.',
  longctx: 'Optimized for long documents, extended reasoning chains, and session-level continuity.',
};

const ROLE_FEATURES = {
  code: ['Code generation & review', 'Multi-language support', 'Architecture reasoning', 'Debugging assistance'],
  vision: ['Vision & document parsing', 'Multimodal reasoning', 'Image understanding', 'Multilingual support'],
  baseline: ['General reasoning', 'Balanced performance', 'High throughput', 'Broad knowledge'],
  conceptual: ['128K context window', 'Deep conceptual analysis', 'Code specialist', 'Efficient inference'],
  longctx: ['262K context window', 'Long document analysis', 'Extended reasoning chains', 'Session continuity'],
};

const ROLE_SPEED = {
  code: 'Fast', vision: 'Medium', baseline: 'Fast', conceptual: 'Medium', longctx: 'Medium',
};

const ROLE_QUALITY = {
  code: 'High', vision: 'High', baseline: 'Good', conceptual: 'Very High', longctx: 'High',
};

/** Static fallback models (only used if backend is offline) */
const STATIC_MODELS = [
  {
    key: 'sentinel-e', name: 'Sentinel-E', provider: 'NeuralOS', role: 'conceptual', enabled: true,
    context_window: 131072, max_output_tokens: 16384,
  },
];

export function ModelsPage() {
  const { mcoModels, loading, error } = useModels();
  const models = mcoModels.length > 0 ? mcoModels : STATIC_MODELS;

  return (
    <div className="min-h-screen bg-[#f5f5f7] dark:bg-[#0f0f10] transition-colors">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-[#1d1d1f] dark:text-[#f1f5f9] mb-4"
            style={{ fontFamily: FONT, fontSize: 'clamp(36px, 5vw, 56px)', fontWeight: 700, letterSpacing: '-0.03em', lineHeight: 1.1 }}>
            Cognitive Model<br />
            <span className="bg-gradient-to-r from-[#8b5cf6] to-[#06b6d4] bg-clip-text text-transparent">Registry</span>
          </h1>
          <p className="text-[#6e6e73] dark:text-[#94a3b8] max-w-lg mx-auto"
            style={{ fontFamily: FONT, fontSize: '17px', lineHeight: 1.6, fontWeight: 400 }}>
            {models.length} specialized models powering the Meta-Cognitive Orchestrator.
            Each model is scored, arbitrated, and selected automatically.
          </p>
          {error && (
            <div className="flex items-center justify-center gap-2 mt-4 text-[#f59e0b]">
              <AlertCircle className="w-4 h-4" />
              <span style={{ fontFamily: FONT, fontSize: '13px' }}>Using cached model data — backend offline</span>
            </div>
          )}
        </motion.div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-8 h-8 border-2 border-[#8b5cf6] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model, index) => {
              const badge = ROLE_BADGES[model.role] || { label: model.role || 'General', color: '#6366f1' };
              const features = ROLE_FEATURES[model.role] || ['General purpose reasoning'];
              const isDisabled = model.enabled === false;
              const color = isDisabled ? '#9ca3af' : badge.color;

              return (
                <motion.div
                  key={model.key || model.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.08 }}
                  className={`group p-6 rounded-3xl border flex flex-col transition-all duration-300 ${
                    isDisabled
                      ? 'bg-[#f9fafb] dark:bg-[#1c1c1e] border-black/5 dark:border-white/10 opacity-60'
                      : 'bg-white dark:bg-[#1c1c1e] border-black/5 dark:border-white/10 hover:shadow-xl hover:shadow-black/5 dark:hover:shadow-black/30 hover:-translate-y-1'
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="w-12 h-12 rounded-2xl flex items-center justify-center"
                      style={{ backgroundColor: color + '15' }}>
                      <Sparkles className="w-6 h-6" style={{ color }} />
                    </div>
                    <div className="flex items-center gap-2">
                      {isDisabled && (
                        <span className="px-2 py-0.5 rounded-full border flex items-center gap-1"
                          style={{ backgroundColor: '#fef2f2', borderColor: '#fecaca', fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#ef4444' }}>
                          <XCircle className="w-3 h-3" /> Disabled
                        </span>
                      )}
                      <span className="px-3 py-1 rounded-full border" style={{
                        backgroundColor: color + '10', borderColor: color + '25', color,
                        fontFamily: FONT, fontSize: '11px', fontWeight: 600,
                      }}>
                        {badge.label}
                      </span>
                    </div>
                  </div>

                  <h3 className="text-[#1d1d1f] dark:text-[#f1f5f9] mb-1" style={{ fontFamily: FONT, fontSize: '20px', fontWeight: 600 }}>
                    {model.name}
                  </h3>
                  <p className="text-[#6e6e73] dark:text-[#94a3b8] mb-1" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}>
                    Provider: {model.provider}
                  </p>
                  {model.model_id && (
                    <p className="text-[#aeaeb2] dark:text-[#64748b] mb-2 font-mono truncate" style={{ fontSize: '11px' }}>
                      {model.model_id}
                    </p>
                  )}
                  <p className="text-[#6e6e73] dark:text-[#94a3b8] mb-4 leading-relaxed" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 400 }}>
                    {ROLE_DESCRIPTIONS[model.role] || 'General-purpose reasoning model for the MCO pipeline.'}
                  </p>

                  <div className="space-y-2 mb-5 flex-1">
                    {features.map((feature) => (
                      <div key={feature} className="flex items-center gap-2">
                        <Check className="w-4 h-4 text-[#34c759] flex-shrink-0" />
                        <span className="text-[#1d1d1f] dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 400 }}>
                          {feature}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Speed & Quality badges */}
                  <div className="flex items-center gap-2 mb-3">
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#fef3c7] dark:bg-[#fbbf24]/10">
                      <Zap className="w-3 h-3 text-[#f59e0b]" />
                      <span className="text-[#92400e] dark:text-[#fbbf24]" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600 }}>
                        {ROLE_SPEED[model.role] || 'Medium'}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#ede9fe] dark:bg-[#8b5cf6]/10">
                      <Star className="w-3 h-3 text-[#7c3aed]" />
                      <span className="text-[#5b21b6] dark:text-[#a78bfa]" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600 }}>
                        {ROLE_QUALITY[model.role] || 'Good'}
                      </span>
                    </div>
                  </div>

                  {/* Specs */}
                  <div className="flex flex-wrap items-center gap-2 mb-4">
                    {model.context_window && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f5f5f7] dark:bg-white/5">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}>
                          {Math.round(model.context_window / 1024)}K ctx
                        </span>
                      </div>
                    )}
                    {model.max_output_tokens && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f5f5f7] dark:bg-white/5">
                        <div className="w-1.5 h-1.5 rounded-full bg-[#007aff]" />
                        <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}>
                          {Math.round(model.max_output_tokens / 1024)}K out
                        </span>
                      </div>
                    )}
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f5f5f7] dark:bg-white/5">
                      <div className={`w-1.5 h-1.5 rounded-full ${isDisabled ? 'bg-[#ef4444]' : 'bg-[#34c759]'}`} />
                      <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}>
                        {isDisabled ? 'Inactive' : 'Active'}
                      </span>
                    </div>
                  </div>

                  <Link to="/chat"
                    className={`group/btn flex items-center justify-center gap-2 w-full py-2.5 rounded-2xl transition-all ${
                      isDisabled ? 'pointer-events-none opacity-50' : ''
                    }`}
                    style={{ backgroundColor: color + '10', color, fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}>
                    {isDisabled ? 'API Key Required' : `Try ${model.name}`}
                    {!isDisabled && <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-0.5 transition-transform" />}
                  </Link>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelsPage;
