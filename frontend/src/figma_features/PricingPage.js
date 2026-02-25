/**
 * PricingPage — Figma Feature Module (ported from figma_ui)
 * 3 pricing tiers (Free/Pro/Team) with FAQ section.
 * Standalone — not wired into the chat engine.
 */
import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Check, Sparkles } from 'lucide-react';

const FONT = "'Inter', -apple-system, sans-serif";

const plans = [
  {
    name: 'Free', price: '$0', period: 'forever', popular: false, gradient: '',
    description: 'Get started with AI — no credit card required.',
    features: ['50 messages per day', 'GPT-4o Mini access', 'Basic chat features', '7-day chat history', 'Community support'],
    cta: 'Get Started',
  },
  {
    name: 'Pro', price: '$20', period: '/month', popular: true, gradient: 'from-[#3b82f6] to-[#06b6d4]',
    description: 'Unlimited access to all models and premium features.',
    features: ['Unlimited messages', 'All AI models access', 'Priority speed', 'Unlimited chat history', 'File uploads & analysis', 'Custom instructions', 'Priority support'],
    cta: 'Start Free Trial',
  },
  {
    name: 'Team', price: '$35', period: '/user/month', popular: false, gradient: '',
    description: 'Collaborate with your team using shared AI workspaces.',
    features: ['Everything in Pro', 'Team workspaces', 'Admin dashboard', 'Usage analytics', 'SSO & SAML', 'API access', 'Dedicated support', 'Custom model fine-tuning'],
    cta: 'Contact Sales',
  },
];

const faqs = [
  { q: 'Can I switch plans anytime?', a: 'Yes, you can upgrade, downgrade, or cancel your plan at any time. Changes take effect at the start of your next billing cycle.' },
  { q: 'Is there a free trial for Pro?', a: "Yes! All Pro features come with a 14-day free trial. No credit card required to start." },
  { q: 'What happens when I reach my message limit?', a: "On the Free plan, you'll be prompted to upgrade. We'll never cut you off mid-conversation." },
];

export function PricingPage() {
  return (
    <div className="min-h-screen bg-[#f5f5f7] dark:bg-[#0f0f10] transition-colors">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-[#1d1d1f] dark:text-[#f1f5f9] mb-4"
            style={{ fontFamily: FONT, fontSize: 'clamp(36px, 5vw, 56px)', fontWeight: 700, letterSpacing: '-0.03em', lineHeight: 1.1 }}>
            Simple,<br />
            <span className="bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] bg-clip-text text-transparent">transparent pricing.</span>
          </h1>
          <p className="text-[#6e6e73] dark:text-[#94a3b8] max-w-lg mx-auto"
            style={{ fontFamily: FONT, fontSize: '17px', lineHeight: 1.6, fontWeight: 400 }}>
            Start free, upgrade when you're ready. No hidden fees, cancel anytime.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`relative p-6 rounded-3xl border flex flex-col ${
                plan.popular
                  ? 'bg-[#1d1d1f] border-transparent shadow-2xl shadow-black/20 md:-mt-4 md:mb-0'
                  : 'bg-white dark:bg-[#1c1c1e] border-black/5 dark:border-white/5'
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="inline-flex items-center gap-1.5 px-4 py-1 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white shadow-lg"
                    style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600 }}>
                    <Sparkles className="w-3 h-3" />
                    Most Popular
                  </span>
                </div>
              )}

              <div className="mb-6">
                <h3 className={plan.popular ? 'text-white mb-2' : 'text-[#1d1d1f] dark:text-[#f1f5f9] mb-2'}
                  style={{ fontFamily: FONT, fontSize: '20px', fontWeight: 600 }}>
                  {plan.name}
                </h3>
                <div className="flex items-baseline gap-1 mb-2">
                  <span className={plan.popular ? 'text-white' : 'text-[#1d1d1f] dark:text-[#f1f5f9]'}
                    style={{ fontFamily: FONT, fontSize: '48px', fontWeight: 700, letterSpacing: '-0.03em' }}>
                    {plan.price}
                  </span>
                  <span className={plan.popular ? 'text-white/50' : 'text-[#6e6e73]'}
                    style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 400 }}>
                    {plan.period}
                  </span>
                </div>
                <p className={plan.popular ? 'text-white/60' : 'text-[#6e6e73] dark:text-[#94a3b8]'}
                  style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.5, fontWeight: 400 }}>
                  {plan.description}
                </p>
              </div>

              <div className="space-y-3 mb-6 flex-1">
                {plan.features.map((feature) => (
                  <div key={feature} className="flex items-center gap-2.5">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${
                      plan.popular ? 'bg-white/10' : 'bg-[#34c759]/10'
                    }`}>
                      <Check className="w-3 h-3" style={{ color: plan.popular ? '#5eead4' : '#34c759' }} />
                    </div>
                    <span className={plan.popular ? 'text-white/80' : 'text-[#1d1d1f] dark:text-[#e2e8f0]'}
                      style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 400 }}>
                      {feature}
                    </span>
                  </div>
                ))}
              </div>

              <Link to="/chat"
                className={`block text-center py-3 rounded-2xl transition-all hover:scale-[1.02] active:scale-[0.98] ${
                  plan.popular
                    ? `bg-gradient-to-r ${plan.gradient} text-white shadow-lg shadow-blue-500/30`
                    : 'bg-[#f5f5f7] dark:bg-white/10 text-[#1d1d1f] dark:text-[#f1f5f9] hover:bg-[#e8e8ed] dark:hover:bg-white/15'
                }`}
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 600 }}>
                {plan.cta}
              </Link>
            </motion.div>
          ))}
        </div>

        {/* FAQ section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mt-20 max-w-2xl mx-auto"
        >
          <h2 className="text-center text-[#1d1d1f] dark:text-[#f1f5f9] mb-8"
            style={{ fontFamily: FONT, fontSize: '28px', fontWeight: 700, letterSpacing: '-0.02em' }}>
            Frequently Asked
          </h2>
          {faqs.map((faq) => (
            <div key={faq.q} className="mb-4 p-5 rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5">
              <h4 className="text-[#1d1d1f] dark:text-[#f1f5f9] mb-2"
                style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 600 }}>
                {faq.q}
              </h4>
              <p className="text-[#6e6e73] dark:text-[#94a3b8]"
                style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, fontWeight: 400 }}>
                {faq.a}
              </p>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}

export default PricingPage;
