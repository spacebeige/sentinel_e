import { motion } from "motion/react";
import { Check, Zap } from "lucide-react";
import { Link } from "react-router";
import { GlassPanel } from "./GlassPanel";

const plans = [
  {
    name: "Free",
    price: "$0",
    period: "forever",
    description: "Get started with AI — no credit card required.",
    features: [
      "50 messages per day",
      "GPT-4o Mini access",
      "Basic chat features",
      "7-day chat history",
      "Community support",
    ],
    cta: "Get Started",
    popular: false,
    gradient: "",
  },
  {
    name: "Pro",
    price: "$20",
    period: "/month",
    description: "Unlimited access to all models and premium features.",
    features: [
      "Unlimited messages",
      "All AI models access",
      "Priority speed",
      "Unlimited chat history",
      "File uploads & analysis",
      "Custom instructions",
      "Priority support",
    ],
    cta: "Start Free Trial",
    popular: true,
    gradient: "from-[#3b82f6] to-[#06b6d4]",
  },
  {
    name: "Team",
    price: "$35",
    period: "/user/month",
    description: "Collaborate with your team using shared AI workspaces.",
    features: [
      "Everything in Pro",
      "Team workspaces",
      "Admin dashboard",
      "Usage analytics",
      "SSO & SAML",
      "API access",
      "Dedicated support",
      "Custom model fine-tuning",
    ],
    cta: "Contact Sales",
    popular: false,
    gradient: "",
  },
];

export function PricingPage() {
  return (
    <div className="min-h-screen pt-24">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(110,231,249,0.15)] bg-[rgba(110,231,249,0.05)] mb-5">
            <div className="w-1 h-1 rounded-full bg-[#6ee7f9]" />
            <span className="text-[#6ee7f9] text-[10px] font-medium tracking-[0.2em] uppercase">Access Tiers</span>
          </div>
          <h1 className="text-[clamp(32px,5vw,54px)] font-bold tracking-[-0.03em] text-[#f3f5f7] leading-tight mb-4 text-balance">
            Transparent access
            <br />
            <span className="text-[#8a9099] font-light">to the runtime</span>
          </h1>
          <p className="text-[#8a9099] max-w-md mx-auto text-sm leading-relaxed">
            Start free, scale when ready. No hidden fees, cancel anytime.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={plan.popular ? "md:-mt-4" : ""}
            >
              <GlassPanel
                glow={plan.popular ? "cyan" : "none"}
                className="relative p-6 flex flex-col"
                border
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-[#6ee7f9] text-[#060708] text-[10px] font-bold tracking-wide">
                      <Zap className="w-2.5 h-2.5" />
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="mb-5">
                  <h3 className="text-[#f3f5f7] font-semibold text-base mb-2 tracking-tight">
                    {plan.name}
                  </h3>
                  <div className="flex items-baseline gap-1 mb-2">
                    <span className={`text-[40px] font-bold tracking-[-0.04em] ${plan.popular ? "text-[#6ee7f9]" : "text-[#f3f5f7]"}`}>
                      {plan.price}
                    </span>
                    <span className="text-[#8a9099] text-sm">{plan.period}</span>
                  </div>
                  <p className="text-[#8a9099] text-xs leading-relaxed">{plan.description}</p>
                </div>

                <div className="space-y-2 mb-5 flex-1">
                  {plan.features.map((feature) => (
                    <div key={feature} className="flex items-center gap-2">
                      <Check className="w-3.5 h-3.5 text-[#34d399] shrink-0" />
                      <span className="text-[#c7cbd1] text-xs">{feature}</span>
                    </div>
                  ))}
                </div>

                <Link
                  to="/chat"
                  className={`block text-center py-2.5 rounded-lg text-xs font-semibold tracking-wide transition-all hover:scale-[1.02] active:scale-[0.98] ${
                    plan.popular
                      ? "bg-[#6ee7f9] text-[#060708] hover:bg-[rgba(110,231,249,0.85)]"
                      : "bg-[rgba(110,231,249,0.06)] border border-[rgba(110,231,249,0.15)] text-[#c7cbd1] hover:text-[#f3f5f7] hover:bg-[rgba(110,231,249,0.1)]"
                  }`}
                >
                  {plan.cta}
                </Link>
              </GlassPanel>
            </motion.div>
          ))}
        </div>

        {/* FAQ */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mt-20 max-w-2xl mx-auto"
        >
          <h2 className="text-center text-[#f3f5f7] font-bold text-2xl tracking-tight mb-8">
            Frequently Asked
          </h2>
          {[
            { q: "Can I switch plans anytime?", a: "Yes, you can upgrade, downgrade, or cancel your plan at any time. Changes take effect at the start of your next billing cycle." },
            { q: "Is there a free trial for Pro?", a: "Yes! All Pro features come with a 14-day free trial. No credit card required to start." },
            { q: "What happens when I reach my message limit?", a: "On the Free plan, you'll be prompted to upgrade. We'll never cut you off mid-conversation." },
          ].map((faq) => (
            <GlassPanel key={faq.q} className="mb-3 p-5">
              <h4 className="text-[#f3f5f7] font-semibold text-sm mb-2">{faq.q}</h4>
              <p className="text-[#8a9099] text-xs leading-relaxed">{faq.a}</p>
            </GlassPanel>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
