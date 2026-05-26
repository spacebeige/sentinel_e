import { Link } from "react-router";
import { motion } from "motion/react";
import { Check } from "lucide-react";

const PLANS = [
  {
    id: "standard",
    name: "Standard",
    price: "Free",
    sub: "Forever",
    description: "The complete Sentinel-E experience for personal use.",
    cta: "Initialize",
    href: "/chat",
    primary: false,
    features: [
      "Unlimited conversations",
      "GPT-4o, Claude, Gemini access",
      "Semantic memory",
      "Conversation export",
      "Mobile-native interface",
    ],
  },
  {
    id: "pro",
    name: "Pro",
    price: "$12",
    sub: "per month",
    description: "Advanced orchestration for power users and researchers.",
    cta: "Initialize Pro",
    href: "/chat",
    primary: true,
    features: [
      "Everything in Standard",
      "Multi-model orchestration",
      "Council, Debate & Sigma modes",
      "Glass transparency layer",
      "Forensic evidence mode",
      "Governance oversight",
      "Priority inference routing",
      "API access",
    ],
  },
];

export function PricingPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-[#090b0f] pt-28 pb-24 px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <div
            className="inline-flex items-center gap-2 mb-5 px-3 py-1.5 rounded-full"
            style={{ background: "rgba(0,0,0,0.04)", border: "1px solid rgba(0,0,0,0.06)" }}
          >
            <span className="text-[10px] font-bold tracking-[0.22em] text-[#8e8e93] uppercase">Access Layer</span>
          </div>
          <h1
            className="text-[#1d1d1f] dark:text-white mb-3"
            style={{ fontFamily: "'Inter', sans-serif", fontSize: "clamp(34px, 5.5vw, 56px)", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 1 }}
          >
            Simple pricing.
          </h1>
          <p className="text-[#8e8e93] dark:text-[#636366] max-w-sm mx-auto" style={{ fontSize: "16px", lineHeight: 1.6 }}>
            Start free. Unlock the full orchestration layer when you need it.
          </p>
        </motion.div>

        {/* Plans */}
        <div className="grid md:grid-cols-2 gap-5">
          {PLANS.map((plan, i) => (
            <motion.div
              key={plan.id}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="p-8 rounded-3xl relative overflow-hidden"
              style={{
                background: plan.primary ? "#1d1d1f" : "rgba(0,0,0,0.03)",
                border: plan.primary ? "none" : "1px solid rgba(0,0,0,0.07)",
              }}
            >
              {/* Pro grid texture */}
              {plan.primary && (
                <div
                  className="absolute inset-0 pointer-events-none opacity-[0.04]"
                  style={{
                    backgroundImage: "linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)",
                    backgroundSize: "32px 32px",
                  }}
                />
              )}

              <div className="relative">
                {plan.primary && (
                  <div className="inline-flex items-center gap-1.5 mb-4 px-2.5 py-1 rounded-full bg-white/10">
                    <span className="text-[10px] font-bold tracking-[0.18em] text-white/60 uppercase">Recommended</span>
                  </div>
                )}

                <div className={`text-[15px] font-semibold mb-1 ${plan.primary ? "text-white" : "text-[#1d1d1f] dark:text-white"}`}>
                  {plan.name}
                </div>
                <div className="flex items-baseline gap-1.5 mb-1">
                  <span
                    className={`font-800 ${plan.primary ? "text-white" : "text-[#1d1d1f] dark:text-white"}`}
                    style={{ fontSize: "42px", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 1 }}
                  >
                    {plan.price}
                  </span>
                  <span className={plan.primary ? "text-white/40 text-[13px]" : "text-[#8e8e93] text-[13px]"}>
                    {plan.sub}
                  </span>
                </div>
                <p className={`mb-7 text-[13px] leading-relaxed ${plan.primary ? "text-white/45" : "text-[#8e8e93]"}`}>
                  {plan.description}
                </p>

                <ul className="space-y-2.5 mb-8">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-2.5">
                      <Check
                        className="w-3.5 h-3.5 flex-shrink-0"
                        style={{ color: plan.primary ? "rgba(255,255,255,0.5)" : "#8e8e93" }}
                      />
                      <span
                        className="text-[13px]"
                        style={{ color: plan.primary ? "rgba(255,255,255,0.75)" : "#6e6e73" }}
                      >
                        {f}
                      </span>
                    </li>
                  ))}
                </ul>

                <Link
                  to={plan.href}
                  className="flex items-center justify-center w-full py-3 rounded-2xl font-semibold text-[14px] transition-all duration-200 hover:scale-[1.01] active:scale-[0.99]"
                  style={{
                    background: plan.primary ? "white" : "#1d1d1f",
                    color: plan.primary ? "#1d1d1f" : "white",
                  }}
                >
                  {plan.cta}
                </Link>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.4 }}
          className="text-center text-[12px] text-[#8e8e93] dark:text-[#636366] mt-8"
        >
          No credit card required for Standard. Cancel Pro anytime.
        </motion.p>
      </div>
    </div>
  );
}
