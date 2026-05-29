import { useTheme } from "next-themes";
import { useState, useEffect } from "react";
import { Link } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { Check, Moon, Sun } from "lucide-react";
import { useAuthContext } from "../providers/AuthProvider";

const PLANS = [
  {
    id: "standard",
    name: "Standard",
    price: "Free",
    sub: "Forever",
    description: "The complete Sentinel-E experience for personal use.",
    cta: "Sign Up",
    href: "/chat",
    features: [
      "Access to base models (GPT-4o mini, Llama 3.1)",
      "Standard inference latency",
      "Basic conversation memory",
      "Standard context windows",
    ],
  },
  {
    id: "pro",
    name: "Sentinel Pro",
    price: "$20",
    sub: "per month",
    description: "Unleash the full orchestration engine for advanced reasoning.",
    cta: "Upgrade to Pro",
    href: "#",
    primary: true,
    features: [
      "Full model suite (GPT-4o, Claude 3.5, Gemini 1.5)",
      "Sentinel Σ orchestration routing",
      "Debate & Synthesis modes",
      "Extended context windows",
      "Priority inference latency",
      "Advanced tool use & code execution",
    ],
  },
  {
    id: "enterprise",
    name: "Enterprise",
    price: "Custom",
    sub: "deployment",
    description: "Custom deployments with dedicated infrastructure.",
    cta: "Contact Sales",
    href: "#",
    features: [
      "Everything in Pro",
      "Dedicated inference endpoints",
      "Custom system prompts & rules",
      "SAML / SSO authentication",
      "SOC2 compliance",
      "Custom model fine-tuning",
    ],
  },
];

export function PricingPage() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  const { isAuthenticated } = useAuthContext();

  const toggleTheme = () => {
    setTheme(isDark ? "light" : "dark");
  };

  return (
    <div className={`min-h-screen transition-colors duration-500 pt-28 pb-24 px-6 ${isDark ? "bg-[#09090b] text-[#f5f5f7]" : "bg-[#f5f5f7] text-[#1d1d1f]"}`}>
      {/* Compact Navbar */}
      <div className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4" style={{ background: isDark ? "rgba(9,9,11,0.8)" : "rgba(245,245,247,0.8)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)", borderBottom: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.05)" }}>
        <Link to="/" className="flex items-center transition-transform hover:scale-105">
          <img src="/logo.png" alt="Logo" className="h-6 w-auto" />
        </Link>
        <button onClick={toggleTheme} className="flex items-center justify-center w-9 h-9 rounded-full transition-colors duration-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.07]">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={isDark ? "sun" : "moon"} initial={{ opacity: 0, rotate: -90, scale: 0.8 }} animate={{ opacity: 1, rotate: 0, scale: 1 }} exit={{ opacity: 0, rotate: 90, scale: 0.8 }} transition={{ duration: 0.2 }}>
              {isDark ? <Sun className="w-[15px] h-[15px] text-[rgba(255,255,255,0.55)]" /> : <Moon className="w-[15px] h-[15px] text-[rgba(0,0,0,0.4)]" />}
            </motion.div>
          </AnimatePresence>
        </button>
      </div>

      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <div
            className="inline-flex items-center gap-2 mb-5 px-3 py-1.5 rounded-full"
            style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)", border: isDark ? "1px solid rgba(255,255,255,0.07)" : "1px solid rgba(0,0,0,0.06)" }}
          >
            <span className="text-[10px] font-bold tracking-[0.22em] text-[#8e8e93] uppercase">Access Layer</span>
          </div>
          <h1
            className={`mb-3 ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`}
            style={{ fontFamily: "'Inter', sans-serif", fontSize: "clamp(34px, 5.5vw, 56px)", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 1 }}
          >
            Simple pricing.
          </h1>
          <p className={`max-w-sm mx-auto ${isDark ? "text-[#636366]" : "text-[#8e8e93]"}`} style={{ fontSize: "16px", lineHeight: 1.6 }}>
            Start free. Unlock the full orchestration layer when you need it.
          </p>
        </motion.div>

        {/* Plans */}
        <div className="grid md:grid-cols-2 gap-5">
          {PLANS.map((plan, i) => {
            const cardBgClass = isDark
              ? "bg-white/[0.04] border-white/[0.08]"
              : "bg-black/[0.04] border-black/[0.06]";
            
            const textClass = isDark
              ? "text-[#f5f5f7]"
              : "text-[#1d1d1f]";

            return (
              <motion.div
                key={plan.id}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
                className={`p-8 rounded-3xl relative overflow-hidden border ${cardBgClass} ${plan.primary ? (isDark ? "!border-white/20" : "!border-black/20") : ""}`}
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
                  <div className={`inline-flex items-center gap-1.5 mb-4 px-2.5 py-1 rounded-full ${isDark ? "bg-white/10" : "bg-black/10"}`}>
                    <span className={`text-[10px] font-bold tracking-[0.18em] uppercase ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`}>Recommended</span>
                  </div>
                )}

                <div className={`text-[15px] font-semibold mb-1 ${textClass}`}>
                  {plan.name}
                </div>
                <div className="flex items-baseline gap-1.5 mb-1">
                  <span
                    className={`font-800 ${textClass}`}
                    style={{ fontSize: "42px", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 1 }}
                  >
                    {plan.price}
                  </span>
                  <span className={isDark ? "text-[#8e8e93] text-[13px]" : "text-[#636366] text-[13px]"}>
                    {plan.sub}
                  </span>
                </div>
                <p className={`mb-7 text-[13px] leading-relaxed ${isDark ? "text-[#8e8e93]" : "text-[#636366]"}`}>
                  {plan.description}
                </p>

                <ul className="space-y-2.5 mb-8">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-2.5">
                      <Check
                        className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`}
                      />
                      <span
                        className={`text-[13px] ${isDark ? "text-[#8e8e93]" : "text-[#636366]"}`}
                      >
                        {f}
                      </span>
                    </li>
                  ))}
                </ul>

                <Link
                  to={plan.href === "/chat" && !isAuthenticated ? "/signup" : plan.href}
                  className={`flex items-center justify-center w-full py-3 rounded-2xl font-semibold text-[14px] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] ${
                    plan.primary
                      ? isDark
                        ? "bg-[#f5f5f7] text-[#1d1d1f]"
                        : "bg-[#1d1d1f] text-[#f5f5f7]"
                      : isDark
                      ? "bg-white/10 text-[#f5f5f7]"
                      : "bg-black/5 text-[#1d1d1f]"
                  }`}
                >
                  {plan.id === "standard" ? (isAuthenticated ? "Chat" : "Sign Up") : plan.cta}
                </Link>
              </div>
            </motion.div>
            );
          })}
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
