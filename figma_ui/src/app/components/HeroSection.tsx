import { Link } from "react-router";
import { ArrowRight, Zap, Shield, Brain } from "lucide-react";
import { motion } from "motion/react";

export function HeroSection() {
  return (
    <section className="relative min-h-screen overflow-hidden">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-[#0a0a1a]">
        {/* Animated gradient orbs */}
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-[#0ea5e9]/20" style={{ filter: 'blur(120px)' }} />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[#5eead4]/15" style={{ filter: 'blur(120px)' }} />
        <div className="absolute top-[30%] right-[20%] w-[30%] h-[30%] rounded-full bg-[#6366f1]/15" style={{ filter: 'blur(100px)' }} />
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
      </div>

      {/* Hero Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-32 pb-20 min-h-screen flex flex-col justify-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="max-w-3xl"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/15 backdrop-blur-md border border-white/20 mb-6"
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 500 }}
          >
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-white/90">Powered by latest AI models</span>
          </div>

          <h1
            className="text-white mb-6"
            style={{
              fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
              fontSize: 'clamp(40px, 7vw, 72px)',
              fontWeight: 700,
              lineHeight: 1.05,
              letterSpacing: '-0.03em',
            }}
          >
            Intelligence,
            <br />
            <span className="bg-gradient-to-r from-[#7dd3fc] to-[#5eead4] bg-clip-text text-transparent">
              Reimagined.
            </span>
          </h1>

          <p
            className="text-white/70 max-w-xl mb-8"
            style={{
              fontFamily: "'Inter', -apple-system, sans-serif",
              fontSize: 'clamp(16px, 2vw, 20px)',
              lineHeight: 1.6,
              fontWeight: 400,
            }}
          >
            Experience the next generation of AI with a beautifully crafted interface.
            Seamless conversations, powerful models, designed for everyone.
          </p>

          <div className="flex flex-wrap gap-3">
            <Link
              to="/chat"
              className="group inline-flex items-center gap-2 px-7 py-3 rounded-2xl bg-white text-[#1d1d1f] transition-all hover:scale-[1.02] active:scale-[0.98] shadow-2xl shadow-black/20"
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '16px', fontWeight: 600 }}
            >
              Start Chatting
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              to="/models"
              className="inline-flex items-center gap-2 px-7 py-3 rounded-2xl bg-white/10 backdrop-blur-md text-white border border-white/20 transition-all hover:bg-white/20 hover:scale-[1.02] active:scale-[0.98]"
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '16px', fontWeight: 500 }}
            >
              Explore Models
            </Link>
          </div>
        </motion.div>

        {/* Feature Pills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-2xl"
        >
          {[
            { icon: <Zap className="w-4 h-4" />, label: "Lightning Fast", desc: "Sub-second responses" },
            { icon: <Shield className="w-4 h-4" />, label: "Private & Secure", desc: "End-to-end encryption" },
            { icon: <Brain className="w-4 h-4" />, label: "Multi-Model", desc: "Qwen, Mistral, Groq" },
          ].map((feature) => (
            <div
              key={feature.label}
              className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-white/10 backdrop-blur-md border border-white/10"
            >
              <div className="w-8 h-8 rounded-xl bg-white/15 flex items-center justify-center text-white/80">
                {feature.icon}
              </div>
              <div>
                <div
                  className="text-white/90"
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 600 }}
                >
                  {feature.label}
                </div>
                <div
                  className="text-white/50"
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 400 }}
                >
                  {feature.desc}
                </div>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}