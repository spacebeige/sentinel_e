import { motion } from "motion/react";
import {
  MessageSquare,
  Layers,
  Palette,
  Globe,
  Lock,
  Cpu,
} from "lucide-react";

const features = [
  {
    icon: <MessageSquare className="w-6 h-6" />,
    title: "Natural Conversations",
    description:
      "Context-aware dialogue that understands nuance, remembers context, and responds naturally.",
    gradient: "from-[#3b82f6] to-[#8b5cf6]",
    shadowColor: "rgba(59,130,246,0.35)",
  },
  {
    icon: <Layers className="w-6 h-6" />,
    title: "Multi-Model Support",
    description:
      "Switch between Qwen, Mistral, Groq, and more with a single tap.",
    gradient: "from-[#06b6d4] to-[#3b82f6]",
    shadowColor: "rgba(6,182,212,0.35)",
  },
  {
    icon: <Palette className="w-6 h-6" />,
    title: "Creative Tools",
    description:
      "Generate images, code, documents, and creative content with integrated AI tools.",
    gradient: "from-[#f59e0b] to-[#ef4444]",
    shadowColor: "rgba(245,158,11,0.35)",
  },
  {
    icon: <Globe className="w-6 h-6" />,
    title: "100+ Languages",
    description:
      "Communicate in over 100 languages with real-time translation and localization.",
    gradient: "from-[#10b981] to-[#06b6d4]",
    shadowColor: "rgba(16,185,129,0.35)",
  },
  {
    icon: <Lock className="w-6 h-6" />,
    title: "Privacy First",
    description:
      "Your conversations are encrypted and never used for training. Full data control.",
    gradient: "from-[#8b5cf6] to-[#ec4899]",
    shadowColor: "rgba(139,92,246,0.35)",
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: "Edge Computing",
    description:
      "On-device processing for faster responses and complete offline capabilities.",
    gradient: "from-[#14b8a6] to-[#22c55e]",
    shadowColor: "rgba(20,184,166,0.35)",
  },
];

export function FeaturesSection() {
  return (
    <section className="py-24 px-6 bg-[#f5f5f7]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2
            className="text-[#1d1d1f] mb-4"
            style={{
              fontFamily: "'Inter', -apple-system, sans-serif",
              fontSize: "clamp(32px, 5vw, 48px)",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              lineHeight: 1.1,
            }}
          >
            Designed for
            <br />
            <span className="bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] bg-clip-text text-transparent">
              the way you think.
            </span>
          </h2>
          <p
            className="text-[#6e6e73] max-w-lg mx-auto"
            style={{
              fontFamily: "'Inter', -apple-system, sans-serif",
              fontSize: "17px",
              lineHeight: 1.6,
              fontWeight: 400,
            }}
          >
            Every feature is crafted to feel intuitive, fast,
            and delightful.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group p-6 rounded-3xl bg-white border border-black/5 hover:shadow-xl hover:shadow-black/5 transition-all duration-300 hover:-translate-y-1 cursor-default"
            >
              <div
                className={`w-11 h-11 rounded-[14px] bg-gradient-to-br ${feature.gradient} flex items-center justify-center text-white mb-5 group-hover:scale-105 group-hover:-rotate-3 transition-all duration-300`}
                style={{ boxShadow: `0 4px 14px -2px ${feature.shadowColor}` }}
              >
                {feature.icon}
              </div>
              <h3
                className="text-[#1d1d1f] mb-2"
                style={{
                  fontFamily:
                    "'Inter', -apple-system, sans-serif",
                  fontSize: "18px",
                  fontWeight: 600,
                }}
              >
                {feature.title}
              </h3>
              <p
                className="text-[#6e6e73]"
                style={{
                  fontFamily:
                    "'Inter', -apple-system, sans-serif",
                  fontSize: "15px",
                  lineHeight: 1.6,
                  fontWeight: 400,
                }}
              >
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}