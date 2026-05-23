import { motion } from "motion/react";
import { Brain, GitBranch, Eye, Shield, Cpu, Network } from "lucide-react";
import { GlassPanel } from "./GlassPanel";

const features = [
  {
    icon: Brain,
    title: "Multi-Agent Deliberation",
    description: "Multiple cognitive systems reason in parallel, forming and contesting positions in real time.",
    color: "cyan" as const,
  },
  {
    icon: Eye,
    title: "Visible Reasoning",
    description: "Every inference step, confidence score, and semantic link is surfaced — nothing is hidden.",
    color: "emerald" as const,
  },
  {
    icon: GitBranch,
    title: "Semantic Routing",
    description: "Queries are parsed, intent extracted, and dynamically routed to the optimal cognitive path.",
    color: "violet" as const,
  },
  {
    icon: Shield,
    title: "Governance Layer",
    description: "Continuous hallucination detection, boundary enforcement, and adversarial integrity checks.",
    color: "cyan" as const,
  },
  {
    icon: Cpu,
    title: "Model Ensemble",
    description: "Llama, Gemma, Mistral, GPT, Claude — orchestrated together for superior reasoning depth.",
    color: "amber" as const,
  },
  {
    icon: Network,
    title: "Topology Mapping",
    description: "Live semantic graph of agent relationships, evidence chains, and consensus emergence.",
    color: "emerald" as const,
  },
];

const colorMap = {
  cyan:    { bg: "bg-[rgba(110,231,249,0.08)]",  text: "text-[#6ee7f9]" },
  emerald: { bg: "bg-[rgba(52,211,153,0.08)]",   text: "text-[#34d399]" },
  violet:  { bg: "bg-[rgba(139,92,246,0.08)]",   text: "text-[#8b5cf6]" },
  amber:   { bg: "bg-[rgba(245,158,11,0.08)]",   text: "text-[#f59e0b]" },
};

export function FeaturesSection() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(110,231,249,0.15)] bg-[rgba(110,231,249,0.05)] mb-5">
            <div className="w-1 h-1 rounded-full bg-[#6ee7f9]" />
            <span className="text-[#6ee7f9] text-[10px] font-medium tracking-[0.2em] uppercase">Orchestration Capabilities</span>
          </div>
          <h2 className="text-[clamp(28px,4.5vw,48px)] font-bold tracking-[-0.025em] text-[#f3f5f7] leading-tight text-balance mb-4">
            Every layer of machine cognition,
            <br />
            <span className="text-[#8a9099] font-light">made visible</span>
          </h2>
          <p className="text-[#8a9099] text-sm max-w-lg mx-auto leading-relaxed">
            A complete orchestration stack — from raw inference to governed output.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, index) => {
            const c = colorMap[feature.color];
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.08 }}
              >
                <GlassPanel
                  glow={feature.color}
                  className="p-6 h-full hover:-translate-y-0.5 transition-transform duration-300 cursor-default"
                >
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center mb-4 ${c.bg} ${c.text}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <h3 className="text-[#f3f5f7] font-semibold mb-2 text-sm tracking-tight">
                    {feature.title}
                  </h3>
                  <p className="text-[#8a9099] text-xs leading-relaxed">
                    {feature.description}
                  </p>
                </GlassPanel>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
