import { Github, Mail, Linkedin, BookOpen, ArrowUpRight } from "lucide-react";

const LINKS = {
  docs: "https://www.producthunt.com/products/sentinel-e?utm_source=other&utm_medium=social",
  github: "https://github.com/spacebeige/sentinel_e",
  email: "mailto:oomkaragarkhed0710@gmail.com",
  linkedin: "https://www.linkedin.com/in/oomkar-agarkhed-978613277/",
};

const NAV_COLS = [
  { title: "Product", items: [{ label: "Chat", href: "/chat" }, { label: "Engines", href: "/engines" }, { label: "Pricing", href: "/pricing" }] },
  { title: "Resources", items: [{ label: "Documentation", href: LINKS.docs, external: true }, { label: "GitHub", href: LINKS.github, external: true }] },
  { title: "Contact", items: [{ label: "oomkaragarkhed0710@gmail.com", href: LINKS.email, external: true }] },
];

export function Footer() {
  return (
    <footer className="bg-[#090b0f] text-white">
      {/* Top */}
      <div className="max-w-5xl mx-auto px-6 pt-16 pb-12">
        <div className="grid grid-cols-2 md:grid-cols-[2fr_1fr_1fr_1fr] gap-10">
          {/* Brand col */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <img
                src="/sentinel-e(1).png"
                onError={(e) => { if (!e.currentTarget.src.endsWith("/logo.png")) e.currentTarget.src = "/logo.png"; }}
                alt="Sentinel-E"
                className="h-[22px] w-auto object-contain"
              />
              <span style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: "14px" }}>Sentinel-E</span>
            </div>
            <p className="text-white/35 text-[13px] leading-relaxed max-w-[240px]">
              A cinematic cognitive operating system with hidden machine intelligence beneath glass.
            </p>
            <p className="text-white/20 text-[12px] leading-relaxed max-w-[240px] mt-2">
              Layered semantic orchestration, living machine reasoning, and multi-model cognition. Designed as a living intelligence system — not merely an interface, but a cognitive environment.
            </p>
            {/* Social icons */}
            <div className="flex items-center gap-3 mt-6">
              {[
                { icon: <Github className="w-4 h-4" />, href: LINKS.github, label: "GitHub" },
                { icon: <Linkedin className="w-4 h-4" />, href: LINKS.linkedin, label: "LinkedIn" },
                { icon: <Mail className="w-4 h-4" />, href: LINKS.email, label: "Email" },
                { icon: <BookOpen className="w-4 h-4" />, href: LINKS.docs, label: "Docs" },
              ].map((s) => (
                <a
                  key={s.label}
                  href={s.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={s.label}
                  className="w-8 h-8 rounded-xl flex items-center justify-center text-white/40 hover:text-white hover:bg-white/8 transition-all duration-200"
                >
                  {s.icon}
                </a>
              ))}
            </div>
          </div>

          {/* Link cols */}
          {NAV_COLS.map((col) => (
            <div key={col.title}>
              <div className="text-[10px] font-bold tracking-[0.18em] text-white/30 uppercase mb-4">{col.title}</div>
              <ul className="space-y-3">
                {col.items.map((item) => (
                  <li key={item.label}>
                    <a
                      href={item.href}
                      target={item.external ? "_blank" : undefined}
                      rel={item.external ? "noopener noreferrer" : undefined}
                      className="group inline-flex items-center gap-1 text-white/45 hover:text-white text-[13px] transition-colors duration-200"
                    >
                      {item.label}
                      {item.external && <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom bar */}
      <div
        className="border-t px-6 py-5"
        style={{ borderColor: "rgba(255,255,255,0.06)" }}
      >
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-white/25 text-[12px]">© 2026 Sentinel-E. All rights reserved.</p>
          <div className="flex items-center gap-5">
            {["Privacy", "Terms", "Security"].map((t) => (
              <a key={t} href="#" className="text-white/25 hover:text-white/50 text-[12px] transition-colors">{t}</a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}