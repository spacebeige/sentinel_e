import { Layers } from "lucide-react";
import { Link } from "react-router";

const FOOTER_LINKS = [
  {
    title: "Platform",
    links: [
      { label: "Deliberation", to: "/chat" },
      { label: "Debate Arena", to: "/debate" },
      { label: "Mission Control", to: "/mission-control" },
      { label: "Governance", to: "/governance" },
    ],
  },
  {
    title: "Models",
    links: [
      { label: "Model Library", to: "/models" },
      { label: "Pricing", to: "/pricing" },
      { label: "API Docs", to: "#" },
      { label: "Status", to: "#" },
    ],
  },
  {
    title: "Company",
    links: [
      { label: "About", to: "#" },
      { label: "Blog", to: "#" },
      { label: "Careers", to: "#" },
      { label: "Press", to: "#" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-[rgba(110,231,249,0.07)] py-14 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-7 h-7 rounded-lg bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)] flex items-center justify-center">
                <Layers className="w-3.5 h-3.5 text-[#6ee7f9]" />
              </div>
              <div className="flex flex-col leading-none">
                <span className="text-[#f3f5f7] font-semibold text-sm tracking-tight">SENTINEL</span>
                <span className="text-[#6ee7f9] text-[9px] font-medium tracking-[0.18em] uppercase">E · Runtime</span>
              </div>
            </div>
            <p className="text-[#8a9099] text-xs leading-relaxed max-w-[200px]">
              Visible machine cognition. Multi-agent AI orchestration for the most demanding reasoning operations.
            </p>
          </div>

          {/* Nav columns */}
          {FOOTER_LINKS.map((section) => (
            <div key={section.title}>
              <h4 className="text-[#6ee7f9] text-[10px] font-semibold tracking-[0.15em] uppercase mb-3">
                {section.title}
              </h4>
              <ul className="space-y-2">
                {section.links.map(({ label, to }) => (
                  <li key={label}>
                    <Link
                      to={to}
                      className="text-[#8a9099] hover:text-[#c7cbd1] transition-colors text-xs"
                    >
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-[rgba(110,231,249,0.07)] pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-[#8a9099] text-xs">
            © 2026 Sentinel-E. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            {["Privacy", "Terms", "Security"].map((link) => (
              <a
                key={link}
                href="#"
                className="text-[#8a9099] hover:text-[#c7cbd1] transition-colors text-xs"
              >
                {link}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
