import { Sigma } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-[#1d1d1f] text-white py-16 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
                <Sigma className="w-4 h-4 text-white" />
              </div>
              <span
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '18px', fontWeight: 600 }}
              >Sentinel-E</span>
            </div>
            <p
              className="text-white/40"
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', lineHeight: 1.6, fontWeight: 400 }}
            >
              Intelligence, reimagined. The next generation of AI, designed for everyone.
            </p>
          </div>

          {[
            {
              title: "Product",
              links: ["Chat", "Models", "API", "Pricing"],
            },
            {
              title: "Company",
              links: ["About", "Blog", "Careers", "Press"],
            },
            {
              title: "Resources",
              links: ["Documentation", "Help Center", "Community", "Status"],
            },
          ].map((section) => (
            <div key={section.title}>
              <h4
                className="text-white/60 mb-3"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}
              >
                {section.title}
              </h4>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-white/40 hover:text-white/80 transition-colors"
                      style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 400 }}
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-white/10 pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p
            className="text-white/30"
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 400 }}
          >© 2026 Sentinel-E. All rights reserved.</p>
          <div className="flex items-center gap-6">
            {["Privacy", "Terms", "Cookies"].map((link) => (
              <a
                key={link}
                href="#"
                className="text-white/30 hover:text-white/60 transition-colors"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 400 }}
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