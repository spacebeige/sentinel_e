/**
 * Footer — Figma Feature Module (ported from figma_ui)
 * 4-column footer with brand, product links, and legal.
 * Standalone — not wired into the chat engine.
 */
import React from 'react';
import SentinelIdentity from '../components/SentinelIdentity';

const FONT = "'Inter', -apple-system, sans-serif";

export function Footer() {
  return (
    <footer className="bg-[#1d1d1f] text-white py-16 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <SentinelIdentity size={34} />
              <span style={{ fontFamily: FONT, fontSize: '18px', fontWeight: 600 }}>Sentinel-E</span>
            </div>
            <p className="text-white/40" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.6, fontWeight: 400 }}>
              Persistent hybrid cognition with transparent orchestration, memory continuity, and permission-aware agentic execution.
            </p>
            <a
              href="https://www.producthunt.com/products/sentinel-e"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex mt-4 text-white/55 hover:text-white transition-colors"
              style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600 }}
            >
              Sentinel-E on Product Hunt
            </a>
          </div>

          {[
            { title: 'Product', links: ['Chat', 'Models', 'Agentic Runtime', 'Pricing'] },
            { title: 'Runtime', links: ['Orchestration Philosophy', 'Memory Behavior', 'Telemetry Handling', 'Permission System'] },
          { title: 'Transparency', links: ['Authentication', 'Privacy Expectations', 'Document Cognition', 'Product Hunt', 'Contact'] },
          ].map((section) => (
            <div key={section.title}>
              <h4 className="text-white/60 mb-3"
                style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {section.title}
              </h4>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link}>
                    <a
                      href={
                        link === 'Product Hunt' ? 'https://www.producthunt.com/products/sentinel-e' :
                        link === 'Contact' ? 'https://www.linkedin.com/in/oomkar-agarkhed-978613277/' :
                        '/'
                      }
                      target={link === 'Product Hunt' || link === 'Contact' ? '_blank' : undefined}
                      rel={link === 'Product Hunt' || link === 'Contact' ? 'noopener noreferrer' : undefined}
                      className="text-white/40 hover:text-white/80 transition-colors"
                      style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 400 }}>
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-white/10 pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-white/30" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 400 }}>
            © 2026 Sentinel-E. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            {['Privacy', 'Terms', 'Cookies'].map((link) => (
              <a key={link} href="/" className="text-white/30 hover:text-white/60 transition-colors"
                style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 400 }}>
                {link}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
