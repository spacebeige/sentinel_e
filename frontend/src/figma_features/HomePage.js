/**
 * HomePage — Figma Feature Module (ported from figma_ui)
 * Composed from HeroSection + FeaturesSection + CTA + Footer.
 * Standalone — not wired into the chat engine.
 */
import React from 'react';
import { HeroSection } from './HeroSection';
import { FeaturesSection } from './FeaturesSection';
import { Footer } from './Footer';

const FONT = "'Inter', -apple-system, sans-serif";

function CTASection() {
  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-[#1d1d1f] mb-4"
          style={{ fontFamily: FONT, fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: 700, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
          Ready to experience<br />the future of AI?
        </h2>
        <p className="text-[#6e6e73] max-w-md mx-auto mb-8"
          style={{ fontFamily: FONT, fontSize: '17px', lineHeight: 1.6, fontWeight: 400 }}>
          Join millions of users who have already made the switch to smarter, more intuitive AI.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <a href="/chat"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-[#1d1d1f] text-white transition-all hover:scale-[1.02] active:scale-[0.98] shadow-xl shadow-black/15"
            style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 600 }}>
            Start Free
          </a>
          <a href="/pricing"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-[#f5f5f7] text-[#1d1d1f] transition-all hover:scale-[1.02] active:scale-[0.98]"
            style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 500 }}>
            View Pricing
          </a>
        </div>

        {/* Floating metrics */}
        <div className="mt-16 grid grid-cols-3 gap-4 max-w-lg mx-auto">
          {[
            { value: '10M+', label: 'Active Users' },
            { value: '500M+', label: 'Messages Sent' },
            { value: '4.9', label: 'App Rating' },
          ].map((stat) => (
            <div key={stat.label}>
              <div className="text-[#1d1d1f]"
                style={{ fontFamily: FONT, fontSize: '28px', fontWeight: 700, letterSpacing: '-0.02em' }}>
                {stat.value}
              </div>
              <div className="text-[#6e6e73]"
                style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 400 }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function HomePage() {
  return (
    <div>
      <HeroSection />
      <FeaturesSection />
      <CTASection />
      <Footer />
    </div>
  );
}

export default HomePage;
