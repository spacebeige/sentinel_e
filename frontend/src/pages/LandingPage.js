/**
 * LandingPage.js — Route: /
 * Renders the Figma-ported marketing landing page.
 * No backend logic. Pure presentation.
 */
import React from 'react';
import { Link } from 'react-router-dom';
import { HeroSection } from '../figma_features/HeroSection';
import { FeaturesSection } from '../figma_features/FeaturesSection';

const FONT = "'Inter', -apple-system, sans-serif";

function CTASection() {
  return (
    <section className="py-24 px-6 bg-white dark:bg-[#1d1d1f] transition-colors duration-300">
      <div className="max-w-4xl mx-auto text-center">
        <h2
          className="text-[#1d1d1f] dark:text-white mb-4"
          style={{ fontFamily: FONT, fontSize: 'clamp(32px, 5vw, 48px)', fontWeight: 700, letterSpacing: '-0.02em', lineHeight: 1.1 }}
        >
          Ready to experience<br />the future of AI?
        </h2>
        <p
          className="text-[#6e6e73] dark:text-white/60 max-w-md mx-auto mb-8"
          style={{ fontFamily: FONT, fontSize: '17px', lineHeight: 1.6, fontWeight: 400 }}
        >
          Join millions of users who have already made the switch to smarter, more intuitive AI.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-[#1d1d1f] dark:bg-white text-white dark:text-[#1d1d1f] transition-all hover:scale-[1.02] active:scale-[0.98] shadow-xl shadow-black/15"
            style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 600 }}
          >
            Start Free
          </Link>
          <Link
            to="/pricing"
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-[#f5f5f7] dark:bg-white/10 text-[#1d1d1f] dark:text-white transition-all hover:scale-[1.02] active:scale-[0.98]"
            style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 500 }}
          >
            View Pricing
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeaturesSection />
      <CTASection />
    </div>
  );
}
