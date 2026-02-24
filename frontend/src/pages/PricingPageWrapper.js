/**
 * PricingPageWrapper.js — Route: /pricing
 * Wraps the Figma-ported PricingPage component.
 * No backend logic. Pure presentation.
 */
import React from 'react';
import { PricingPage } from '../figma_features/PricingPage';

export default function PricingPageWrapper() {
  return <PricingPage />;
}
