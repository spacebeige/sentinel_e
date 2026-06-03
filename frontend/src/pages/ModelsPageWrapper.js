/**
 * ModelsPageWrapper.js — Route: /models
 * Wraps the Figma-ported ModelsPage component.
 * No backend logic. Pure presentation.
 */
import React from 'react';
import { ModelsPage } from '../figma_features/ModelsPage';

export default function ModelsPageWrapper() {
  return <ModelsPage />;
}
