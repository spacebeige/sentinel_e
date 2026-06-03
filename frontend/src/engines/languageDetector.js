/**
 * ============================================================
 * Language Detector & Model Selector
 * ============================================================
 *
 * Handles:
 *   - Language detection from transcribed text
 *   - Script detection (Devanagari, Tamil, Telugu, etc.)
 *   - Model & TTS voice selection based on language
 *   - Multi-language support mapping
 */

export const SUPPORTED_LANGUAGES = {
  en: {
    name: 'English',
    nativeName: 'English',
    scripts: ['Latin'],
    gtts: 'en',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus', 'llama-3.1-8b'],
  },
  hi: {
    name: 'Hindi',
    nativeName: 'हिंदी',
    scripts: ['Devanagari'],
    gtts: 'hi',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus', 'llama-3.1-8b'],
  },
  ta: {
    name: 'Tamil',
    nativeName: 'தமிழ்',
    scripts: ['Tamil'],
    gtts: 'ta',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  te: {
    name: 'Telugu',
    nativeName: 'తెలుగు',
    scripts: ['Telugu'],
    gtts: 'te',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  kn: {
    name: 'Kannada',
    nativeName: 'ಕನ್ನಡ',
    scripts: ['Kannada'],
    gtts: 'kn',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  ml: {
    name: 'Malayalam',
    nativeName: 'മലയാളം',
    scripts: ['Malayalam'],
    gtts: 'ml',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  mr: {
    name: 'Marathi',
    nativeName: 'मराठी',
    scripts: ['Devanagari'],
    gtts: 'mr',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  gu: {
    name: 'Gujarati',
    nativeName: 'ગુજરાતી',
    scripts: ['Gujarati'],
    gtts: 'gu',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  bn: {
    name: 'Bengali',
    nativeName: 'বাংলা',
    scripts: ['Bengali'],
    gtts: 'bn',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  pa: {
    name: 'Punjabi',
    nativeName: 'ਪੰਜਾਬੀ',
    scripts: ['Gurmukhi'],
    gtts: 'pa',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  or: {
    name: 'Odia',
    nativeName: 'ଓଡ଼ିଆ',
    scripts: ['Odia'],
    gtts: 'or',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
  as: {
    name: 'Assamese',
    nativeName: 'অসমীয়া',
    scripts: ['Bengali'],
    gtts: 'bn',
    ttsProvider: 'google',
    models: ['gpt-4'],
  },
  ur: {
    name: 'Urdu',
    nativeName: 'اردو',
    scripts: ['Nastaliq'],
    gtts: 'ur',
    ttsProvider: 'google',
    models: ['gpt-4', 'claude-3-opus'],
  },
};

// Script detection patterns
const SCRIPT_PATTERNS = {
  Devanagari: /[\u0900-\u097F]/u,
  Tamil: /[\u0B80-\u0BFF]/u,
  Telugu: /[\u0C00-\u0C7F]/u,
  Kannada: /[\u0C80-\u0CFF]/u,
  Malayalam: /[\u0D00-\u0D7F]/u,
  Gujarati: /[\u0A80-\u0AFF]/u,
  Gurmukhi: /[\u0A00-\u0A7F]/u,
  Bengali: /[\u0980-\u09FF]/u,
  Odia: /[\u0B00-\u0B7F]/u,
  Nastaliq: /[\u0600-\u06FF]/u,
};

/**
 * Detect language from text using script analysis
 * @param {string} text - Text to analyze
 * @returns {Object} - Language detection result
 */
export function detectLanguageFromText(text) {
  if (!text || !text.trim()) {
    return {
      language: 'en',
      confidence: 0,
      detectedScripts: [],
    };
  }

  const detectedScripts = [];
  const scriptCounts = {};

  // Check for script patterns
  for (const [script, pattern] of Object.entries(SCRIPT_PATTERNS)) {
    if (pattern.test(text)) {
      detectedScripts.push(script);
      scriptCounts[script] = (text.match(pattern) || []).length;
    }
  }

  // Map scripts to languages
  let detectedLanguage = 'en';
  let confidence = 0;

  if (detectedScripts.length > 0) {
    const primaryScript = detectedScripts[0];
    const languageMap = {
      Devanagari: 'hi',
      Tamil: 'ta',
      Telugu: 'te',
      Kannada: 'kn',
      Malayalam: 'ml',
      Gujarati: 'gu',
      Gurmukhi: 'pa',
      Bengali: 'bn',
      Odia: 'or',
      Nastaliq: 'ur',
    };

    detectedLanguage = languageMap[primaryScript] || 'en';
    // Confidence based on percentage of script characters
    const scriptCharCount = scriptCounts[primaryScript] || 0;
    confidence = Math.min(scriptCharCount / text.length, 1);
  }

  return {
    language: detectedLanguage,
    confidence,
    detectedScripts,
    isMultilingual: detectedScripts.length > 1,
  };
}

/**
 * Get language info
 */
export function getLanguageInfo(languageCode) {
  return (
    SUPPORTED_LANGUAGES[languageCode] || SUPPORTED_LANGUAGES.en
  );
}

/**
 * Select appropriate model based on language and task
 */
export function selectModel(
  language,
  taskType = 'standard',
  preferredModel = null
) {
  const langInfo = getLanguageInfo(language);
  const availableModels = langInfo.models || ['gpt-4'];

  // If preferred model is available in this language, use it
  if (preferredModel && availableModels.includes(preferredModel)) {
    return preferredModel;
  }

  // Select based on task type
  switch (taskType) {
    case 'debate':
      // Use most capable model for debate
      if (availableModels.includes('claude-3-opus')) {
        return 'claude-3-opus';
      }
      if (availableModels.includes('gpt-4')) {
        return 'gpt-4';
      }
      break;

    case 'evidence':
      // Include evidence synthesis models
      if (availableModels.includes('gpt-4')) {
        return 'gpt-4';
      }
      if (availableModels.includes('claude-3-opus')) {
        return 'claude-3-opus';
      }
      break;

    case 'standard':
    default:
      // Return first available model
      return availableModels[0];
  }

  return availableModels[0];
}

/**
 * Get TTS voice for language
 */
export function getTTSVoice(language, gender = 'female') {
  const langInfo = getLanguageInfo(language);
  return {
    language: language,
    gttslang: langInfo.gtts,
    provider: langInfo.ttsProvider,
    gender: gender,
  };
}

/**
 * Check if language is RTL (right-to-left)
 */
export function isRTLLanguage(language) {
  return ['ur', 'ar'].includes(language);
}

/**
 * Format language detection result for display
 */
export function formatLanguageDetection(detectionResult) {
  const langInfo = getLanguageInfo(detectionResult.language);
  return {
    displayLanguage: langInfo.name,
    nativeScript: langInfo.nativeName,
    confidence: `${Math.round(detectionResult.confidence * 100)}%`,
    isMultilingual: detectionResult.isMultilingual,
    scripts: detectionResult.detectedScripts.join(', '),
  };
}

/**
 * Comprehensive language & model selection for audio input
 */
export function analyzeLanguageAndSelectModel(
  transcribedText,
  detectedLanguageFromAPI = null,
  taskType = 'standard',
  preferredModel = null
) {
  // First try API-detected language
  let targetLanguage = detectedLanguageFromAPI || 'en';

  // If not provided, detect from text
  if (!detectedLanguageFromAPI) {
    const detection = detectLanguageFromText(transcribedText);
    targetLanguage = detection.language;
  }

  // Validate language is supported
  if (!SUPPORTED_LANGUAGES[targetLanguage]) {
    targetLanguage = 'en';
  }

  // Select model based on language and task
  const selectedModel = selectModel(
    targetLanguage,
    taskType,
    preferredModel
  );

  // Get TTS voice
  const ttsVoice = getTTSVoice(targetLanguage);

  return {
    language: targetLanguage,
    languageInfo: getLanguageInfo(targetLanguage),
    selectedModel,
    ttsVoice,
    isRTL: isRTLLanguage(targetLanguage),
    preserveLanguageInResponse: targetLanguage !== 'en',
  };
}

const languageDetectorExports = {
  detectLanguageFromText,
  getLanguageInfo,
  selectModel,
  getTTSVoice,
  isRTLLanguage,
  formatLanguageDetection,
  analyzeLanguageAndSelectModel,
  SUPPORTED_LANGUAGES,
};

export default languageDetectorExports;
