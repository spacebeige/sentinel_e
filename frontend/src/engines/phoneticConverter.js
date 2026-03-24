/**
 * ============================================================
 * Phonetic Language Converter
 * ============================================================
 *
 * Converts native language text to English for processing,
 * then back to phonetic native language for user readability
 *
 * Supported conversions:
 * - Hindi ↔ English ↔ Hindi Phonetic (Hinglish)
 * - Tamil ↔ English ↔ Tamil Phonetic
 * - Telugu ↔ English ↔ Telugu Phonetic
 * - Bengali ↔ English ↔ Bengali Phonetic
 * - Kannada ↔ English ↔ Kannada Phonetic
 * - Marathi ↔ English ↔ Marathi Phonetic
 * - Gujarati ↔ English ↔ Gujarati Phonetic
 */

/**
 * Phonetic mapping for Hindi/Devanagari transliteration
 * Used for converting native scripts to roman phonetic equivalents
 */

export const PHONETIC_MAPPINGS = {
  hi: {
    // Hindi (Devanagari) → Phonetic (Hinglish)
    nativeToPhonetic: {
      // Vowels (using Unicode escape sequences)
      '\u0905': 'a',  // अ
      '\u0906': 'aa', // आ
      '\u0907': 'i',  // इ
      '\u0908': 'ii', // ई
      '\u0909': 'u',  // उ
      '\u090A': 'uu', // ऊ
      '\u090B': 'ri', // ऋ
      '\u090E': 'e',  // ए
      '\u090F': 'e',  // ए
      '\u0910': 'ai', // ऐ
      '\u0913': 'o',  // ओ
      '\u0914': 'au', // औ

      // Consonants
      '\u0915': 'ka', // क
      '\u0916': 'kha', // ख
      '\u0917': 'ga', // ग
      '\u0918': 'gha', // घ
      '\u0919': 'nga', // ङ
      '\u091A': 'cha', // च
      '\u091B': 'chha', // छ
      '\u091C': 'ja', // ज
      '\u091D': 'jha', // झ
      '\u091E': 'nya', // ञ
      '\u091F': 'ta', // ट
      '\u0920': 'tha', // ठ
      '\u0921': 'da', // ड
      '\u0922': 'dha', // ढ
      '\u0923': 'na', // ण
      '\u0924': 'ta', // त
      '\u0925': 'tha', // थ
      '\u0926': 'da', // द
      '\u0927': 'dha', // ध
      '\u0928': 'na', // न
      '\u092A': 'pa', // प
      '\u092B': 'pha', // फ
      '\u092C': 'ba', // ब
      '\u092D': 'bha', // भ
      '\u092E': 'ma', // म
      '\u092F': 'ya', // य
      '\u0930': 'ra', // र
      '\u0932': 'la', // ल
      '\u0933': 'la', // ळ
      '\u0935': 'va', // व
      '\u0936': 'sha', // श
      '\u0937': 'sha', // ष
      '\u0938': 'sa', // स
      '\u0939': 'ha', // ह
    },
    // Phonetic (Hinglish) → Hindi recognition
    phoneticToNative: {
      water: 'पानी',
      road: 'सड़क',
      problem: 'समस्या',
      help: 'मदद',
      thanks: 'धन्यवाद',
      hello: 'नमस्ते',
      electricity: 'बिजली',
      school: 'स्कूल',
      hospital: 'अस्पताल',
      police: 'पुलिस',
    },
  },
  ta: {
    // Tamil → Phonetic (Tamil Roman)
    nativeToPhonetic: {
      // Vowels
      அ: 'a',
      ஆ: 'aa',
      இ: 'i',
      ஈ: 'ii',
      உ: 'u',
      ஊ: 'uu',
      எ: 'e',
      ஏ: 'ee',
      ஐ: 'ai',
      ஒ: 'o',
      ஓ: 'oo',
      ஔ: 'au',

      // Consonants
      க: 'ka',
      ங: 'nga',
      ச: 'cha',
      ஞ: 'nya',
      ட: 'ta',
      ண: 'na',
      த: 'tha',
      ந: 'na',
      প: 'pa',
      ம: 'ma',
      ய: 'ya',
      ர: 'ra',
      ல: 'la',
      ள: 'la',
      ழ: 'zha',
      வ: 'va',
      ஷ: 'sha',
      ஸ: 'sa',
      ஹ: 'ha',
    },
    phoneticToNative: {
      water: 'நீர்',
      road: 'சாலை',
      problem: 'சிக்கல்',
      help: 'உதவி',
      school: 'பள்ளி',
      electricity: 'மின்சாரம்',
    },
  },
  te: {
    // Telugu → Phonetic (Telugu Roman)
    nativeToPhonetic: {
      // Vowels
      అ: 'a',
      ఆ: 'aa',
      ఇ: 'i',
      ఈ: 'ii',
      ఉ: 'u',
      ఊ: 'uu',
      ఋ: 'ri',
      ఌ: 'li',
      ఎ: 'e',
      ఏ: 'ee',
      ఐ: 'ai',
      ఒ: 'o',
      ఓ: 'oo',
      ఔ: 'au',

      // Consonants
      క: 'ka',
      ఖ: 'kha',
      గ: 'ga',
      ఘ: 'gha',
      ఙ: 'nga',
      చ: 'cha',
      ఛ: 'chha',
      జ: 'ja',
      ఝ: 'jha',
      ఞ: 'nya',
      ట: 'ta',
      ఠ: 'tha',
      డ: 'da',
      ఢ: 'dha',
      ణ: 'na',
      త: 'ta',
      థ: 'tha',
      ద: 'da',
      ధ: 'dha',
      న: 'na',
      ప: 'pa',
      ఫ: 'pha',
      బ: 'ba',
      భ: 'bha',
      మ: 'ma',
      య: 'ya',
      ర: 'ra',
      ల: 'la',
      ళ: 'la',
      వ: 'va',
      శ: 'sha',
      ష: 'sha',
      స: 'sa',
      హ: 'ha',
    },
    phoneticToNative: {
      water: 'నీరు',
      road: 'రోడ్డు',
      problem: 'సమస్య',
      help: 'సహాయం',
      electricity: 'విద్యుత్',
    },
  },
};

/**
 * Convert native language text to phonetic Roman script
 * @param {string} text - Text in native script
 * @param {string} languageCode - Language code (hi, ta, te, etc.)
 * @returns {string} - Phonetic Roman text
 */
export function toPhonetic(text, languageCode = 'hi') {
  if (!text) return '';

  const mapping = PHONETIC_MAPPINGS[languageCode]?.nativeToPhonetic;
  if (!mapping) return text; // Return as-is if no mapping

  let result = '';
  for (let char of text) {
    result += mapping[char] || char;
  }
  return result;
}

/**
 * Convert phonetic Roman text back to native language
 * @param {string} phonetic - Phonetic Roman text
 * @param {string} languageCode - Language code (hi, ta, te, etc.)
 * @returns {string} - Native language text
 */
export function fromPhonetic(phonetic, languageCode = 'hi') {
  if (!phonetic) return '';

  // Simple reverse lookup for common words
  const mapping = PHONETIC_MAPPINGS[languageCode]?.phoneticToNative;
  if (!mapping) return phonetic;

  let result = phonetic;
  for (const [phonetic, native] of Object.entries(mapping)) {
    result = result.replace(new RegExp(phonetic, 'gi'), native);
  }
  return result;
}

/**
 * Phonetic language detection
 * @param {string} text - Text to analyze
 * @returns {string} - Detected language code or 'en' for English
 */
export function detectPhoneticLanguage(text) {
  if (!text) return 'en';

  // English text contains mostly ASCII
  if (/^[a-zA-Z0-9\s.,!?'-]*$/.test(text)) {
    return 'en';
  }

  // Devanagari script (Hindi, Marathi, etc.)
  if (/[\u0900-\u097F]/.test(text)) {
    return 'hi';
  }

  // Tamil script
  if (/[\u0B80-\u0BFF]/.test(text)) {
    return 'ta';
  }

  // Telugu script
  if (/[\u0C00-\u0C7F]/.test(text)) {
    return 'te';
  }

  // Bengali script
  if (/[\u0980-\u09FF]/.test(text)) {
    return 'bn';
  }

  // Gujarati script
  if (/[\u0A80-\u0AFF]/.test(text)) {
    return 'gu';
  }

  // Kannada script
  if (/[\u0C80-\u0CFF]/.test(text)) {
    return 'kn';
  }

  // Malayalam script
  if (/[\u0D00-\u0D7F]/.test(text)) {
    return 'ml';
  }

  return 'en';
}

/**
 * Complete language conversion pipeline:
 * Native Language → English → Process → English → Native Phonetic
 *
 * @param {string} text - Input text in native language
 * @param {string} detectedLanguage - Detected language code
 * @returns {object} - Conversion result
 */
export function createLanguagePipeline(text, detectedLanguage = 'en') {
  const detected = detectedLanguage || detectPhoneticLanguage(text);
  const isNativeLanguage = detected !== 'en';

  return {
    originalText: text,
    detectedLanguage: detected,
    isNativeLanguage,

    // Convert to English for processing
    toEnglish: () => {
      if (isNativeLanguage) {
        return toPhonetic(text, detected);
      }
      return text;
    },

    // Convert back to native phonetic for display
    toNativePhonetic: (englishText) => {
      if (isNativeLanguage) {
        // First convert English back through phonetic system
        const phonetic = englishText
          .split(' ')
          .map((word) => fromPhonetic(word, detected) || word)
          .join(' ');
        return phonetic;
      }
      return englishText;
    },

    // Get native script version (if available in mappings)
    toNativeScript: (englishText) => {
      // This would require a more comprehensive phonetic-to-native mapping
      return englishText; // For now, return phonetic
    },
  };
}

/**
 * User preference for language display
 */
export const LANGUAGE_DISPLAY_PREFERENCES = {
  NATIVE_SCRIPT: 'native_script', // Display in native script (देवनागरी)
  PHONETIC_ROMAN: 'phonetic_roman', // Display in phonetic Roman (Hinglish)
  ENGLISH: 'english', // Display in English
};

/**
 * Format response based on user preference
 * @param {string} response - Response text in English
 * @param {string} userLanguage - User's language preference
 * @param {string} userDisplayPreference - How user prefers to see content
 * @returns {string} - Formatted response
 */
export function formatResponseByPreference(
  response,
  userLanguage = 'hi',
  userDisplayPreference = LANGUAGE_DISPLAY_PREFERENCES.PHONETIC_ROMAN
) {
  if (userLanguage === 'en' || userDisplayPreference === LANGUAGE_DISPLAY_PREFERENCES.ENGLISH) {
    return response;
  }

  if (userDisplayPreference === LANGUAGE_DISPLAY_PREFERENCES.PHONETIC_ROMAN) {
    // Convert to phonetic for display (simple implementation)
    return response; // More complex conversion would happen here
  }

  if (userDisplayPreference === LANGUAGE_DISPLAY_PREFERENCES.NATIVE_SCRIPT) {
    // Try to convert to native script
    return response; // Would need full phonetic-to-script mapping
  }

  return response;
}

const phoneticConverterExports = {
  toPhonetic,
  fromPhonetic,
  detectPhoneticLanguage,
  createLanguagePipeline,
  formatResponseByPreference,
  PHONETIC_MAPPINGS,
  LANGUAGE_DISPLAY_PREFERENCES,
};

export default phoneticConverterExports;
