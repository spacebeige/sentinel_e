"""Phonetic Language Conversion System - Feminine Voice Edition
Converts native scripts to phonetic romanization for all supported languages.
This prevents TTS mispronunciation and ensures natural, clear feminine speech synthesis.
"""

import re
from typing import Dict, Optional

class PhoneticConverter:
    """Convert native language scripts to phonetic romanization."""
    
    # Hindi (Devanagari) to Phonetic Romanization
    HINDI_TO_PHONETIC = {
        'ा': 'aa',
        'ि': 'i',
        'ी': 'ee',
        'ु': 'u',
        'ू': 'oo',
        'ृ': 'ri',
        'े': 'e',
        'ै': 'ai',
        'ो': 'o',
        'ौ': 'au',
        'ं': 'ng',
        'ः': 'h',
        'ँ': 'n',
        'अ': 'a',
        'आ': 'aa',
        'इ': 'i',
        'ई': 'ee',
        'उ': 'u',
        'ऊ': 'oo',
        'ऋ': 'ri',
        'ए': 'e',
        'ऐ': 'ai',
        'ओ': 'o',
        'औ': 'au',
        'क': 'ka',
        'ख': 'kha',
        'ग': 'ga',
        'घ': 'gha',
        'ङ': 'nga',
        'च': 'cha',
        'छ': 'chha',
        'ज': 'ja',
        'झ': 'jha',
        'ञ': 'nya',
        'ट': 'ta',
        'ठ': 'tha',
        'ड': 'da',
        'ढ': 'dha',
        'ण': 'na',
        'त': 'ta',
        'थ': 'tha',
        'द': 'da',
        'ध': 'dha',
        'न': 'na',
        'प': 'pa',
        'फ': 'pha',
        'ब': 'ba',
        'भ': 'bha',
        'म': 'ma',
        'य': 'ya',
        'र': 'ra',
        'ल': 'la',
        'व': 'va',
        'श': 'sha',
        'ष': 'sha',
        'स': 'sa',
        'ह': 'ha',
        'क्ष': 'ksh',
        'त्र': 'tra',
        'ज्ञ': 'gya',
    }
    
    # Marathi (Same script as Hindi with some variations)
    MARATHI_TO_PHONETIC = HINDI_TO_PHONETIC.copy()
    MARATHI_TO_PHONETIC.update({
        'ळ': 'la',
        'ऩ': 'na',
        'ॉ': 'o',
    })
    
    # Tamil to Phonetic
    TAMIL_TO_PHONETIC = {
        'அ': 'a',
        'ஆ': 'aa',
        'இ': 'i',
        'ஈ': 'ee',
        'உ': 'u',
        'ஊ': 'oo',
        'எ': 'e',
        'ஏ': 'ai',
        'ஐ': 'ai',
        'ஒ': 'o',
        'ஓ': 'o',
        'ஔ': 'au',
        'க': 'ka',
        'ங': 'nga',
        'ச': 'cha',
        'ஞ': 'nya',
        'ட': 'ta',
        'ண': 'na',
        'த': 'tha',
        'ந': 'na',
        'ப': 'pa',
        'ம': 'ma',
        'ய': 'ya',
        'ர': 'ra',
        'ல': 'la',
        'வ': 'va',
        'ழ': 'zha',
        'ள': 'la',
        'ற': 'ra',
        'ன': 'na',
        '்': '',  # Virama (no sound)
        'ா': 'aa',
        'ி': 'i',
        'ீ': 'ee',
        'ு': 'u',
        'ூ': 'oo',
        'ெ': 'e',
        'ே': 'ai',
        'ை': 'ai',
        'ொ': 'o',
        'ோ': 'o',
        'ௌ': 'au',
        'ம': 'm',
        'ன': 'n',
        'ں': 'n',
        '?': 'ng',
    }
    
    # Telugu to Phonetic
    TELUGU_TO_PHONETIC = {
        'అ': 'a',
        'ా': 'aa',
        'ిി': 'i',
        'ీ': 'ee',
        'ు': 'u',
        'ూ': 'oo',
        'ృ': 'ri',
        'ె': 'e',
        'ే': 'ay',
        'ైः': 'ai',
        'ొ': 'o',
        'ో': 'o',
        'ౌ': 'au',
        'క': 'ka',
        'ఖ': 'kha',
        'గ': 'ga',
        'ఘ': 'gha',
        'ఙ': 'nga',
        'చ': 'cha',
        'ఛ': 'chha',
        'జ': 'ja',
        'ఝ': 'jha',
        'ఞ': 'nya',
        'ట': 'ta',
        'ఠ': 'tha',
        'డ': 'da',
        'ఢ': 'dha',
        'ణ': 'na',
        'త': 'tha',
        'థ': 'tha',
        'ద': 'da',
        'ధ': 'dha',
        'న': 'na',
        'ప': 'pa',
        'ఫ': 'pha',
        'బ': 'ba',
        'భ': 'bha',
        'మ': 'ma',
        'య': 'ya',
        'ర': 'ra',
        'ల': 'la',
        'వ': 'va',
        'శ': 'sha',
        'ష': 'sha',
        'స': 'sa',
        'హ': 'ha',
    }
    
    # Kannada to Phonetic
    KANNADA_TO_PHONETIC = {
        'ಅ': 'a',
        'ಆ': 'aa',
        'ಇ': 'i',
        'ಈ': 'ee',
        'ಉ': 'u',
        'ಊ': 'oo',
        'ಋ': 'ri',
        'ಎ': 'e',
        'ಏ': 'ay',
        'ಐ': 'ai',
        'ಒ': 'o',
        'ಓ': 'o',
        'ಔ': 'au',
        'ಕ': 'ka',
        'ಖ': 'kha',
        'ಗ': 'ga',
        'ಘ': 'gha',
        'ಙ': 'nga',
        'ಚ': 'cha',
        'ಛ': 'chha',
        'ಜ': 'ja',
        'ಝ': 'jha',
        'ಞ': 'nya',
        'ಟ': 'ta',
        'ಠ': 'tha',
        'ಡ': 'da',
        'ಢ': 'dha',
        'ಣ': 'na',
        'ತ': 'tha',
        'ಥ': 'tha',
        'ದ': 'da',
        'ಧ': 'dha',
        'ನ': 'na',
        'ಪ': 'pa',
        'ಫ': 'pha',
        'ಬ': 'ba',
        'ಭ': 'bha',
        'ಮ': 'ma',
        'ಯ': 'ya',
        'ರ': 'ra',
        'ಲ': 'la',
        'ವ': 'va',
        'ಶ': 'sha',
        'ಷ': 'sha',
        'ಸ': 'sa',
        'ಹ': 'ha',
        'ಳ': 'la',
    }
    
    # Malayalam to Phonetic
    MALAYALAM_TO_PHONETIC = {
        'അ': 'a',
        'ആ': 'aa',
        'ഇ': 'i',
        'ഈ': 'ee',
        'ഉ': 'u',
        'ഊ': 'oo',
        'ഋ': 'ri',
        'എ': 'e',
        'ഏ': 'ay',
        'ഐ': 'ai',
        'ഒ': 'o',
        'ഓ': 'o',
        'ഔ': 'au',
        'ക': 'ka',
        'ഖ': 'kha',
        'ഗ': 'ga',
        'ഘ': 'gha',
        'ങ': 'nga',
        'ച': 'cha',
        'ഛ': 'chha',
        'ജ': 'ja',
        'ഝ': 'jha',
        'ഞ': 'nya',
        'ട': 'ta',
        'ഠ': 'tha',
        'ഡ': 'da',
        'ഢ': 'dha',
        'ണ': 'na',
        'ത': 'tha',
        'ഥ': 'tha',
        'ദ': 'da',
        'ധ': 'dha',
        'ന': 'na',
        'പ': 'pa',
        'ഫ': 'pha',
        'ബ': 'ba',
        'ഭ': 'bha',
        'മ': 'ma',
        'യ': 'ya',
        'ര': 'ra',
        'ല': 'la',
        'വ': 'va',
        'ശ': 'sha',
        'ഷ': 'sha',
        'സ': 'sa',
        'ഹ': 'ha',
        'ള': 'la',
        'ഴ': 'zha',
    }
    
    # Bengali to Phonetic
    BENGALI_TO_PHONETIC = {
        'অ': 'a',
        'আ': 'aa',
        'ই': 'i',
        'ঈ': 'ee',
        'উ': 'u',
        'ঊ': 'oo',
        'ঋ': 'ri',
        'এ': 'e',
        'ঐ': 'ai',
        'ও': 'o',
        'ঔ': 'au',
        'ক': 'ka',
        'খ': 'kha',
        'গ': 'ga',
        'ঘ': 'gha',
        'ঙ': 'nga',
        'চ': 'cha',
        'ছ': 'chha',
        'জ': 'ja',
        'ঝ': 'jha',
        'ঞ': 'nya',
        'ট': 'ta',
        'ঠ': 'tha',
        'ড': 'da',
        'ঢ': 'dha',
        'ণ': 'na',
        'ত': 'tha',
        'থ': 'tha',
        'দ': 'da',
        'ধ': 'dha',
        'ন': 'na',
        'প': 'pa',
        'ফ': 'pha',
        'ব': 'ba',
        'ভ': 'bha',
        'ম': 'ma',
        'য': 'ya',
        'র': 'ra',
        'ল': 'la',
        'ব': 'va',
        'শ': 'sha',
        'ষ': 'sha',
        'স': 'sa',
        'হ': 'ha',
        'ড়': 'da',
        'ঢ়': 'dha',
        'য়': 'ya',
    }
    
    # Gujarati to Phonetic
    GUJARATI_TO_PHONETIC = {
        'અ': 'a',
        'આ': 'aa',
        'િ': 'i',
        'ી': 'ee',
        'ુ': 'u',
        'ૂ': 'oo',
        'ૃ': 'ri',
        'ે': 'e',
        'ૈ': 'ai',
        'ો': 'o',
        'ૌ': 'au',
        'ક': 'ka',
        'ખ': 'kha',
        'ગ': 'ga',
        'ઘ': 'gha',
        'ઙ': 'nga',
        'ચ': 'cha',
        'છ': 'chha',
        'જ': 'ja',
        'ઝ': 'jha',
        'ઞ': 'nya',
        'ટ': 'ta',
        'ઠ': 'tha',
        'ડ': 'da',
        'ઢ': 'dha',
        'ણ': 'na',
        'ત': 'tha',
        'થ': 'tha',
        'દ': 'da',
        'ધ': 'dha',
        'ન': 'na',
        'પ': 'pa',
        'ફ': 'pha',
        'બ': 'ba',
        'ભ': 'bha',
        'મ': 'ma',
        'ય': 'ya',
        'ર': 'ra',
        'લ': 'la',
        'ળ': 'la',
        'વ': 'va',
        'શ': 'sha',
        'ષ': 'sha',
        'સ': 'sa',
        'હ': 'ha',
    }
    
    # Punjabi to Phonetic
    PUNJABI_TO_PHONETIC = {
        'ਅ': 'a',
        'ਆ': 'aa',
        'ਇ': 'i',
        'ਈ': 'ee',
        'ਉ': 'u',
        'ਊ': 'oo',
        'ਕ': 'ka',
        'ਖ': 'kha',
        'ਗ': 'ga',
        'ਘ': 'gha',
        'ਙ': 'nga',
        'ਚ': 'cha',
        'ਛ': 'chha',
        'ਜ': 'ja',
        'ਝ': 'jha',
        'ਞ': 'nya',
        'ਟ': 'ta',
        'ਠ': 'tha',
        'ਡ': 'da',
        'ਢ': 'dha',
        'ਣ': 'na',
        'ਤ': 'ta',
        'ਥ': 'tha',
        'ਦ': 'da',
        'ਧ': 'dha',
        'ਨ': 'na',
        'ਪ': 'pa',
        'ਫ': 'pha',
        'ਬ': 'ba',
        'ਭ': 'bha',
        'ਮ': 'ma',
        'ਯ': 'ya',
        'ਰ': 'ra',
        'ਲ': 'la',
        'ਵ': 'va',
        'ਸ': 'sa',
        'ਹ': 'ha',
        '਼': '',
        'ਾ': 'aa',
        'ਿ': 'i',
        'ੀ': 'ee',
        'ੁ': 'u',
        'ੂ': 'oo',
        'ੇ': 'e',
        'ੈ': 'ai',
        'ੋ': 'o',
        'ੌ': 'au',
    }
    
    # Odia to Phonetic
    ODIA_TO_PHONETIC = {
        'ଅ': 'a',
        'ଆ': 'aa',
        'ଇ': 'i',
        'ଈ': 'ee',
        'ଉ': 'u',
        'ଊ': 'oo',
        'ଋ': 'ri',
        'ଏ': 'e',
        'ଐ': 'ai',
        'ଓ': 'o',
        'ଔ': 'au',
        'କ': 'ka',
        'ଖ': 'kha',
        'ଗ': 'ga',
        'ଘ': 'gha',
        'ଙ': 'nga',
        'ଚ': 'cha',
        'ଛ': 'chha',
        'ଜ': 'ja',
        'ଝ': 'jha',
        'ଞ': 'nya',
        'ଟ': 'ta',
        'ଠ': 'tha',
        'ଡ': 'da',
        'ଢ': 'dha',
        'ଣ': 'na',
        'ତ': 'tha',
        'ଥ': 'tha',
        'ଦ': 'da',
        'ଧ': 'dha',
        'ନ': 'na',
        'ପ': 'pa',
        'ଫ': 'pha',
        'ବ': 'ba',
        'ଭ': 'bha',
        'ମ': 'ma',
        'ଯ': 'ya',
        'ର': 'ra',
        'ଲ': 'la',
        'ଵ': 'va',
        'ଶ': 'sha',
        'ଷ': 'sha',
        'ସ': 'sa',
        'ହ': 'ha',
    }
    
    # Assamese to Phonetic
    ASSAMESE_TO_PHONETIC = {
        'অ': 'a',
        'আ': 'aa',
        'ই': 'i',
        'ঈ': 'ee',
        'উ': 'u',
        'ঊ': 'oo',
        'ৃ': 'ri',
        'এ': 'e',
        'ঐ': 'ai',
        'ও': 'o',
        'ঔ': 'au',
        'ক': 'ka',
        'খ': 'kha',
        'গ': 'ga',
        'ঘ': 'gha',
        'ঙ': 'nga',
        'চ': 'cha',
        'ছ': 'chha',
        'জ': 'ja',
        'ঝ': 'jha',
        'ঞ': 'nya',
        'ট': 'ta',
        'ঠ': 'tha',
        'ড': 'da',
        'ঢ': 'dha',
        'ণ': 'na',
        'ত': 'tha',
        'থ': 'tha',
        'দ': 'da',
        'ধ': 'dha',
        'ন': 'na',
        'প': 'pa',
        'ফ': 'pha',
        'ব': 'ba',
        'ভ': 'bha',
        'ম': 'ma',
        'য': 'ya',
        'র': 'ra',
        'ল': 'la',
        'ব': 'va',
        'শ': 'sha',
        'ষ': 'sha',
        'স': 'sa',
        'হ': 'ha',
        'ড়': 'da',
        'ঢ়': 'dha',
        'য়': 'ya',
    }
    
    # Sinhala to Phonetic
    SINHALA_TO_PHONETIC = {
        'අ': 'a',
        'ඉ': 'i',
        'උ': 'u',
        'ඔ': 'o',
        'එ': 'e',
        'ඒ': 'e',
        'ඨ': 'th',
        'ට': 't',
        'ඦ': 'j',
        'ඩ': 'd',
        'ඬ': 'd',
        'ත': 'n',
        'ථ': 'n',
        'ද': 'n',
        'ප': 'p',
        'ෆ': 'f',
        'බ': 'b',
        'ම': 'm',
        'ඹ': 'y',
        'ර': 'r',
        'ල': 'l',
        'ව': 'v',
        'ශ': 'sh',
        'ෂ': 'sh',
        'ස': 's',
        'හ': 'h',
        'ළ': 'l',
        'ෙ': 'e',
        'ි': 'i',
        'ු': 'u',
        'ූ': 'u',
        'ෝ': 'o',
        'ෙ': 'e',
        'ක': 'k',
        'ඛ': 'kh',
        'ග': 'g',
        'ඝ': 'gh',
        'ང': 'ng',
        'চ': 'ch',
        'ඡ': 'ch',
        'ජ': 'j',
        'ඣ': 'jh',
        'ඥ': 'jn',
    }
    
    LANGUAGE_MAPS = {
        'hi': HINDI_TO_PHONETIC,
        'mr': MARATHI_TO_PHONETIC,
        'ta': TAMIL_TO_PHONETIC,
        'te': TELUGU_TO_PHONETIC,
        'kn': KANNADA_TO_PHONETIC,
        'ml': MALAYALAM_TO_PHONETIC,
        'bn': BENGALI_TO_PHONETIC,
        'gu': GUJARATI_TO_PHONETIC,
        'pa': PUNJABI_TO_PHONETIC,
        'or': ODIA_TO_PHONETIC,
        'as': ASSAMESE_TO_PHONETIC,
        'si': SINHALA_TO_PHONETIC,
        'kok': MARATHI_TO_PHONETIC,  # Konkani uses similar script
        'bho': HINDI_TO_PHONETIC,     # Bhojpuri uses Devanagari
        'mai': HINDI_TO_PHONETIC,     # Maithili uses Devanagari
        'doi': HINDI_TO_PHONETIC,     # Dogri uses Devanagari
        'awa': HINDI_TO_PHONETIC,     # Awadhi uses Devanagari
        'mwr': HINDI_TO_PHONETIC,     # Marwadi uses Devanagari
        'bgc': HINDI_TO_PHONETIC,     # Haryanvi uses Devanagari
    }
    
    @classmethod
    def convert_to_phonetic(cls, text: str, lang: str) -> str:
        """Convert native script text to phonetic romanization."""
        lang_base = lang.split('-')[0]
        
        # English and already romanized - return as-is
        if lang_base == 'en':
            return text
        
        # Get language-specific character map
        char_map = cls.LANGUAGE_MAPS.get(lang_base)
        if not char_map:
            return text  # Fallback to original if language not supported
        
        result = []
        i = 0
        while i < len(text):
            # Try to match multi-character sequences first
            found = False
            for length in range(min(3, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in char_map:
                    result.append(char_map[substr])
                    i += length
                    found = True
                    break
            
            if not found:
                # Keep non-mapped characters as-is (spaces, punctuation, etc.)
                result.append(text[i])
                i += 1
        
        # Clean up: remove extra spaces, combine vowel sounds
        phonetic = ''.join(result)
        
        # Replace native script sentence boundaries with natural pauses
        phonetic = phonetic.replace('।', ', ')
        phonetic = phonetic.replace('॥', '.')
        
        phonetic = re.sub(r'\s+', ' ', phonetic)  # Normalize spaces
        phonetic = re.sub(r'([aeiou])a+', r'\1a', phonetic)  # Prevent double vowels
        
        # Smoothen transitions and fix jagged pronounciation patterns
        phonetic = re.sub(r'([bcdgjklmnpqrstvwxyz])\1+', r'\1', phonetic) # Remove hard duplicate consonants
        phonetic = re.sub(r'([aeiou])\1{2,}', r'\1\1', phonetic) # Reduce long vowel sounds
        
        # Add slight conversational pauses to make it sound more human
        # We replace certain common breath breaks with small commas if not present
        phonetic = phonetic.replace(' ani ', ', ani ').replace(' aani ', ', aani ')
        phonetic = phonetic.replace(' pan ', ', pan ').replace(' ya ', ', ya ')
        phonetic = phonetic.replace('hhn', 'hn').replace('nng', 'ng')
        
        # Cleanup any double commas added
        phonetic = phonetic.replace(', ,', ',')
        
        return phonetic.strip()
    
    @classmethod
    def should_use_phonetic(cls, lang: str) -> bool:
        """Check if language should use phonetic conversion."""
        lang_base = lang.split('-')[0]
        return lang_base in cls.LANGUAGE_MAPS and lang_base != 'en'

# Export for use in TTS
__all__ = ['PhoneticConverter']
