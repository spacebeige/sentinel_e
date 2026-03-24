#!/usr/bin/env python3
"""
COMPREHENSIVE LANGUAGE DETECTION AUDIT
Self-audit script to check all languages, scripts, and regional variants
for proper detection, marker coverage, and script-based routing.
"""

import sys
import os

# Get root directory and configure paths
root_dir = os.path.dirname(os.path.abspath(__file__))
awaaz_dir = os.path.join(root_dir, 'awaaz')

# Add paths
sys.path.insert(0, awaaz_dir)
sys.path.insert(0, root_dir)

# Now imports should work
from src.pipeline.lang_detect import TokenLevelLangDetector
from src.pipeline.nlp import LANGUAGE_CONFIG

def audit_language_coverage():
    """Audit all configured languages for proper markers and detection."""
    print("\n" + "="*80)
    print("COMPREHENSIVE LANGUAGE DETECTION AUDIT")
    print("="*80)
    
    detector = TokenLevelLangDetector.get()
    markers = detector._LANGUAGE_MARKERS
    
    print("\n📋 AUDIT 1: Language Markers Coverage")
    print("-" * 80)
    
    # Get all languages from config
    all_langs = set(LANGUAGE_CONFIG.keys())
    # Remove mixed variants (e.g., "hi-en")
    core_langs = {lang for lang in all_langs if not lang.endswith("-en") and lang != "en"}
    
    marked_langs = set(markers.keys())
    missing_langs = core_langs - marked_langs
    
    print(f"\nTotal configured languages: {len(core_langs)}")
    print(f"Languages with markers: {len(marked_langs)}")
    print(f"Missing markers: {len(missing_langs)}")
    
    if missing_langs:
        print(f"\n❌ MISSING LANGUAGE MARKERS:")
        for lang in sorted(missing_langs):
            config = LANGUAGE_CONFIG.get(lang, {})
            print(f"  - {lang:6} ({config.get('name', 'Unknown'):20}) Script: {config.get('script', 'Unknown')}")
    
    # Audit marker quality (count)
    print(f"\n📊 MARKER QUALITY CHECK:")
    print("-" * 80)
    weak_markers = []
    for lang in marked_langs:
        native_count = len(markers[lang].get("native", []))
        latin_count = len(markers[lang].get("latin", []))
        total = native_count + latin_count
        
        if total < 5:
            weak_markers.append((lang, native_count, latin_count))
        
        status = "✅" if total >= 8 else "⚠️" if total >= 5 else "❌"
        print(f"{status} {lang:6} - Native: {native_count:2} | Latin: {latin_count:2} | Total: {total}")
    
    if weak_markers:
        print(f"\n⚠️ WEAK MARKERS (< 5 total):")
        for lang, native, latin in weak_markers:
            print(f"  - {lang}: {native} native + {latin} latin = {native + latin} total")

def audit_devanagari_differentiation():
    """Audit differentiation of all Devanagari-based languages."""
    print("\n" + "="*80)
    print("AUDIT 2: Devanagari Language Differentiation")
    print("="*80)
    
    detector = TokenLevelLangDetector.get()
    markers = detector._LANGUAGE_MARKERS
    
    # All Devanagari languages
    devanagari_langs = {}
    for lang_code, config in LANGUAGE_CONFIG.items():
        if config.get("script") == "Devanagari" and not lang_code.endswith("-en"):
            devanagari_langs[lang_code] = config.get("name")
    
    print(f"\n📍 Total Devanagari-based languages: {len(devanagari_langs)}")
    print("\nLanguage | Name             | Unique Markers | Native | Latin | Status")
    print("-" * 80)
    
    issues = []
    for lang in sorted(devanagari_langs.keys()):
        name = devanagari_langs[lang]
        
        if lang not in markers:
            print(f"{lang:8} | {name:16} | ❌ NO MARKERS   | -      | -     | MISSING")
            issues.append((lang, "NO MARKERS DEFINED"))
            continue
        
        native_markers = markers[lang].get("native", [])
        latin_markers = markers[lang].get("latin", [])
        
        # Check for uniqueness
        all_markers = set(native_markers + latin_markers)
        duplicate_count = len(native_markers) + len(latin_markers) - len(all_markers)
        
        status = "❌ DUPLICATES" if duplicate_count > 0 else "✅ UNIQUE" if len(all_markers) >= 8 else "⚠️ WEAK"
        
        print(f"{lang:8} | {name:16} | {len(all_markers):14} | {len(native_markers):6} | {len(latin_markers):5} | {status}")
        
        if duplicate_count > 0:
            issues.append((lang, f"Has {duplicate_count} duplicate markers"))

    if issues:
        print(f"\n❌ DEVANAGARI DIFFERENTIATION ISSUES:")
        for lang, issue in issues:
            print(f"  - {lang}: {issue}")

def audit_script_detection():
    """Audit script-based detection routing."""
    print("\n" + "="*80)
    print("AUDIT 3: Script-Based Detection Routing")
    print("="*80)
    
    script_mappings = {
        "Devanagari": {"range": (0x0900, 0x097F), "langs": []},
        "Gujarati": {"range": (0x0A80, 0x0AFF), "langs": []},
        "Gurmukhi": {"range": (0x0A00, 0x0A7F), "langs": []},
        "Bengali": {"range": (0x0980, 0x09FF), "langs": []},
        "Odia": {"range": (0x0B00, 0x0B7F), "langs": []},
        "Tamil": {"range": (0x0B80, 0x0BFF), "langs": []},
        "Telugu": {"range": (0x0C00, 0x0C7F), "langs": []},
        "Kannada": {"range": (0x0C80, 0x0CFF), "langs": []},
        "Malayalam": {"range": (0x0D00, 0x0D7F), "langs": []},
        "Ol Chiki": {"range": (0x1C50, 0x1C7F), "langs": []},
        "Arabic": {"range": (0x0600, 0x06FF), "langs": []},
    }
    
    # Map all languages to scripts
    for lang_code, config in LANGUAGE_CONFIG.items():
        if lang_code.endswith("-en") or lang_code == "en":
            continue
        script = config.get("script", "").split(" ")[0]  # Get base script
        if script in script_mappings:
            script_mappings[script]["langs"].append((lang_code, config.get("name")))
    
    print("\nScript Detection Analysis:")
    print("-" * 80)
    
    for script, data in sorted(script_mappings.items()):
        if not data["langs"]:
            print(f"\n{script:15} (0x{data['range'][0]:04X}-0x{data['range'][1]:04X})")
            print(f"  ⚠️ No languages mapped")
            continue
        
        print(f"\n{script:15} (0x{data['range'][0]:04X}-0x{data['range'][1]:04X})")
        print(f"  Languages: {len(data['langs'])}")
        for lang, name in data["langs"]:
            print(f"    - {lang:6} ({name})")
    
    # Check for detection logic in detect() method
    print("\n" + "-" * 80)
    print("Script Priority in detect() method:")
    print("-" * 80)
    
    detector = TokenLevelLangDetector.get()
    
    # Test detection with sample texts from each script
    test_samples = {
        "pa": ("ਹੋ ਤੇਰੇ ਨਾਲ", "Punjabi - Gurmukhi"),
        "gu": ("સલામ, તમે", "Gujarati"),
        "hi": ("नमस्ते, कैसे", "Hindi - Devanagari"),
        "mr": ("आहे, मला", "Marathi - Devanagari"),
        "sa": ("नमस्ते स्वागतम्", "Sanskrit - Devanagari"),
        "ta": ("வணக்கம், எப்படி", "Tamil"),
        "te": ("హలో, ఎలా", "Telugu"),
        "kn": ("ಹಲೋ, ಹೇಗೆ", "Kannada"),
        "ml": ("ഹലോ, എങ്ങനെ", "Malayalam"),
        "bn": ("নমস্কার, কেমন", "Bengali"),
        "ur": ("السلام علیکم", "Urdu - Arabic"),
    }
    
    print(f"\n{'Lang':6} | {'Sample Text':25} | Detected | Expected | Status")
    print("-" * 80)
    
    detection_issues = []
    for expected_lang, (text, description) in test_samples.items():
        detected_lang, _ = detector.detect(text)
        status = "✅" if detected_lang == expected_lang else "❌"
        print(f"{expected_lang:6} | {text:25} | {detected_lang:8} | {expected_lang:8} | {status}")
        
        if detected_lang != expected_lang:
            detection_issues.append((expected_lang, detected_lang, description))
    
    if detection_issues:
        print(f"\n❌ DETECTION MISMATCHES:")
        for expected, detected, desc in detection_issues:
            print(f"  - {desc}: expected {expected}, got {detected}")

def audit_missing_regional_languages():
    """Audit coverage of regional/minority languages."""
    print("\n" + "="*80)
    print("AUDIT 4: Regional & Minority Language Coverage")
    print("="*80)
    
    detector = TokenLevelLangDetector.get()
    markers = detector._LANGUAGE_MARKERS
    
    regional_langs = {
        "brx": "Bodo",
        "mni": "Manipuri/Meitei",
        "tcy": "Tulu",
        "sat": "Santali",
        "kru": "Kurukh",
        "mah": "Magahi",
        "lmn": "Lambadi",
        "dcc": "Dakhini Urdu",
        "saz": "Saurashtra",
    }
    
    print(f"\nRegional language marker audit:")
    print("-" * 80)
    
    coverage_status = {"✅ Full": 0, "⚠️ Partial": 0, "❌ Missing": 0}
    
    for lang_code, name in sorted(regional_langs.items()):
        if lang_code not in markers:
            status = "❌ Missing"
            coverage_status["❌ Missing"] += 1
            markers_count = 0
        else:
            native = len(markers[lang_code].get("native", []))
            latin = len(markers[lang_code].get("latin", []))
            total = native + latin
            
            if total >= 8:
                status = "✅ Full"
                coverage_status["✅ Full"] += 1
            elif total >= 4:
                status = "⚠️ Partial"
                coverage_status["⚠️ Partial"] += 1
            else:
                status = "❌ Missing"
                coverage_status["❌ Missing"] += 1
            
            markers_count = total
        
        script = LANGUAGE_CONFIG.get(lang_code, {}).get("script", "Unknown")
        print(f"{status} | {lang_code:6} | {name:20} | Script: {script:15} | Markers: {markers_count}")
    
    print(f"\nCoverage Summary:")
    for status, count in sorted(coverage_status.items()):
        print(f"  {status}: {count}")

def audit_script_overlaps():
    """Check for potential script range overlaps that could cause confusion."""
    print("\n" + "="*80)
    print("AUDIT 5: Script Range Overlap Analysis")
    print("="*80)
    
    script_ranges = {
        "Devanagari": (0x0900, 0x097F),
        "Bengali": (0x0980, 0x09FF),
        "Odia": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Gurmukhi": (0x0A00, 0x0A7F),
        "Arabic": (0x0600, 0x06FF),
        "Ol Chiki": (0x1C50, 0x1C7F),
    }
    
    print("\nChecking for overlaps...")
    overlaps_found = False
    
    scripts = sorted(script_ranges.items())
    for i in range(len(scripts)):
        for j in range(i + 1, len(scripts)):
            script1, (start1, end1) = scripts[i]
            script2, (start2, end2) = scripts[j]
            
            # Check if ranges overlap
            if not (end1 < start2 or end2 < start1):
                print(f"⚠️ OVERLAP: {script1} (0x{start1:04X}-0x{end1:04X}) "
                      f"overlaps with {script2} (0x{start2:04X}-0x{end2:04X})")
                overlaps_found = True
    
    if not overlaps_found:
        print("✅ No script range overlaps detected")
    
    print("\nScript Range Continuity:")
    ranges_list = sorted(script_ranges.values())
    for (start, end) in ranges_list:
        print(f"  0x{start:04X} - 0x{end:04X}")

def main():
    """Run all audits."""
    try:
        audit_language_coverage()
        audit_devanagari_differentiation()
        audit_script_detection()
        audit_missing_regional_languages()
        audit_script_overlaps()
        
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during audit: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
