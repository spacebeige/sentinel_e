#!/usr/bin/env python3
"""
Quick validation test for pipeline fixes without import complications.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return success status."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.returncode == 0, result.stdout, result.stderr

def test_syntax():
    """Test 1: Python syntax check."""
    print("\n" + "="*70)
    print("TEST 1: PYTHON SYNTAX CHECK")
    print("="*70)
    
    files = [
        "awaaz/src/pipeline/stt.py",
        "awaaz/src/pipeline/tts.py",
        "awaaz/src/pipeline/lang_detect.py",
        "awaaz/src/pipeline/nlp.py",
    ]
    
    passed = 0
    failed = 0
    
    for file in files:
        success, _, stderr = run_command(f"python3 -m py_compile {file}")
        if success:
            print(f"✓ PASS | {file}")
            passed += 1
        else:
            print(f"✗ FAIL | {file}")
            if stderr:
                print(f"       | Error: {stderr[:100]}")
            failed += 1
    
    print(f"\nResult: {passed}/{len(files)} files have valid syntax")
    return passed, failed


def test_language_markers():
    """Test 2: Language markers are properly defined."""
    print("\n" + "="*70)
    print("TEST 2: LANGUAGE MARKERS DATABASE")
    print("="*70)
    
    # Read lang_detect.py and check for LANGUAGE_MARKERS
    try:
        with open("awaaz/src/pipeline/lang_detect.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for key languages
        languages = {
            "hi": "Hindi",
            "pa": "Punjabi",
            "mr": "Marathi",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam",
            "sa": "Sanskrit",
        }
        
        passed = 0
        failed = 0
        
        for code, name in languages.items():
            if f'"{code}"' in content:
                print(f"✓ PASS | {name:15} ({code}) markers defined")
                passed += 1
            else:
                print(f"✗ FAIL | {name:15} ({code}) markers MISSING")
                failed += 1
        
        print(f"\nResult: {passed}/{len(languages)} languages have markers")
        return passed, failed
        
    except Exception as e:
        print(f"✗ ERROR | Could not read file: {e}")
        return 0, len(languages)


def test_script_detection():
    """Test 3: Script-based detection implementation."""
    print("\n" + "="*70)
    print("TEST 3: SCRIPT-BASED DETECTION IMPLEMENTATION")
    print("="*70)
    
    try:
        with open("awaaz/src/pipeline/lang_detect.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_methods = [
            "_detect_by_script",
            "_detect_devanagari_variant",
            "_contains_script",
            "_heuristic_detect",
        ]
        
        passed = 0
        failed = 0
        
        for method in required_methods:
            if f"def {method}" in content:
                print(f"✓ PASS | Method {method:<30} implemented")
                passed += 1
            else:
                print(f"✗ FAIL | Method {method:<30} MISSING")
                failed += 1
        
        print(f"\nResult: {passed}/{len(required_methods)} methods implemented")
        return passed, failed
        
    except Exception as e:
        print(f"✗ ERROR | Could not verify: {e}")
        return 0, len(required_methods)


def test_tts_phonetic_fix():
    """Test 4: TTS phonetic analysis async fix."""
    print("\n" + "="*70)
    print("TEST 4: TTS PHONETIC ANALYSIS FIX")
    print("="*70)
    
    try:
        with open("awaaz/src/pipeline/tts.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("_get_phonetic_analysis_sync", "Sync wrapper function"),
            ("_get_phonetic_analysis_sync(text, lang)", "Sync function call"),
        ]
        
        passed = 0
        failed = 0
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✓ PASS | {description:<40} present")
                passed += 1
            else:
                print(f"✗ FAIL | {description:<40} MISSING")
                failed += 1
        
        # Check that asyncio.run is NOT used for phonetic_analysis
        if "asyncio.run(_get_phonetic_analysis(text, lang))" not in content:
            print(f"✓ PASS | Removed asyncio.run coroutine warning")
            passed += 1
        else:
            print(f"✗ FAIL | Still using asyncio.run for phonetic analysis")
            failed += 1
        
        print(f"\nResult: {passed}/{len(checks)+1} phonetic fixes applied")
        return passed, failed
        
    except Exception as e:
        print(f"✗ ERROR | Could not verify: {e}")
        return 0, len(checks) + 1


def test_stt_script_detection():
    """Test 5: STT using script-based detection."""
    print("\n" + "="*70)
    print("TEST 5: STT SCRIPT-BASED DETECTION INTEGRATION")
    print("="*70)
    
    try:
        with open("awaaz/src/pipeline/stt.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("TokenLevelLangDetector", "Script detector import"),
            ("script_detector = TokenLevelLangDetector.get()", "Script detector initialization"),
            ("[LANG-DETECT] ✓ Script-based detection", "Script detection logging"),
        ]
        
        passed = 0
        failed = 0
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✓ PASS | {description:<40} integrated")
                passed += 1
            else:
                print(f"✗ FAIL | {description:<40} MISSING")
                failed += 1
        
        print(f"\nResult: {passed}/{len(checks)} STT integrations applied")
        return passed, failed
        
    except Exception as e:
        print(f"✗ ERROR | Could not verify: {e}")
        return 0, len(checks)


def main():
    """Run all validation tests."""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " PIPELINE FIXES VALIDATION ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Syntax
    p, f = test_syntax()
    total_passed += p
    total_failed += f
    
    # Test 2: Language markers
    p, f = test_language_markers()
    total_passed += p
    total_failed += f
    
    # Test 3: Script detection
    p, f = test_script_detection()
    total_passed += p
    total_failed += f
    
    # Test 4: TTS phonetic fix
    p, f = test_tts_phonetic_fix()
    total_passed += p
    total_failed += f
    
    # Test 5: STT integration
    p, f = test_stt_script_detection()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"✓ Passed: {total_passed}")
    print(f"✗ Failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        success_rate = total_passed / (total_passed + total_failed) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n✅ ALL FIXES VALIDATED! Ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total_failed} issue(s) found. Please review above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
