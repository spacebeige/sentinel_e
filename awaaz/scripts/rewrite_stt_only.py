import re

with open('awaaz_voice_backup.py', 'r') as f:
    text = f.read()

# We want to remove the classes we don't need.
classes_to_remove = [
    r'class LangMetaResolver:.*?(?=^class |^def |^# ───)',
    r'class ProfileDetector:.*?(?=^class |^def |^# ───)',
    r'class ModelProcessor:.*?(?=^class |^def |^# ───)',
    r'class TTSProcessor:.*?(?=^class |^def |^# ───)',
    r'class TransliterationHook:.*?(?=^class |^def |^# ───)',
    r'def get_lang_meta\(.*?(?=^class |^def |^# ───)',
    r'def get_pace\(.*?(?=^class |^def |^# ───)',
    r'def check_emergency\(.*?(?=^class |^def |^# ───)',
]

for pattern in classes_to_remove:
    text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)

# Now we need to overwrite AWAAZVoiceLoop to only do STT. 
# It's easier to just match from "class AWAAZVoiceLoop:" to the end and replace it.

new_loop = """class AWAAZVoiceLoop:
    \"\"\"
    Main loop for STT-only AWAAZ prototype.
    Captures audio -> runs VAD -> Detects Language -> Transcribes -> Outputs text.
    \"\"\"
    def __init__(self, mode: str = "mic", input_file: Optional[str] = None):
        self.mode = mode
        self.session_id = str(uuid.uuid4())
        self.audio_input = AudioInput(mode=mode, input_file=input_file)
        self.vad = VADProcessor()
        self.stt = STTProcessor()
        self.lid = TokenLevelLangDetector.get()
        self.profile = CallerProfile(session_id=self.session_id)
        
        print("Loading models for STT-only pipeline...")
        self.vad.load()
        self.stt.load()
        print("Models loaded successfully.")

    def run(self):
        print(f"[{self.session_id}] AWAAZ STT-only pipeline started in {self.mode} mode.")
        print("Listening for speech...\\n")
        
        try:
            while self.profile.turn_number < MAX_TURNS:
                audio_path = self.audio_input.record_utterance(self.session_id)
                if not audio_path:
                    # End of file or Asterisk stream dropped
                    break
                    
                self.profile.turn_number += 1
                print(f"\\n--- Turn {self.profile.turn_number} ---")
                
                # STEP 2: Detect Language (first turn only)
                if not self.profile.lang:
                    print("Detecting primary language...")
                    lang, conf = self.stt.detect_language(audio_path)
                    self.profile.lang = lang
                    self.profile.confidence = conf
                    print(f"Detected: {lang} (confidence: {conf:.2f})")
                
                # STEP 3: Transcribe
                print("Transcribing...")
                text = self.stt.transcribe(audio_path, self.profile.lang)
                print(f"Transcript: {text.strip()}")
                
                if not text.strip():
                    print("No speech detected in this turn.")
                    continue
                    
                # STEP 4: Token-level Language Detection (Mixed Language identification)
                self.lid.update_profile(text, self.profile)
                
                print(f"Language Mode: {self.profile.lang_mode}")
                if self.profile.lang_distribution:
                    print(f"Distribution: {self.profile.lang_distribution}")
                
        except KeyboardInterrupt:
            print("\\nPipeline interrupted by user.")
        finally:
            print("\\n[Pipeline finished]")

def main():
    parser = argparse.ArgumentParser(description="AWAAZ STT-Only Pipeline")
    parser.add_argument("--mode", choices=["mic", "file", "asterisk"], default="mic")
    parser.add_argument("--input", help="Path to WAV file for file mode")
    args = parser.parse_args()

    if args.mode == "file" and not args.input:
        print("ERROR: --input required for file mode")
        sys.exit(1)

    loop = AWAAZVoiceLoop(mode=args.mode, input_file=args.input)
    loop.run()

if __name__ == "__main__":
    main()
"""

text = re.sub(r'class AWAAZVoiceLoop:.*', new_loop, text, flags=re.MULTILINE | re.DOTALL)

with open('awaaz_voice.py', 'w') as f:
    f.write(text)
