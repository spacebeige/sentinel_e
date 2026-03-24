#!/usr/bin/env python3
"""
Generate a test audio file with speech for pipeline testing
Uses text-to-speech to create a simple test audio
"""

import os
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
except ImportError:
    print("Installing audio generation dependencies...")
    os.system("pip install -q scipy numpy")
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile

# Generate simple sine wave test audio (440 Hz dial tone)
sample_rate = 16000
duration = 2  # seconds
frequency = 440  # Hz (A note)

t = np.linspace(0, duration, int(sample_rate * duration))
sine_wave = 0.3 * np.sin(2 * np.pi * frequency * t)

# Add envelope to avoid clicks
envelope = np.ones_like(sine_wave)
envelope[:sample_rate // 10] *= np.linspace(0, 1, sample_rate // 10)  # fade in
envelope[-sample_rate // 10:] *= np.linspace(1, 0, sample_rate // 10)  # fade out
sine_wave = sine_wave * envelope

# Convert to int16
audio = np.int16(sine_wave * 32767)

# Save
output_file = Path(__file__).parent / "test_audio.wav"
wavfile.write(output_file, sample_rate, audio)

print(f"✓ Test audio file created: {output_file}")
print(f"  Duration: {duration}s @ {sample_rate}Hz")
print(f"  Frequency: {frequency} Hz (dial tone)")
