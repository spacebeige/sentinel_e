import os
import sys
import time
import wave
import struct
import argparse
import tempfile
import warnings
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile as scipy_wavfile


# ══════════════════════════════════════════════════════════════════
# CONSTANTS — all tunable via CLI args
# ══════════════════════════════════════════════════════════════════

TARGET_SR          = 16000    # Hz — optimal for Whisper / faster-whisper
CALIBRATION_SECS   = 1.5      # seconds of silence to sample for noise floor
SILENCE_THRESHOLD  = 700      # ms of silence = end of utterance (VAD)
MAX_DURATION_SECS  = 30       # hard cutoff
MIN_SPEECH_SECS    = 0.5      # reject if less speech than this
CHUNK_MS           = 30       # ms per VAD frame (webrtcvad requires 10/20/30)
MIN_SNR_DB         = 6.0      # reject recording if SNR below this after denoising
ENERGY_THRESHOLD   = 0.005    # RMS threshold for energy-based VAD fallback


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """Microphone calibration measurements."""
    device_name:       str
    device_index:      int
    sample_rate:       int
    channels:          int
    noise_floor_rms:   float    # RMS of ambient noise
    noise_floor_db:    float    # dB equivalent
    recommended_gain:  float    # gain multiplier to apply
    clipping_risk:     bool     # True if gain would cause clipping
    quality_label:     str      # EXCELLENT / GOOD / FAIR / POOR


@dataclass
class RecordingResult:
    """Full result from one recording session."""
    path:              str
    duration_s:        float
    sample_rate:       int
    word_count_est:    int      # estimated from speech duration / avg word rate
    snr_before_db:     float
    snr_after_db:      float
    noise_reduced:     bool
    vad_method:        str      # webrtcvad / silero / energy
    speech_frames:     int
    silence_frames:    int
    calibration:       CalibrationResult
    quality_label:     str      # EXCELLENT / GOOD / FAIR / POOR / REJECTED
    rejection_reason:  str      # set if quality_label == REJECTED
    raw_path:          str      # path to pre-denoised WAV (temp, deleted after)


# ══════════════════════════════════════════════════════════════════
# MODULE 1 — DEVICE MANAGER
# ══════════════════════════════════════════════════════════════════

class DeviceManager:
    """Lists and selects microphone devices."""

    def list_devices(self) -> list:
        devices = []
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    devices.append({
                        "index":    i,
                        "name":     d["name"],
                        "channels": d["max_input_channels"],
                        "sr":       int(d["default_samplerate"]),
                        "backend":  "sounddevice",
                    })
        except ImportError:
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                for i in range(pa.get_device_count()):
                    d = pa.get_device_info_by_index(i)
                    if d["maxInputChannels"] > 0:
                        devices.append({
                            "index":    i,
                            "name":     d["name"],
                            "channels": int(d["maxInputChannels"]),
                            "sr":       int(d["defaultSampleRate"]),
                            "backend":  "pyaudio",
                        })
                pa.terminate()
            except ImportError:
                pass
        return devices

    def print_devices(self):
        devices = self.list_devices()
        if not devices:
            print("No input devices found.")
            return
        print("\nAvailable microphones:")
        print(f"  {'IDX':>4}  {'NAME':<40}  {'CH':>3}  {'SR':>6}  BACKEND")
        print("  " + "-"*65)
        for d in devices:
            print(f"  {d['index']:>4}  {d['name']:<40}  {d['channels']:>3}  "
                  f"{d['sr']:>6}  {d['backend']}")
        print()

    def get_default_input_index(self) -> Optional[int]:
        try:
            import sounddevice as sd
            return sd.default.device[0]
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — MIC CALIBRATOR
# Samples ambient noise, computes noise floor, recommends gain.
# ══════════════════════════════════════════════════════════════════

class MicCalibrator:

    def calibrate(self, device_index: Optional[int] = None,
                  duration_s: float = CALIBRATION_SECS) -> CalibrationResult:
        """
        Record silence to measure ambient noise floor.
        Prints a visual level meter during calibration.
        """
        print(f"[CAL] Calibrating microphone ({duration_s:.1f}s) — stay silent...")

        audio = self._record_raw(duration_s, device_index)
        if audio is None or len(audio) == 0:
            return self._default_calibration(device_index)

        # Compute noise metrics
        rms    = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        db     = 20 * np.log10(rms + 1e-10)
        peak   = float(np.max(np.abs(audio)))

        # Recommend gain: target speech RMS at -12dB (0.25 float)
        target_rms     = 0.25
        if rms > 1e-6:
            recommended = min(4.0, target_rms / (rms * 10))  # noise is ~10x quieter
        else:
            recommended = 1.0

        clipping_risk = (peak * recommended) > 0.95

        if db < -50:
            label = "EXCELLENT"
        elif db < -40:
            label = "GOOD"
        elif db < -30:
            label = "FAIR"
        else:
            label = "POOR"

        devices  = DeviceManager().list_devices()
        dev_name = next((d["name"] for d in devices
                         if d["index"] == device_index), "default")
        result   = CalibrationResult(
            device_name=dev_name,
            device_index=device_index or -1,
            sample_rate=TARGET_SR,
            channels=1,
            noise_floor_rms=round(rms, 6),
            noise_floor_db=round(db, 1),
            recommended_gain=round(recommended, 3),
            clipping_risk=clipping_risk,
            quality_label=label,
        )

        bar   = self._level_bar(db, -60, -20)
        print(f"[CAL] Noise floor: {db:.1f} dB  {bar}  [{label}]")
        if label == "POOR":
            print("[CAL] WARNING: High ambient noise detected. Move to a quieter location.")
        if clipping_risk:
            print("[CAL] WARNING: Clipping risk with recommended gain — mic sensitivity may be too high.")

        return result

    def _record_raw(self, duration_s: float,
                    device_index: Optional[int]) -> Optional[np.ndarray]:
        """Record raw audio for calibration purposes."""
        n_samples = int(duration_s * TARGET_SR)
        try:
            import sounddevice as sd
            audio = sd.rec(n_samples, samplerate=TARGET_SR, channels=1,
                           dtype="float32", device=device_index)
            sd.wait()
            return audio.flatten()
        except Exception:
            pass
        try:
            import pyaudio
            pa     = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paFloat32, channels=1,
                             rate=TARGET_SR, input=True,
                             input_device_index=device_index,
                             frames_per_buffer=512)
            frames = []
            for _ in range(int(TARGET_SR / 512 * duration_s)):
                data = stream.read(512, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return np.concatenate(frames)
        except Exception as e:
            print(f"[CAL] Recording failed: {e}")
            return None

    def _default_calibration(self, device_index) -> CalibrationResult:
        return CalibrationResult(
            device_name="unknown", device_index=device_index or -1,
            sample_rate=TARGET_SR, channels=1,
            noise_floor_rms=0.01, noise_floor_db=-40.0,
            recommended_gain=1.0, clipping_risk=False,
            quality_label="GOOD",
        )

    def _level_bar(self, db: float, min_db: float, max_db: float,
                   width: int = 20) -> str:
        fraction = (db - min_db) / (max_db - min_db)
        fraction = max(0.0, min(1.0, fraction))
        filled   = int(fraction * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — VAD ENGINE
# Three-layer voice activity detection:
#   1. webrtcvad (Google WebRTC, BSD licence, very fast)
#   2. Silero VAD (neural, MIT, better accuracy)
#   3. Energy fallback (no external deps)
# ══════════════════════════════════════════════════════════════════

class VADEngine:

    def __init__(self, aggressiveness: int = 2):
        self._webrtc   = None
        self._silero   = None
        self._method   = "energy"
        self._agg      = aggressiveness
        self._load()

    def _load(self):
        # Try webrtcvad first (fastest, lowest latency)
        try:
            import webrtcvad
            self._webrtc = webrtcvad.Vad(self._agg)
            self._method = "webrtcvad"
            print(f"[VAD] webrtcvad loaded (aggressiveness={self._agg})")
            return
        except ImportError:
            print("[VAD] webrtcvad not installed — trying silero...")

        # Try Silero
        try:
            import torch
            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload=False, onnx=False
            )
            self._silero = model
            self._method = "silero"
            print("[VAD] Silero VAD loaded")
            return
        except Exception:
            print("[VAD] Silero unavailable — using energy-based fallback")

    def is_speech_frame(self, frame: np.ndarray, sr: int = TARGET_SR) -> bool:
        """Check if a single audio frame contains speech."""
        if self._method == "webrtcvad":
            return self._webrtc_check(frame, sr)
        elif self._method == "silero":
            return self._silero_check(frame)
        else:
            return self._energy_check(frame)

    def _webrtc_check(self, frame: np.ndarray, sr: int) -> bool:
        try:
            import webrtcvad
            # webrtcvad needs 16-bit PCM bytes, 8/16/32 kHz, 10/20/30ms frames
            pcm = (frame * 32767).clip(-32768, 32767).astype(np.int16)
            return self._webrtc.is_speech(pcm.tobytes(), sr)
        except Exception:
            return self._energy_check(frame)

    def _silero_check(self, frame: np.ndarray) -> bool:
        try:
            import torch
            t = torch.FloatTensor(frame)
            conf = self._silero(t, TARGET_SR).item()
            return conf > 0.5
        except Exception:
            return self._energy_check(frame)

    def _energy_check(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return rms > ENERGY_THRESHOLD

    @property
    def method(self) -> str:
        return self._method


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — NOISE REDUCER
# Uses noisereduce if available (better quality).
# Falls back to our own scipy spectral subtraction.
# ══════════════════════════════════════════════════════════════════

class NoiseReducer:

    def __init__(self):
        self._has_nr = False
        try:
            import noisereduce
            self._has_nr = True
            print("[DENOISE] noisereduce library loaded (spectral gating)")
        except ImportError:
            print("[DENOISE] noisereduce not installed — using scipy fallback")
            print("[DENOISE] Install: pip install noisereduce")

    def reduce(self, audio: np.ndarray, sr: int,
               noise_clip: Optional[np.ndarray] = None,
               stationary: bool = False) -> np.ndarray:
        """
        Remove noise from audio.
        noise_clip: optional separate noise sample (silence clip from calibration)
        stationary: True = constant noise (fan, AC), False = variable noise (crowd, wind)
        """
        if self._has_nr:
            return self._noisereduce(audio, sr, noise_clip, stationary)
        return self._scipy_spectral_sub(audio, sr, noise_clip)

    def _noisereduce(self, audio: np.ndarray, sr: int,
                     noise_clip: Optional[np.ndarray],
                     stationary: bool) -> np.ndarray:
        import noisereduce as nr
        try:
            if noise_clip is not None:
                reduced = nr.reduce_noise(
                    y=audio, sr=sr,
                    y_noise=noise_clip,
                    stationary=stationary,
                    prop_decrease=0.85,
                )
            else:
                reduced = nr.reduce_noise(
                    y=audio, sr=sr,
                    stationary=stationary,
                    prop_decrease=0.85,
                )
            return reduced.astype(np.float32)
        except Exception as e:
            print(f"[DENOISE] noisereduce failed ({e}), using scipy fallback")
            return self._scipy_spectral_sub(audio, sr, noise_clip)

    def _scipy_spectral_sub(self, audio: np.ndarray, sr: int,
                             noise_clip: Optional[np.ndarray]) -> np.ndarray:
        """Spectral subtraction fallback using scipy."""
        audio = audio.astype(np.float32)
        peak  = np.max(np.abs(audio)) or 1.0
        audio = audio / peak

        n_fft    = 512
        hop      = 128
        window   = np.hanning(n_fft)

        def stft(x):
            n_fr = 1 + (len(x) - n_fft) // hop
            out  = np.zeros((n_fft // 2 + 1, n_fr), dtype=complex)
            for i in range(n_fr):
                s = i * hop
                f = x[s:s + n_fft]
                if len(f) < n_fft:
                    f = np.pad(f, (0, n_fft - len(f)))
                out[:, i] = np.fft.rfft(f * window)
            return out

        def istft(S, length):
            n_fr  = S.shape[1]
            out   = np.zeros(length + n_fft)
            norm  = np.zeros(length + n_fft)
            for i in range(n_fr):
                s = i * hop
                f = np.fft.irfft(S[:, i])
                out[s:s + n_fft] += f * window
                norm[s:s + n_fft] += window ** 2
            norm = np.where(norm < 1e-8, 1.0, norm)
            return (out / norm)[:length]

        sig_stft   = stft(audio)
        noise_src  = noise_clip if noise_clip is not None else audio[:int(sr * 0.5)]
        noise_stft = stft(noise_src.astype(np.float32) / peak)
        noise_prof = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        clean_mag  = np.maximum(np.abs(sig_stft) - 1.5 * noise_prof, 0)
        clean_stft = clean_mag * np.exp(1j * np.angle(sig_stft))
        clean      = istft(clean_stft, len(audio))
        cp         = np.max(np.abs(clean))
        if cp > 0:
            clean = clean / cp * peak
        return clean.astype(np.float32)

    def snr_db(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        noise_power  = np.mean((original - cleaned) ** 2)
        signal_power = np.mean(cleaned ** 2)
        if noise_power < 1e-10:
            return 60.0
        return float(10 * np.log10(signal_power / noise_power + 1e-10))

    def rms_db(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        return float(20 * np.log10(rms + 1e-10))


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — RECORDER
# The main recording loop with VAD-gated capture.
# ══════════════════════════════════════════════════════════════════

class Recorder:

    def __init__(self, vad: VADEngine, device_index: Optional[int] = None):
        self._vad    = vad
        self._device = device_index

    def record(self, max_duration_s: float = MAX_DURATION_SECS,
               silence_ms: int = SILENCE_THRESHOLD,
               calibration: Optional[CalibrationResult] = None) -> Optional[np.ndarray]:
        """
        Record until silence_ms of silence after speech begins,
        or max_duration_s total.
        Applies calibration gain if provided.
        Shows a real-time level meter while recording.
        """
        frame_samples  = int(TARGET_SR * CHUNK_MS / 1000)
        silence_frames = int(silence_ms / CHUNK_MS)
        max_frames     = int(max_duration_s * 1000 / CHUNK_MS)
        gain           = calibration.recommended_gain if calibration else 1.0

        print(f"\n[REC] Ready — speak now  (max {max_duration_s}s, "
              f"silence cutoff {silence_ms}ms)")
        print("[REC] Level: ", end="", flush=True)

        frames_raw      = []
        silent_count    = 0
        speech_started  = False
        speech_frames   = 0
        silence_frames_count = 0

        backend, stream_ctx = self._open_stream(frame_samples)
        if stream_ctx is None:
            print("\n[REC] ERROR: No audio input device available.")
            return None

        with stream_ctx:
            for _ in range(max_frames):
                frame = self._read_frame(backend, stream_ctx, frame_samples)
                if frame is None:
                    break

                # Apply calibration gain
                frame = frame * gain
                frame = np.clip(frame, -1.0, 1.0)

                is_speech = self._vad.is_speech_frame(frame)
                frames_raw.append(frame.copy())

                # Real-time level meter
                rms = float(np.sqrt(np.mean(frame ** 2)))
                bar = self._mini_bar(rms)
                print(f"\r[REC] Level: {bar}  {'SPEECH' if is_speech else 'silent'}  "
                      f"[{len(frames_raw) * CHUNK_MS / 1000:.1f}s]   ",
                      end="", flush=True)

                if is_speech:
                    speech_started      = True
                    silent_count        = 0
                    speech_frames      += 1
                elif speech_started:
                    silent_count       += 1
                    silence_frames_count += 1
                    if silent_count >= silence_frames:
                        print(f"\r[REC] Done — silence detected at "
                              f"{len(frames_raw) * CHUNK_MS / 1000:.1f}s   ")
                        break

        if not speech_started:
            print("\n[REC] No speech detected.")
            return None

        audio = np.concatenate(frames_raw)
        speech_duration = speech_frames * CHUNK_MS / 1000

        if speech_duration < MIN_SPEECH_SECS:
            print(f"[REC] Too short ({speech_duration:.2f}s < {MIN_SPEECH_SECS}s)")
            return None

        print(f"[REC] Captured: {len(audio)/TARGET_SR:.2f}s  "
              f"speech: {speech_duration:.2f}s  "
              f"vad: {self._vad.method}")
        return audio

    def _open_stream(self, frame_samples: int):
        """Try sounddevice first, then pyaudio. Returns (backend, context)."""
        try:
            import sounddevice as sd

            class SDStream:
                def __init__(self, device, sr, frames):
                    self.q    = []
                    self._buf = np.zeros(frames, dtype=np.float32)
                    self._sd  = sd
                    self._dev = device
                    self._sr  = sr
                    self._n   = frames
                    self._stream = sd.InputStream(
                        samplerate=sr, channels=1, dtype="float32",
                        blocksize=frames, device=device,
                        callback=self._cb
                    )
                def _cb(self, indata, frames, time, status):
                    self.q.append(indata[:, 0].copy())
                def __enter__(self):
                    self._stream.start()
                    return self
                def __exit__(self, *a):
                    self._stream.stop()
                    self._stream.close()
                def read(self):
                    while not self.q:
                        time_mod = __import__("time")
                        time_mod.sleep(0.005)
                    return self.q.pop(0)

            import time as time_mod
            return "sounddevice", SDStream(self._device, TARGET_SR, frame_samples)

        except Exception:
            pass

        try:
            import pyaudio

            class PAStream:
                def __init__(self, device, sr, frames):
                    self._pa  = pyaudio.PyAudio()
                    self._dev = device
                    self._sr  = sr
                    self._n   = frames
                    self._stream = None
                def __enter__(self):
                    self._stream = self._pa.open(
                        format=pyaudio.paFloat32, channels=1,
                        rate=self._sr, input=True,
                        input_device_index=self._dev,
                        frames_per_buffer=self._n,
                    )
                    return self
                def __exit__(self, *a):
                    self._stream.stop_stream()
                    self._stream.close()
                    self._pa.terminate()
                def read(self):
                    data = self._stream.read(self._n, exception_on_overflow=False)
                    return np.frombuffer(data, dtype=np.float32)

            return "pyaudio", PAStream(self._device, TARGET_SR, frame_samples)

        except Exception as e:
            print(f"[REC] Audio backend error: {e}")
            return "none", None

    def _read_frame(self, backend: str, stream, n_samples: int) -> Optional[np.ndarray]:
        try:
            return stream.read()
        except Exception:
            return None

    def _mini_bar(self, rms: float, width: int = 20) -> str:
        db       = 20 * np.log10(rms + 1e-10)
        fraction = (db + 60) / 40   # -60dB to -20dB range
        fraction = max(0.0, min(1.0, fraction))
        filled   = int(fraction * width)
        bar      = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"


# ══════════════════════════════════════════════════════════════════
# MODULE 6 — QUALITY VALIDATOR
# Checks SNR, duration, clipping, speech fraction.
# ══════════════════════════════════════════════════════════════════

class QualityValidator:

    def validate(self, audio: np.ndarray, sr: int,
                 snr_db: float) -> Tuple[str, str]:
        """
        Returns (quality_label, rejection_reason).
        quality_label: EXCELLENT / GOOD / FAIR / POOR / REJECTED
        rejection_reason: empty string unless REJECTED
        """
        duration = len(audio) / sr
        peak     = float(np.max(np.abs(audio)))
        rms      = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        rms_db   = 20 * np.log10(rms + 1e-10)

        # Hard rejections
        if duration < MIN_SPEECH_SECS:
            return "REJECTED", f"too short ({duration:.2f}s < {MIN_SPEECH_SECS}s)"
        if snr_db < MIN_SNR_DB:
            return "REJECTED", f"SNR too low ({snr_db:.1f}dB < {MIN_SNR_DB}dB)"
        if rms_db < -50:
            return "REJECTED", f"signal too weak ({rms_db:.1f}dB) — check mic"
        if peak > 0.99:
            return "REJECTED", f"clipping detected (peak={peak:.3f})"

        # Quality grading
        if snr_db >= 20 and rms_db >= -25:
            return "EXCELLENT", ""
        elif snr_db >= 12:
            return "GOOD", ""
        elif snr_db >= MIN_SNR_DB:
            return "FAIR", ""
        else:
            return "POOR", ""


# ══════════════════════════════════════════════════════════════════
# MODULE 7 — WAV WRITER
# ══════════════════════════════════════════════════════════════════

def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Save float32 audio as 16-bit PCM WAV at target sample rate."""
    # Ensure no clipping
    peak = np.max(np.abs(audio))
    if peak > 0.98:
        audio = audio / peak * 0.97
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ══════════════════════════════════════════════════════════════════
# MODULE 8 — MAIN PIPELINE
# Ties everything together.
# ══════════════════════════════════════════════════════════════════

def record_and_clean(
    out_path:       str,
    device_index:   Optional[int] = None,
    max_duration_s: float = MAX_DURATION_SECS,
    silence_ms:     int   = SILENCE_THRESHOLD,
    skip_denoise:   bool  = False,
    save_raw:       Optional[str] = None,
    vad_agg:        int   = 2,
    verbose:        bool  = True,
) -> Optional[RecordingResult]:
    """
    Full pipeline:
    1. Calibrate microphone
    2. Record with VAD gating
    3. Remove noise
    4. Validate quality
    5. Save clean WAV
    Returns RecordingResult or None on failure.
    """
    t0 = time.time()

    print("\n" + "═"*55)
    print("  AWAAZ — Calibrated Voice Recorder")
    print("═"*55)

    # ── Step 1: Calibrate ─────────────────────────────────────────
    calibrator   = MicCalibrator()
    calibration  = calibrator.calibrate(device_index, CALIBRATION_SECS)

    if verbose:
        print(f"\n[CAL] Device  : {calibration.device_name}")
        print(f"[CAL] Noise   : {calibration.noise_floor_db:.1f} dB  [{calibration.quality_label}]")
        print(f"[CAL] Gain    : {calibration.recommended_gain:.2f}x")

    if calibration.quality_label == "POOR":
        print("[CAL] WARNING: Very noisy environment. Recording quality may be poor.")

    # ── Step 2: Record ────────────────────────────────────────────
    vad       = VADEngine(aggressiveness=vad_agg)
    recorder  = Recorder(vad=vad, device_index=device_index)
    audio_raw = recorder.record(
        max_duration_s=max_duration_s,
        silence_ms=silence_ms,
        calibration=calibration,
    )

    if audio_raw is None:
        print("[REC] Recording failed or no speech detected.")
        return None

    duration_raw = len(audio_raw) / TARGET_SR
    snr_before   = 20 * np.log10(
        np.sqrt(np.mean(audio_raw ** 2)) / max(calibration.noise_floor_rms, 1e-10)
    )

    # Save raw if requested
    if save_raw:
        save_wav(audio_raw, TARGET_SR, save_raw)
        print(f"[REC] Raw audio saved: {save_raw}")

    # ── Step 3: Denoise ───────────────────────────────────────────
    denoiser = NoiseReducer()

    if skip_denoise:
        audio_clean = audio_raw.copy()
        snr_after   = snr_before
        print("[DENOISE] Skipped")
    else:
        # Build noise clip from calibration measurement
        noise_clip = None
        if calibration.noise_floor_rms > 1e-6:
            # Synthesise a noise clip from calibration RMS
            np.random.seed(42)
            noise_clip = (np.random.randn(int(TARGET_SR * 0.5))
                          * calibration.noise_floor_rms).astype(np.float32)

        print("[DENOISE] Removing background noise...")
        t_nr        = time.time()
        audio_clean = denoiser.reduce(audio_raw, TARGET_SR, noise_clip,
                                      stationary=False)
        snr_after   = denoiser.snr_db(audio_raw, audio_clean)
        snr_gain    = snr_after - snr_before
        print(f"[DENOISE] Done in {time.time()-t_nr:.2f}s  "
              f"SNR: {snr_before:.1f}→{snr_after:.1f} dB  (+{snr_gain:.1f}dB)")

    # ── Step 4: Validate ──────────────────────────────────────────
    validator = QualityValidator()
    quality, reason = validator.validate(audio_clean, TARGET_SR, snr_after)

    if quality == "REJECTED":
        print(f"\n[QUALITY] REJECTED — {reason}")
        print("[QUALITY] Recording not saved. Please try again.")
        return RecordingResult(
            path="", duration_s=duration_raw,
            sample_rate=TARGET_SR,
            word_count_est=0,
            snr_before_db=round(snr_before, 1),
            snr_after_db=round(snr_after, 1),
            noise_reduced=not skip_denoise,
            vad_method=vad.method,
            speech_frames=0, silence_frames=0,
            calibration=calibration,
            quality_label="REJECTED",
            rejection_reason=reason,
            raw_path=save_raw or "",
        )

    # ── Step 5: Save ──────────────────────────────────────────────
    save_wav(audio_clean, TARGET_SR, out_path)
    elapsed = time.time() - t0

    # Estimate word count from duration (avg 2.5 words/sec for Indian speech)
    speech_frac    = min(1.0, snr_after / 30)
    speech_dur     = duration_raw * speech_frac
    word_count_est = int(speech_dur * 2.5)

    result = RecordingResult(
        path=out_path,
        duration_s=round(duration_raw, 2),
        sample_rate=TARGET_SR,
        word_count_est=word_count_est,
        snr_before_db=round(snr_before, 1),
        snr_after_db=round(snr_after, 1),
        noise_reduced=not skip_denoise,
        vad_method=vad.method,
        speech_frames=0, silence_frames=0,
        calibration=calibration,
        quality_label=quality,
        rejection_reason="",
        raw_path=save_raw or "",
    )

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "─"*55)
    print(f"  Recording complete [{quality}]")
    print("─"*55)
    print(f"  Saved to  : {out_path}")
    print(f"  Duration  : {duration_raw:.2f}s")
    print(f"  SNR       : {snr_before:.1f} → {snr_after:.1f} dB")
    print(f"  VAD method: {vad.method}")
    print(f"  Est. words: ~{word_count_est}")
    print(f"  Time taken: {elapsed:.2f}s")
    print("─"*55)

    if verbose:
        print("\n  Next step:")
        print(f"  python awaaz_analyser.py --input {out_path}")
        print()

    return result


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

def self_test():
    """
    Run a self-test without needing a microphone.
    Generates a synthetic speech-like signal, runs it through
    the full pipeline, and checks every component works.
    """
    print("\n[TEST] Running self-test...")

    # Generate synthetic audio: 440Hz tone (speech-like) + white noise
    sr       = TARGET_SR
    duration = 3.0
    t        = np.linspace(0, duration, int(sr * duration))
    speech   = (0.3 * np.sin(2 * np.pi * 200 * t)    # fundamental
              + 0.2 * np.sin(2 * np.pi * 400 * t)    # 2nd harmonic
              + 0.1 * np.sin(2 * np.pi * 800 * t))   # 3rd harmonic
    noise    = 0.05 * np.random.randn(len(t))
    audio    = (speech + noise).astype(np.float32)

    tests_passed = []

    # Test 1: NoiseReducer
    try:
        nr      = NoiseReducer()
        cleaned = nr.reduce(audio, sr)
        snr     = nr.snr_db(audio, cleaned)
        assert len(cleaned) == len(audio), "Length mismatch after denoising"
        tests_passed.append(f"NoiseReducer: OK  (SNR {snr:.1f}dB)")
    except Exception as e:
        tests_passed.append(f"NoiseReducer: FAIL — {e}")

    # Test 2: VADEngine
    try:
        vad    = VADEngine()
        frame  = audio[:int(sr * CHUNK_MS / 1000)]
        result = vad.is_speech_frame(frame, sr)
        tests_passed.append(f"VADEngine ({vad.method}): OK  (is_speech={result})")
    except Exception as e:
        tests_passed.append(f"VADEngine: FAIL — {e}")

    # Test 3: QualityValidator
    try:
        qv    = QualityValidator()
        label, reason = qv.validate(audio, sr, snr_db=15.0)
        tests_passed.append(f"QualityValidator: OK  ({label})")
    except Exception as e:
        tests_passed.append(f"QualityValidator: FAIL — {e}")

    # Test 4: WAV save/load
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
        save_wav(audio, sr, tmp)
        _, loaded = scipy_wavfile.read(tmp)
        os.unlink(tmp)
        assert len(loaded) == len(audio), "Saved/loaded length mismatch"
        tests_passed.append("WAV save/load: OK")
    except Exception as e:
        tests_passed.append(f"WAV save/load: FAIL — {e}")

    # Test 5: MicCalibrator (without actual mic)
    try:
        cal = MicCalibrator()
        result = cal._default_calibration(None)
        tests_passed.append(f"MicCalibrator (default): OK  ({result.quality_label})")
    except Exception as e:
        tests_passed.append(f"MicCalibrator: FAIL — {e}")

    print("\n  Self-test results:")
    all_pass = True
    for t in tests_passed:
        ok = "FAIL" not in t
        print(f"  {'✓' if ok else '✗'}  {t}")
        if not ok:
            all_pass = False

    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AWAAZ Calibrated Voice Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python awaaz_recorder.py --out recording.wav
  python awaaz_recorder.py --out rec.wav --verbose
  python awaaz_recorder.py --out rec.wav --max-duration 20 --silence-ms 1000
  python awaaz_recorder.py --out rec.wav --save-raw raw.wav
  python awaaz_recorder.py --out rec.wav --device 2
  python awaaz_recorder.py --out rec.wav --no-denoise
  python awaaz_recorder.py --list-devices
  python awaaz_recorder.py --test

Then analyse the recording:
  python awaaz_analyser.py --input recording.wav
        """
    )

    parser.add_argument("--out",          default=None,
                        help="Output WAV file path (required unless --test or --list-devices)")
    parser.add_argument("--device",       type=int, default=None,
                        help="Microphone device index (see --list-devices)")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_SECS,
                        help=f"Max recording duration in seconds (default: {MAX_DURATION_SECS})")
    parser.add_argument("--silence-ms",   type=int, default=SILENCE_THRESHOLD,
                        help=f"Silence cutoff in ms (default: {SILENCE_THRESHOLD})")
    parser.add_argument("--no-denoise",   action="store_true",
                        help="Skip noise removal")
    parser.add_argument("--save-raw",     default=None, metavar="PATH",
                        help="Also save the raw (pre-denoised) audio")
    parser.add_argument("--vad-agg",      type=int, default=2, choices=[0,1,2,3],
                        help="WebRTC VAD aggressiveness 0-3 (default: 2)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print detailed calibration report")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available microphones and exit")
    parser.add_argument("--test",         action="store_true",
                        help="Run self-test without microphone")

    args = parser.parse_args()

    if args.list_devices:
        DeviceManager().print_devices()
        return

    if args.test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    if not args.out:
        parser.error("--out is required (output WAV file path)")

    result = record_and_clean(
        out_path=args.out,
        device_index=args.device,
        max_duration_s=args.max_duration,
        silence_ms=args.silence_ms,
        skip_denoise=args.no_denoise,
        save_raw=args.save_raw,
        vad_agg=args.vad_agg,
        verbose=args.verbose,
    )

    if result and result.quality_label == "REJECTED":
        sys.exit(1)


if __name__ == "__main__":
    main()
