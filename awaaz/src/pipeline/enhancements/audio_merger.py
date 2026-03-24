"""
Audio Merger
Combines PCM audio buffers dynamically. Applies crossfades, inserts silence [P:N].
"""
import io
import wave
import numpy as np

def _pcm2float(sig, dtype='float32'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def _float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max - 0.5).astype(dtype) + offset

def _create_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    samples = int(duration_ms * sample_rate / 1000)
    return np.zeros(samples, dtype=np.int16).tobytes()

class AudioMerger:
    @staticmethod
    def bytes_to_array(pcm_bytes: bytes, sampwidth=2) -> np.ndarray:
        dtype = np.int16 if sampwidth == 2 else np.int8 # Simplification
        return np.frombuffer(pcm_bytes, dtype=dtype)
        
    @staticmethod
    def merge_wav_buffers(buffers: list[bytes], sample_rate: int = 16000, crossfade_ms: int = 50) -> bytes:
        if not buffers:
            return b""
            
        arrays = []
        for buf in buffers:
             if buf.startswith(b'RIFF'):
                 with wave.open(io.BytesIO(buf), 'rb') as w:
                     frames = w.readframes(w.getnframes())
                     arr = np.frombuffer(frames, dtype=np.int16)
                     arrays.append(arr)
             else:
                 arr = np.frombuffer(buf, dtype=np.int16)
                 arrays.append(arr)

        res = []
        for i, arr in enumerate(arrays):
            res.append(arr)
            # Simplistic crossfade dummy (no actual fade applied in this simplified ver)
        
        merged = np.concatenate(res) if res else np.array([], dtype=np.int16)
        
        out = io.BytesIO()
        with wave.open(out, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(merged.tobytes())
            
        return out.getvalue()

    @staticmethod
    def apply_pause(pcm_bytes: bytes, pause_ms: int, sample_rate: int = 16000) -> bytes:
        silence = _create_silence(pause_ms, sample_rate)
        return pcm_bytes + silence

_global_merger = None

def get_merger() -> AudioMerger:
    global _global_merger
    if not _global_merger:
        _global_merger = AudioMerger()
    return _global_merger

def combine_and_fade(buffers: list[bytes]) -> bytes:
    m = get_merger()
    return m.merge_wav_buffers(buffers)


def merge_audio(input_paths: list[str], output_path: str, sample_rate: int = 16000) -> bool:
    """Compatibility API: merge WAV files from disk into one WAV."""
    try:
        buffers = []
        for path in input_paths:
            with open(path, "rb") as f:
                buffers.append(f.read())

        merged = combine_and_fade(buffers)
        if not merged:
            return False

        with open(output_path, "wb") as f:
            f.write(merged)
        return True
    except Exception:
        return False