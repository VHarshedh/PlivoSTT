import numpy as np
import librosa
import os
import soundfile as sf

def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def speech_before(y, sr, pause_start, window_s=1.5):
    """Returns the last window_s seconds of audio strictly before pause_start."""
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int(window_s * sr))
    return y[start_sample:end_sample]

def extract_pitch_slope(y, sr, pause_start, window_ms=300):
    # 1. Strictly bound the causal audio array up to pause_start
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int((window_ms / 1000.0) * sr))
    
    y_window = y[start_sample:end_sample]
    
    # Handle edge case where the window is empty or too short
    if len(y_window) < 256: 
        return 0.0  # Stable fallback slope
        
    # 2. Optimized YIN execution with tight human speech bounds
    try:
        f0 = librosa.yin(
            y_window, 
            fmin=65, 
            fmax=400, 
            sr=sr, 
            frame_length=min(len(y_window), 2048)
        )
        
        # Clean unvoiced/failed tracking frames
        f0 = f0[np.isfinite(f0)]
        if len(f0) < 2:
            return 0.0
            
        # 3. Compute linear pitch slope
        x = np.arange(len(f0))
        slope, _ = np.polyfit(x, f0, 1)
        return float(slope)
    except Exception:
        return 0.0

def extract_continuous_pitch_duration(y, sr, pause_start, window_ms=1000):
    """
    Duration of the continuous valid pitch block immediately preceding pause_start 
    to capture vowel elongation.
    """
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int((window_ms / 1000.0) * sr))
    
    y_window = y[start_sample:end_sample]
    
    if len(y_window) < 256:
        return 0.0
        
    try:
        # Use a small frame_length to get better resolution
        frame_length = min(len(y_window), 2048)
        f0 = librosa.yin(
            y_window, 
            fmin=65, 
            fmax=400, 
            sr=sr, 
            frame_length=frame_length
        )
        
        # Consider valid if it's finite and within normal pitch range
        valid_frames = np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)
        
        continuous_frames = 0
        for val in reversed(valid_frames):
            if val:
                continuous_frames += 1
            else:
                break
                
        # default hop_length in librosa.yin is frame_length // 4
        hop_length = frame_length // 4
        duration_s = continuous_frames * hop_length / float(sr)
        return float(duration_s)
    except Exception:
        return 0.0

def extract_zcr_rms_ratio(y, sr, pause_start, window_ms=400):
    """
    The ratio of the mean Zero-Crossing Rate to the RMS energy in the final 400ms window 
    to catch trailing breath signatures.
    """
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int((window_ms / 1000.0) * sr))
    
    y_window = y[start_sample:end_sample]
    if len(y_window) < 256:
        return 0.0
        
    try:
        zcr = librosa.feature.zero_crossing_rate(y_window)[0]
        rms = librosa.feature.rms(y=y_window)[0]
        
        mean_zcr = np.mean(zcr)
        mean_rms = np.mean(rms)
        
        if mean_rms < 1e-6:
            return 0.0
            
        return float(mean_zcr / mean_rms)
    except Exception:
        return 0.0

def extract_energy_decay(y, sr, pause_start, window_s=1.5):
    """
    RMS energy of the last 1.5 seconds. Compute the gradient/slope of this energy.
    """
    y_window = speech_before(y, sr, pause_start, window_s)
    if len(y_window) < 256:
        return 0.0
        
    try:
        rms = librosa.feature.rms(y=y_window)[0]
        if len(rms) < 2:
            return 0.0
        
        x = np.arange(len(rms))
        slope, _ = np.polyfit(x, rms, 1)
        return float(slope)
    except Exception:
        return 0.0

def extract_language_flag(audio_path):
    """1 if 'hindi' in path, 0 if 'english'."""
    return 1.0 if 'hindi' in audio_path.lower() else 0.0

def extract_all_features(y, sr, pause_start, audio_path):
    f_pitch_slope = extract_pitch_slope(y, sr, pause_start)
    f_energy_decay = extract_energy_decay(y, sr, pause_start)
    f_lang = extract_language_flag(audio_path)
    f_pitch_dur = extract_continuous_pitch_duration(y, sr, pause_start)
    f_zcr_rms = extract_zcr_rms_ratio(y, sr, pause_start)
    
    return np.array([
        f_pitch_slope,
        f_energy_decay,
        f_lang,
        f_pitch_dur,
        f_zcr_rms
    ], dtype=np.float32)
