import numpy as np
import librosa
import os
import soundfile as sf
import warnings

def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def extract_language_flag(audio_path):
    return 1.0 if 'hindi' in audio_path.lower() else 0.0

def extract_all_features(y, sr, pause_start, audio_path):
    # 1. Turn Duration
    f_turn_duration = float(pause_start)
    
    # 2-4. Acoustic Features
    window_s = 1.0
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int(window_s * sr))
    
    y_window = y[start_sample:end_sample]
    
    f_lang = extract_language_flag(audio_path)
    
    # Fallback for extremely short or empty audio
    if len(y_window) < 256:
        return np.array([f_turn_duration, 0.0, 0.0, 0.0, f_lang], dtype=np.float32)
        
    rms_frames = librosa.feature.rms(y=y_window)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch_frames = librosa.yin(
            y_window, 
            fmin=65, 
            fmax=400, 
            sr=sr, 
            frame_length=min(len(y_window), 2048)
        )
        
    # Ensure same length
    min_len = min(len(rms_frames), len(pitch_frames))
    rms_frames = rms_frames[:min_len]
    pitch_frames = pitch_frames[:min_len]
    
    rms_max = np.max(rms_frames)
    energy_floor = 0.01 * rms_max
    
    voiced_mask = (~np.isnan(pitch_frames)) & (rms_frames >= energy_floor)
    
    hop_length = 512
    frame_rate = sr / hop_length
    
    frames_200ms = max(1, int(0.2 * frame_rate))
    frames_400ms = max(1, int(0.4 * frame_rate))
    
    # Slice arrays
    rms_200 = rms_frames[-frames_200ms:] if len(rms_frames) >= frames_200ms else rms_frames
    rms_800 = rms_frames[:-frames_200ms] if len(rms_frames) > frames_200ms else np.array([])
    
    pitch_200 = pitch_frames[-frames_200ms:] if len(pitch_frames) >= frames_200ms else pitch_frames
    pitch_800 = pitch_frames[:-frames_200ms] if len(pitch_frames) > frames_200ms else np.array([])
    
    voiced_200 = voiced_mask[-frames_200ms:] if len(voiced_mask) >= frames_200ms else voiced_mask
    voiced_800 = voiced_mask[:-frames_200ms] if len(voiced_mask) > frames_200ms else np.array([])
    
    voiced_400 = voiced_mask[-frames_400ms:] if len(voiced_mask) >= frames_400ms else voiced_mask
    
    # 2. Robust Energy Drop
    mean_rms_200 = np.mean(rms_200) if len(rms_200) > 0 else 0.0
    mean_rms_800 = np.mean(rms_800) if len(rms_800) > 0 else 0.0
    f_energy_drop = mean_rms_200 / (mean_rms_800 + 1e-6)
    
    # 3. Voiced-Only Pitch Change
    voiced_pitch_200 = pitch_200[voiced_200]
    voiced_pitch_800 = pitch_800[voiced_800]
    
    if len(voiced_pitch_200) > 0 and len(voiced_pitch_800) > 0:
        f_pitch_change = float(np.mean(voiced_pitch_200) - np.mean(voiced_pitch_800))
    else:
        f_pitch_change = 0.0
        
    # 4. Voicing Density
    if len(voiced_400) > 0:
        f_voicing_density = float(np.sum(voiced_400) / len(voiced_400))
    else:
        f_voicing_density = 0.0
        
    return np.array([
        f_turn_duration,
        f_energy_drop,
        f_pitch_change,
        f_voicing_density,
        f_lang
    ], dtype=np.float32)
