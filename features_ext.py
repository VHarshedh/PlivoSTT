import numpy as np
import librosa
import soundfile as sf
import os
import warnings

# Resemblyzer imports

from resemblyzer import VoiceEncoder

# Initialize globally to prevent reloading weights on every extraction

_encoder = VoiceEncoder()

def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def extract_language_flag(audio_path):
    return 1.0 if 'hindi' in audio_path.lower() else 0.0

def extract_all_features(y, sr, pause_start, audio_path):
# 1. Turn Duration (Numeric)
    f_turn_duration = float(pause_start)

# 2. Language Flag
    f_lang = extract_language_flag(audio_path)

    # Causal Boundaries (Strictly up to pause_start)
    end_sample = int(pause_start * sr)

    # 3. Acoustic Hand-Crafted Features (1.0s causal window)
    window_ac = max(0, end_sample - int(1.0 * sr))
    y_ac = y[window_ac:end_sample]

    f_energy_drop = 1.0
    f_voicing_density = 0.0

    if len(y_ac) > int(0.5 * sr):
        rms = librosa.feature.rms(y=y_ac)[0]
        if len(rms) > 4:
            # Energy drop: mean RMS of last 20% vs previous 80%
            split_idx = int(len(rms) * 0.8)
            recent_rms = np.mean(rms[split_idx:])
            past_rms = np.mean(rms[:split_idx])
            f_energy_drop = recent_rms / (past_rms + 1e-6)
            
            # Voicing density: fraction of frames in final 40% above relative floor
            vd_idx = int(len(rms) * 0.6)
            recent_frames = rms[vd_idx:]
            threshold = 0.01 * np.max(rms)
            f_voicing_density = np.mean(recent_frames > threshold)

# 4. Acoustic Embedding (1.5s causal window)
    window_emb = max(0, end_sample - int(1.5 * sr))
    y_emb = y[window_emb:end_sample]

    # Handle extremely short/empty audio
    if len(y_emb) < int(0.1 * sr): # less than 100ms
        emb = np.zeros(256, dtype=np.float32)
    else:
        # Resemblyzer requires 16000 Hz
        if sr != 16000:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_resampled = librosa.resample(y_emb, orig_sr=sr, target_sr=16000)
        else:
            y_resampled = y_emb
            
        # Extract 256-dimensional prosody embedding
        emb = _encoder.embed_utterance(y_resampled)
        
    # Combine features: [duration, lang, energy_drop, voicing, emb_0 ... emb_255]
    features = np.concatenate([[f_turn_duration, f_lang, f_energy_drop, f_voicing_density], emb]).astype(np.float32)
    return features