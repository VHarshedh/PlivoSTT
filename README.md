# PlivoSTT: End-of-Turn Detection

A high-performance CPU-only predictive model designed to calculate `p_eot` (the probability that a human conversational turn is over at a given pause). This solution dramatically reduces Mean Response Delay from the naive 1600ms baseline down to **100ms** at a <5% false cutoff rate.

## Key Highlights
* **Strict Causal Feature Extraction**: Processes only audio from `0` to `pause_start`. Includes optimized pitch mapping (`librosa.yin`), energy decay gradients, continuous pitch block durations for vowel elongation, and ZCR-to-RMS ratios for trailing breath signatures.
* **Model Architecture**: Uses a robust `HistGradientBoostingClassifier` trained on dev/validation splits grouped by `turn_id`. 
* **Raw Probability Output**: `predict.py` directly exposes uncalibrated `predict_proba()` signals, empowering the official evaluation scorer to dynamically sweep and discover the perfect non-distorted boundary matrix.

## Human & AI Collaboration
This project represents a balanced split between human intuition and AI syntax generation:
1. **Human Design**: Conceptualized the acoustic feature paradigms (vowel elongation blocks, breath signatures via ZCR), mandated raw probability outputs for density preservation, and defined YIN bounding boundaries.
2. **AI Execution**: Generated robust numpy/librosa boilerplate, managed WSL cross-validation pipelines, and handled strict serialization safety logic via `pickle`.

See `SUMMARY.html` for graphical breakdowns and in-depth performance stats.
