# Notes - STT End-of-Turn Detection Design

1. The model relies on a HistGradientBoostingClassifier trained entirely on engineered causal features extracted from the final 1.5 seconds prior to a pause.
2. Primary signals include the fundamental frequency ($F_0$) trajectory using an optimized YIN implementation and localized energy decay gradients.
3. Advanced indicators track vowel elongation through continuous harmonic block durations and trailing exhalations via localized Zero-Crossing Rate (ZCR) spikes.
4. Language boundaries are preserved using an explicit binary file-path flag for English vs. Hindi speech pacing.
5. The model outputs raw probabilities to allow the evaluation script to sweep optimal thresholds dynamically without warping the distribution density.
6. The current architecture achieves a mean response delay of 100 ms at a 4.0% false cutoff rate on the dev set.
7. While robust against acoustic shifts, the model could plateau if encountering extremely noisy environment audio where the ZCR floor is masked.
8. With one additional day, I would implement robust spectral flux tracking to better handle environmental audio distortions and voice filtering variations.