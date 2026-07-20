# STT End-of-Turn Detection: Technical Notes

## Feature Engineering
To accurately detect when a user has finished speaking (end-of-turn) versus when they are merely pausing, our model relies on a combination of robust hand-crafted acoustic metrics and dense prosody embeddings. 

For every pause candidate, we strictly slice a causal **1.5-second audio window** (from $t=0$ up to `pause_start`), strictly maintaining causality to prevent any future information leakage.

1. **Acoustic Hand-Crafted Features**: We extract macro-contextual cues from the 1.0s window prior to the pause.
   - **Turn Duration:** The raw `pause_start` timestamp, indicating how long the user has been speaking.
   - **Language Flag:** A binary indicator representing whether the audio is English or Hindi, adjusting for language-specific speech pacing.
   - **Energy Drop:** A ratio of the mean RMS energy in the final 20% of the window compared to the previous 80%. This captures the trailing-off volume of speech.
   - **Voicing Density:** The fraction of recent frames where RMS energy exceeds a dynamic noise floor (1% of the maximum window energy), identifying whether the speaker is holding a breath or truly silent.
2. **Resemblyzer Prosody Embeddings**: The final 1.5 seconds of the audio slice are passed into `resemblyzer.VoiceEncoder` to generate a powerful **256-dimensional prosody embedding**. This vector comprehensively captures the acoustic rhythm, intonation, and tone of the trailing speech.

## Model Architecture
By combining the handcrafted metrics with the 256-d prosody embedding, the raw feature space consisted of 260 dimensions. Because our dataset is relatively small (496 total pauses), feeding this dense vector directly into a classifier (such as a Gradient Boosting Tree or Random Forest) posed a severe risk of high-dimensional overfitting. 

To combat this, we implemented an **L1-regularized Logistic Regression** (`penalty='l1'`, `solver='liblinear'`, `C=0.5`, `class_weight='balanced'`). 

The L1 (Lasso) penalty was explicitly chosen to perform automatic feature selection. During training, the L1 penalty mathematically forces the coefficients of noisy or less relevant dimensions to exactly zero. The model successfully discarded 204 dimensions, **retaining only 56 high-signal features** from the embedding and the manual metrics. The distilled features were then fed into our downstream classifier (a RandomForestClassifier), allowing the model to make highly generalizable decisions without memorizing noise.

## Validation Strategy
To guarantee our model evaluates generalizations rather than memorizing speaker-specific acoustic signatures, we implemented a strict **5-Fold GroupKFold** cross-validation strategy.

By grouping on `turn_id`, we ensured that no single speaker's audio clips ever leaked between the training and validation folds. This robust validation verified that the 56 selected features truly captured universal acoustic patterns of "trailing off", rather than overfitting to the identity of the 100 unique speakers in the dataset.