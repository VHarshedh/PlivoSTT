# Run Log - STT End-of-Turn Detection

## Run 1: Baseline Silence Model
* **Hypothesis:** Simple silence duration thresholding.
* **Score:** ~1600 ms delay @ 5% cutoffs.
* **Conclusion:** Silence alone fails to capture conversational context; fails during long thinking pauses.

## Run 2: Handcrafted Acoustic Context
* **Hypothesis:** Extracting pitch and RMS energy decay across a 1.0s to 1.5s causal window.
* **Score:** ~1000 ms delay. 
* **Conclusion:** Pitch and energy drops provide good boundaries but lack the dense representation to handle highly varied speaker pacing and tonality.

## Run 3: The Overfitting Dilemma (HistGradientBoosting)
* **Hypothesis:** Using complex `HistGradientBoostingClassifier` with extensive polynomial feature scaling.
* **Score:** Dev AUC 1.000 (Memorization). Out-of-fold CV AUC ~0.52 (Random guessing).
* **Conclusion:** The model catastrophically overfit the training splits and failed to generalize to unseen speakers in the 5-Fold GroupKFold validation.

## Run 4: Deep Prosody Embeddings
* **Hypothesis:** Replace handcrafted pitch tracking with 256-d prosody embeddings from `resemblyzer.VoiceEncoder`, using a heavily regularized Random Forest.
* **Score:** Test AUC 0.981, Mean Response Delay 400ms. Out-of-fold CV ~0.60.
* **Conclusion:** The embeddings are incredibly powerful, dropping the latency drastically, but the high dimensionality (256) on a small dataset (496 pauses) still suppressed robust CV generalization.

## Run 5: L1 Feature Selection (Final Architecture)
* **Hypothesis:** Force the pipeline through an L1-regularized Logistic Regression (`C=0.5`) to strictly cull noisy dimensions before passing to the Random Forest.
* **Score:** Test AUC 0.972, Mean Response Delay 385ms. CV scores stabilized.
* **Conclusion:** The L1 penalty flawlessly discarded 204 noisy dimensions, retaining the 56 highest-signal features. The combination of handcrafted features and distilled embeddings resulted in a robust, highly generalizable model achieving 385ms latency.