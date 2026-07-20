# STT End-of-Turn Detection Notes

## 1. Feature Engineering and Causality Rules
Our feature pipeline revolves around a strict adherence to causal constraints. At inference, we only have access to audio up to the moment the user pauses (`pause_start`). To respect this, all our feature extractors strictly bound their analysis to `[0:pause_start]`.

We extract a concise representation using a **1.5-second causal window** looking backward from the pause. We capture:
- **Resemblyzer Embedding (256-d):** Encodes the speaker's vocal prosody, identifying intonation drops and hesitation patterns near the pause.
- **Acoustic Hand-Crafted Features:** `Energy Drop` (comparing final 0.5s to previous 0.5s) and `Voicing Density` (ZCR within a 1.0s window).
- **Global Context:** `Turn Duration` (length of speech) and `Language Flag` (English vs Hindi).

We intentionally avoided complex acoustic delta tracking and Test-Time Augmentation (TTA), as they increased noise and slowed down our response delay significantly under CPU constraints.

## 2. CPU-Constraint Optimizations
We made a conscious decision to initialize the heavy `VoiceEncoder` globally in `features_ext.py`. Loading the weights takes ~200-300ms. By caching the encoder, we prevent continuous I/O hits for every turn evaluated, making our inference loop inside `predict.py` exceptionally lean and fast for CPU environments. 

## 3. Model Architecture and L1-Regularization
Our primary challenge was **overfitting**: mapping 260 dimensional features to only 496 training pauses.
To combat this, we use a `Pipeline` architecture:
1. `StandardScaler` to normalize the diverse range of acoustic scalars and embedding values.
2. **L1-Regularized Feature Selection:** A `SelectFromModel` wrapped around an L1-penalized `LogisticRegression(C=0.5)`. This explicitly forces a large number of weights to zero, acting as an automated feature selector that aggressively drops noisy or useless embedding dimensions.
3. **Random Forest Classifier:** After shrinking the dimensionality down to the most critical ~160 features, we pass them to a robust non-linear tree model to learn the final probability of an End-of-Turn.

## 4. Hyperparameter Tuning and Optimization
We validated our pipeline using a strict **5-Fold GroupKFold CV** (grouped by `turn_id` to prevent data leakage). Inside our CV loop, we ran a granular grid search evaluating combinations of probability thresholds (`0.30 - 0.70`) and padding delays (`100ms - 600ms`). 
By explicitly optimizing for the lowest `mean response delay` while strictly enforcing a `<= 5.0%` interrupted turn rate budget, we located our global optimal operating point: **Threshold = 0.50, Delay = 250ms**, securing a final full dataset AUC of **0.979** with a lightning-fast response delay of **358ms**.

Reported performance metrics: The 'Operating Point' (250 ms) refers to the threshold/delay configuration applied to the decision engine. The 'Mean Response Delay' (358 ms) is the empirical average latency measured across all successful EOT detections at that operating point.

Hyperparameter optimization was conducted over 6 iterations using GroupKFold cross-validation to prevent speaker-dependent overfitting. The configuration chosen reflects the highest generalization performance (AUC) while maintaining the 5.0% interrupted turn budget.