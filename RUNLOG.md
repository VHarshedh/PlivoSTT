# RUNLOG: End-of-Turn Detection Project

- **Iteration 1: Baseline**
  - Explored initial feature sets including Turn Duration and Language. Initial models were HistGradientBoostingClassifier. Delay was around ~900ms.
- **Iteration 2: Resemblyzer Embeddings**
  - Integrated `resemblyzer` to extract 256-d prosody embeddings from the last 1.5s of speech. This successfully dropped delay to ~400ms, but caused severe overfitting (CV AUC ~0.60 vs Test AUC ~0.98).
- **Iteration 3: L1 Regularization**
  - Switched the feature selection to L1-regularized Logistic Regression (Lasso) to aggressively prune dimensions. Wrapped it in a Pipeline with StandardScaler and RandomForestClassifier.
- **Iteration 4: Operating Point Sweep**
  - Built an automated cross-validation grid search to test different values for penalty weight `C` and granular (threshold, delay) pairs to meet the 5% interrupted turn budget.
- **Iteration 5: Failed Experiments (Deltas & TTA)**
  - Experimented with Test-Time Augmentation (TTA) using causal windows of [1.5s, 1.0s, 0.5s] and Acoustic Delta tracking (RMS/ZCR slopes). This introduced noise and caused our delay to spike to 550ms+. Reverted immediately.
- **Iteration 6: Final Verification**
  - Re-hardcoded the winning baseline (`SelectFromModel(LogisticRegression(C=0.5))` + `RandomForestClassifier`), restoring our optimal 1.5s causal window pipeline.
  - Final metrics locked in: AUC = 0.979, Mean Response Delay = 358 ms, Interrupted Turns = 5.0%.