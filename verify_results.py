import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def verify():
    # Load your outputs
    preds = pd.read_csv('predictions.csv')
    labels = pd.read_csv('eot_handout/eot_handout/eot_data/english/labels.csv') # Adjust path if needed
    
    # Merge on turn_id and pause_index to ensure alignment
    df = preds.merge(labels, on=['turn_id', 'pause_index'])
    
    # Calculate AUC
    y_true = (df['label'] == 'eot').astype(int)
    y_score = df['p_eot']
    auc = roc_auc_score(y_true, y_score)
    print(f"Verified AUC: {auc:.4f}")

    # Check Mean Latency calculation logic
    # Ensure this matches your train_model.py evaluation logic exactly
    print("Check: Ensure your NOTES.md explains that 358ms is the mean delay observed at the 250ms operating point.")

if __name__ == "__main__":
    verify()