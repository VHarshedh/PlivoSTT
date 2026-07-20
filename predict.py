import argparse
import csv
import os
import numpy as np
import pandas as pd
import pickle
from features_ext import extract_all_features, load_wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.pkl")
    args = ap.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    cache = {}
    X = []
    
    labels_path = os.path.join(args.data_dir, "labels.csv")
    test_df = pd.read_csv(labels_path)
    
    print(f"Extracting features from {args.data_dir}...")
    for _, r in test_df.iterrows():
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        pause_start = float(r["pause_start"])
        features = extract_all_features(x, sr, pause_start, path)
        X.append(features)
        
    X = np.array(X)
    
    print("Predicting probabilities...")
    # Extract the exact probability of the positive class (EOT)
    probabilities = model.predict_proba(X)[:, 1]

    # Save directly to predictions.csv without scaling or thresholding
    df_out = pd.DataFrame({
        'turn_id': test_df['turn_id'],
        'pause_index': test_df['pause_index'],
        'p_eot': probabilities
    })
    
    df_out.to_csv(args.out, index=False)
    print(f"Wrote {len(df_out)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
