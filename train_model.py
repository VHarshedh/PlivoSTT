import argparse
import csv
import os
import numpy as np
import soundfile as sf
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from features_ext import extract_all_features

def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

def evaluate(pauses, threshold, delay, timeout_s=1.6):
    turns_cut = set()
    turn_ids = set()
    latencies = []
    for pz in pauses:
        turn_ids.add(pz["turn_id"])
        fires = pz["p"] >= threshold
        if pz["label"] == "hold":
            # false cutoff only if the delay elapses before the user resumes
            if fires and delay < pz["dur"]:
                turns_cut.add(pz["turn_id"])
        else:  # true end of turn
            latencies.append(delay if fires else timeout_s)
    cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
    return cutoff_rate, float(np.mean(latencies)) if latencies else timeout_s

def score_predictions(pauses, budget=0.05):
    TIMEOUT_S = 1.6
    THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
    DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)
    
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d, TIMEOUT_S)
            if cut <= budget and (best is None or lat < best["latency"]):
                best = {"latency": lat, "cutoff": cut, "threshold": t, "delay": d}
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs='+', required=True, help="List of data directories (e.g., eot_data/english eot_data/hindi)")
    ap.add_argument("--out_model", default="model.pkl")
    args = ap.parse_args()

    cache = {}
    X, y, groups, keys = [], [], [], []
    durations = []
    
    print(f"Loading data from {args.data_dirs}...")
    for data_dir in args.data_dirs:
        labels_path = os.path.join(data_dir, "labels.csv")
        with open(labels_path) as f:
            for r in csv.DictReader(f):
                path = os.path.join(data_dir, r["audio_file"])
                if path not in cache:
                    cache[path] = load_wav(path)
                x, sr = cache[path]
                
                pause_start = float(r["pause_start"])
                features = extract_all_features(x, sr, pause_start, path)
                
                X.append(features)
                y.append(1 if r["label"] == "eot" else 0)
                groups.append(r["turn_id"])
                keys.append((r["turn_id"], int(r["pause_index"]), r["label"], float(r["pause_end"]) - float(r["pause_start"])))
                
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"Total pauses loaded: {len(X)}")

    # Split into train/dev
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, dev_idx = next(gss.split(X, y, groups))
    
    print(f"Training on {len(train_idx)} pauses, validating on {len(dev_idx)} pauses...")
    
    # Train the model
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=200, class_weight='balanced')
    clf.fit(X[train_idx], y[train_idx])
    
    # Predict on dev set
    dev_probs = clf.predict_proba(X[dev_idx])[:, 1]
    
    # Create pause structures for scoring
    pauses_dev = []
    for i, idx in enumerate(dev_idx):
        tid, p_index, label, dur = keys[idx]
        pauses_dev.append({
            "turn_id": tid,
            "dur": dur,
            "label": label,
            "p": dev_probs[i]
        })
        
    best_op = score_predictions(pauses_dev, budget=0.05)
    print(f"Optimal operating point on Dev set:")
    print(f"  Threshold: {best_op['threshold']}")
    print(f"  Delay: {best_op['delay']}")
    print(f"  Latency: {best_op['latency']}")
    print(f"  Cutoff: {best_op['cutoff']}")
    
    # Refit on all data
    print("Refitting on all data...")
    clf.fit(X, y)
    
    # Save the model
    with open(args.out_model, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {args.out_model}")

if __name__ == "__main__":
    main()
