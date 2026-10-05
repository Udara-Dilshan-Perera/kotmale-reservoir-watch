"""Checks that the application prepares model inputs exactly as the model was trained.
 
Run in the backend folder (needs lstm_data.npz and the historical CSV):
 
    python verify_parity.py
 
Part 1 rebuilds the inputs for every test-set day with the application's own feature
code and compares them with the arrays saved at training time (lstm_data.npz).
Part 2, if the model and scalers can be loaded, predicts every test day through the
same code path as the deployed system and recomputes the test-set scores.
"""
import sys
from datetime import timedelta
 
import numpy as np
import pandas as pd
 
import config
import database
from features import build_model_inputs
 
TOLERANCE = 1e-3
 
 
def load_history():
    path = next((p for p in config.SEED_CSV_CANDIDATES if p.exists()), None)
    if path is None:
        sys.exit("No kotmale_clean.csv or kotmale_features.csv found in this folder.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=database.canonicalise_columns(df.columns))
    storage_col = "storage_mcm" if "storage_mcm" in df.columns else "storage_actual"
    df = df[["date", storage_col] + config.CATCHMENTS].rename(columns={storage_col: "storage_mcm"})
    df["date"] = pd.to_datetime(df["date"])
    print(f"History file: {path.name} ({len(df)} days)")
    return df.sort_values("date").set_index("date")
 
 
def main():
    data = np.load("lstm_data.npz", allow_pickle=True)
    needed = ["X_seq_test", "X_static_test", "y_test", "test_dates"]
    missing = [k for k in needed if k not in data.files]
    if missing:
        sys.exit(f"lstm_data.npz is missing {missing}. It contains: {list(data.files)}")
 
    x_seq, x_static, y_true = data["X_seq_test"], data["X_static_test"], data["y_test"].astype(float)
    dates = pd.to_datetime(data["test_dates"])
    if x_static.shape[1] != len(config.STATIC_ORDER):
        sys.exit(f"Training used {x_static.shape[1]} static features; the application builds "
                 f"{len(config.STATIC_ORDER)}.")
 
    history = load_history()
    day = timedelta(days=1)
    rebuilt_seq, rebuilt_static = [], []
    seq_bad = target_bad = 0
    static_bad = {name: 0 for name in config.STATIC_ORDER}
 
    for i, d in enumerate(dates):
        window = history.loc[d - 30 * day: d - day].reset_index()
        forecast = history.loc[d, config.CATCHMENTS].to_dict()
        seq, static = build_model_inputs(window, d.date(), forecast)
        rebuilt_seq.append(seq)
        rebuilt_static.append(static)
 
        if np.abs(seq - x_seq[i]).max() > TOLERANCE:
            seq_bad += 1
        for j in np.where(np.abs(static - x_static[i]) > TOLERANCE)[0]:
            static_bad[config.STATIC_ORDER[j]] += 1
        actual_delta = history.loc[d, "storage_mcm"] - history.loc[d - day, "storage_mcm"]
        if abs(actual_delta - y_true[i]) > TOLERANCE:
            target_bad += 1
 
    static_bad = {k: v for k, v in static_bad.items() if v}
    print(f"\nPART 1 - input parity over {len(dates)} test days ({dates[0].date()} to {dates[-1].date()})")
    print(f"  sequence inputs that differ from training : {seq_bad}")
    print(f"  static features that differ from training : {static_bad or 'none'}")
    print(f"  target values that differ from training   : {target_bad}")
    parity_ok = seq_bad == 0 and not static_bad and target_bad == 0
    print("  RESULT:", "PASS - the application builds the inputs exactly as in training"
          if parity_ok else "FAIL - see the differences above")
 
    print("\nPART 2 - reproducing the reported test scores through the deployed code path")
    from predictor import LSTMPredictor
    predictor = LSTMPredictor(config.MODEL_PATH, config.SEQ_SCALER_PATH,
                              config.STATIC_SCALER_PATH, config.Y_SCALER_PATH)
    try:
        predictor.load()
    except Exception as exc:
        print(f"  skipped - the model could not be loaded ({type(exc).__name__}: {exc})")
        return
    predictions = []
    for i in range(len(dates)):
        predictions.append(predictor.predict_delta(rebuilt_seq[i], rebuilt_static[i]))
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1} of {len(dates)} predicted")
    predictions = np.array(predictions)
    rmse = float(np.sqrt(np.mean((y_true - predictions) ** 2)))
    mae = float(np.mean(np.abs(y_true - predictions)))
    nse = float(1 - np.sum((y_true - predictions) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    print(f"\n  recomputed : NSE {nse:.3f} | RMSE {rmse:.2f} MCM | MAE {mae:.2f} MCM")
    print("  reported   : NSE 0.455 | RMSE 2.54 MCM | MAE 1.07 MCM")
    print("  RESULT:", "PASS - matches the reported test performance"
          if abs(nse - 0.455) < 0.005 else "DIFFERENT - the deployed pipeline does not reproduce the reported score")
 
 
if __name__ == "__main__":
    main()
 