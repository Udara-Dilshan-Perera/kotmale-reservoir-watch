"""Generates Figure 5.5 (or wherever you place it): predicted vs. actual reservoir
storage across the test partition, with the highest-error days highlighted.

Run in the folder that has lstm_data.npz, kotmale_features.csv (or kotmale_clean.csv),
and the model + scalers (same requirements as verify_parity.py):

    pip install matplotlib
    python make_test_figure.py

Produces: test_storage_figure.png

This reuses the exact same rebuilding logic as verify_parity.py, so if that script
already printed PASS for you, this one will reproduce the same predictions.
"""
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import database
from features import build_model_inputs
from predictor import LSTMPredictor
from verify_parity import load_history

N_HIGHLIGHT = 5  # matches "5 of 731 test days" reported in the thesis


def main():
    data = np.load("lstm_data.npz", allow_pickle=True)
    dates = pd.to_datetime(data["test_dates"])
    y_true_delta = data["y_test"].astype(float)

    history = load_history()
    day = timedelta(days=1)

    predictor = LSTMPredictor(config.MODEL_PATH, config.SEQ_SCALER_PATH,
                              config.STATIC_SCALER_PATH, config.Y_SCALER_PATH)
    predictor.load()

    predicted_delta = np.empty(len(dates))
    actual_storage = np.empty(len(dates))
    predicted_storage = np.empty(len(dates))

    for i, d in enumerate(dates):
        window = history.loc[d - 30 * day: d - day].reset_index()
        forecast = history.loc[d, config.CATCHMENTS].to_dict()
        seq, static = build_model_inputs(window, d.date(), forecast)
        predicted_delta[i] = predictor.predict_delta(seq, static)

        lag1 = history.loc[d - day, "storage_mcm"]
        actual_storage[i] = history.loc[d, "storage_mcm"]
        predicted_storage[i] = lag1 + predicted_delta[i]
        if (i + 1) % 100 == 0:
            print(f"...{i + 1} of {len(dates)}")

    # Identify the highest-error days automatically, rather than assuming exact dates.
    squared_error = (y_true_delta - predicted_delta) ** 2
    order = np.argsort(squared_error)[::-1]
    highlight_idx = order[:N_HIGHLIGHT]
    share = squared_error[highlight_idx].sum() / squared_error.sum() * 100
    print(f"\nTop {N_HIGHLIGHT} error days account for {share:.1f}% of total squared error "
          f"(thesis reports 72%).")
    print("Dates identified as the highest-error cluster:")
    for i in sorted(highlight_idx):
        print(f"  {dates[i].date()}  actual Δ={y_true_delta[i]:+.2f}  "
              f"predicted Δ={predicted_delta[i]:+.2f} MCM")

    def nse(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    keep = np.ones(len(dates), dtype=bool)
    keep[highlight_idx] = False
    print(f"\nRecomputed with the {N_HIGHLIGHT} identified highest-error days excluded "
          f"({keep.sum()} of {len(dates)} days remain):")
    print(f"  NSE  = {nse(y_true_delta[keep], predicted_delta[keep]):.3f}")
    print(f"  RMSE = {rmse(y_true_delta[keep], predicted_delta[keep]):.2f} MCM")
    print(f"  MAE  = {mae(y_true_delta[keep], predicted_delta[keep]):.2f} MCM")

    pd.DataFrame({
        "date": dates.date, "actual_delta_mcm": y_true_delta, "predicted_delta_mcm": predicted_delta,
        "actual_storage_mcm": actual_storage, "predicted_storage_mcm": predicted_storage,
        "squared_error": squared_error, "is_top_error_day": ~keep,
    }).to_csv("test_predictions_full.csv", index=False)
    print("Saved test_predictions_full.csv (every test day, for your own checking)")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual_storage, label="Actual storage", color="#1C5C5E", linewidth=1.3)
    ax.plot(dates, predicted_storage, label="Predicted storage", color="#9C5330",
            linewidth=1.1, linestyle="--")
    ax.scatter(dates[highlight_idx], actual_storage[highlight_idx],
              color="#9C5330", zorder=5, s=40,
              label=f"Highest-error days (n={N_HIGHLIGHT})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Reservoir storage (MCM)")
    ax.set_title("Predicted vs. Actual Kotmale Reservoir Storage — Test Partition (2024–2025)")
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("test_storage_figure.png", dpi=200)
    print("\nSaved test_storage_figure.png")


if __name__ == "__main__":
    main()
