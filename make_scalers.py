"""Creates seq_scaler.joblib, static_scaler.joblib and y_scaler.joblib.

Run once, in the folder that contains lstm_data.npz (from the LSTM data preparation
step). The scalers are fitted on the TRAINING partition only, exactly as during model
training, and are created with the scikit-learn version installed on this computer.

    python make_scalers.py
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.load("lstm_data.npz", allow_pickle=True)
x_seq_train = data["X_seq_train"]
x_static_train = data["X_static_train"]
y_train = data["y_train"]
print("Training shapes:", x_seq_train.shape, x_static_train.shape, y_train.shape)

n_catchments = x_seq_train.shape[2]
joblib.dump(StandardScaler().fit(x_seq_train.reshape(-1, n_catchments)), "seq_scaler.joblib")
joblib.dump(StandardScaler().fit(x_static_train), "static_scaler.joblib")
joblib.dump(StandardScaler().fit(y_train.reshape(-1, 1)), "y_scaler.joblib")
print("Saved seq_scaler.joblib, static_scaler.joblib, y_scaler.joblib")
