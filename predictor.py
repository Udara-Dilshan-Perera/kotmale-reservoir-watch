"""Wraps the trained LSTM and its scalers behind one small interface.

TensorFlow is imported only inside load(), so the rest of the application (and its
tests) work without it. Any object with a predict_delta(sequence, static) method
can stand in for this class.
"""
from pathlib import Path

import numpy as np

from config import WINDOW, CATCHMENTS


class LSTMPredictor:
    def __init__(self, model_path, seq_scaler_path, static_scaler_path, y_scaler_path):
        self.model_path = Path(model_path)
        self.seq_scaler_path = Path(seq_scaler_path)
        self.static_scaler_path = Path(static_scaler_path)
        self.y_scaler_path = Path(y_scaler_path)
        self.model = None

    def load(self):
        for path in (self.model_path, self.seq_scaler_path, self.static_scaler_path, self.y_scaler_path):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path.name}")
        import joblib
        from tensorflow import keras

        self.seq_scaler = joblib.load(self.seq_scaler_path)
        self.static_scaler = joblib.load(self.static_scaler_path)
        self.y_scaler = joblib.load(self.y_scaler_path)
        self.model = keras.models.load_model(str(self.model_path))

    def predict_delta(self, sequence, static):
        """Predicted change in storage (MCM) for the target day."""
        seq_scaled = self.seq_scaler.transform(sequence).reshape(1, WINDOW, len(CATCHMENTS))
        static_scaled = self.static_scaler.transform(static.reshape(1, -1))
        prediction = self.model.predict(
            {"rainfall_sequence": seq_scaled.astype("float32"),
             "static_features": static_scaled.astype("float32")},
            verbose=0,
        )
        delta = self.y_scaler.inverse_transform(np.asarray(prediction).reshape(-1, 1))[0, 0]
        return float(delta)
