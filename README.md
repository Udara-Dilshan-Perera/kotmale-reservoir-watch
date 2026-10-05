# Kotmale Reservoir Watch — application

Predicts the next day's storage change of the Kotmale Reservoir from forecasted catchment
rainfall (LSTM model), and stores the daily real records the prediction depends on.
One process serves both the API and the web interface.

## What you must add from your own project folders

| File | Where it comes from |
|---|---|
| `lstm_model.keras` | your LSTM training run |
| `kotmale_clean.csv` **or** `kotmale_features.csv` | your data preparation (either works; used once to fill the database) |
| `lstm_data.npz` | your LSTM data preparation (only needed to create the scalers) |

## First-time setup (PowerShell, inside this folder)

```
pip install -r requirements.txt
python make_scalers.py
```

`make_scalers.py` creates `seq_scaler.joblib`, `static_scaler.joblib` and `y_scaler.joblib`
from the training partition, using the scikit-learn version on your computer.

Then open `catchment_coordinates.json` and fill in the latitude and longitude of each
catchment (Hope Estate is pre-filled). These are the coordinates you used to download the
rainfall from Open-Meteo; the first lines of each downloaded CSV normally list them, together
with the timezone. Only the two "Fetch from Open-Meteo" buttons need this file.

## Run

Double-click **`start_kotmale.bat`**. It starts the service and opens
`http://localhost:8000`. (Always use this address, not the HTML files directly.)

The first start creates `kotmale.db` and imports the historical dataset. This happens once.

## Viva demonstration script

`start_kotmale.bat` sets the system date to **2026-01-01** because the historical data ends on
2025-12-31 (the real calendar date is far later). Delete that line for a real deployment.

1. **Overview** — the banner warns: *"Today's data (2026-01-01) has not been entered."*
2. **Predict** — the Predict button is disabled and explains why. This is the safeguard.
3. **Data Entry** — record 2026-01-01: type the storage, then type rainfall or press
   *Fetch from Open-Meteo* and review the values. Save.
4. The banner turns green: *Ready to predict 2026-01-02.*
5. **Predict** — load a forecast (or use a preset) and predict. Point out the typical-error
   band, and try *Extreme Event* to show the caution message.
6. **Model & Method** — the Ditwah toggle.

Say clearly if asked: adding records does **not** retrain the model. The model is fixed.
Retraining is a separate, deliberate offline step followed by re-evaluation.

## Configuration (environment variables)

| Variable | Purpose | Default |
|---|---|---|
| `KOTMALE_SYSTEM_DATE` | treat this date as "today" (demo) | the computer's date |
| `KOTMALE_ENTRY_PIN` | require a PIN to add or correct records | none |
| `OPEN_METEO_VARIABLE` | daily variable to fetch. Check the column name in your downloaded CSVs (`precipitation_sum` or `rain_sum`) | `precipitation_sum` |
| `OPEN_METEO_TIMEZONE` | day boundaries for Open-Meteo. Match the timezone in your downloaded CSVs | `Asia/Colombo` |
| `KOTMALE_DB` | database file location | `kotmale.db` |

## Rules the data entry enforces

- Days are entered **one at a time, in order** (the model needs an unbroken 30-day history).
- Records cannot be dated after the system date.
- Rainfall must be 0–600 mm per catchment; storage 0–180 MCM.
- A day-to-day storage change above 20 MCM asks for explicit confirmation.
- Only the latest record entered here can be corrected. Imported history is read-only.

## Tests

```
python -m pytest tests -v
```

The suite checks leakage prevention in the feature builder, every validation rule, the
"data not current" warning, the add-data-then-predict flow, Open-Meteo request building,
and the web layer. It uses a stand-in model, so it runs without TensorFlow.

## Offline use

Charts use Chart.js. To work without internet, download `chart.umd.min.js` (Chart.js 4.4.0)
and save it as `static/vendor/chart.umd.min.js`. The pages use it automatically.

## Known limits (state these honestly)

- One day ahead only. Multi-day forecasting needs extra methods and is future work.
- Real rainfall entered by hand may differ from the Open-Meteo values the model was trained
  on. The fetch button exists to keep inputs consistent with the training source.
- Accuracy is lower for extreme rainfall (see the Cyclone Ditwah analysis).
