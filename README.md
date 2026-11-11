# AI-Based Predictive Maintenance System for Real-Time Industrial Motor Health Monitoring and Failure Prediction

A software-first, end-to-end predictive-maintenance project: condition-monitoring sensors → ESP32 edge device →
preprocessing → feature extraction → AI/ML engine → health assessment → failure-risk prediction →
maintenance decision support → dashboard & alerts → maintenance action (feedback to the motor).

> **Honest scope note:** the ML models in this repository are trained on **simulated** sensor data
> (`src/data_generator.py`). Reported accuracy therefore shows that the pipeline works – it is *not* evidence of
> performance on a real motor. Replace the generator with data recorded from your own motor and sensors
> (see "Using real data") before making any real-world claims. The ESP32 firmware is provided as a
> template and has not been calibrated or tested on specific hardware.

## System architecture

```
Industrial Motor ─► Condition Monitoring Sensors ─► ESP32 Data Acquisition / Edge Device
  (vibration, temperature, current, voltage, speed, pressure[optional])        │ Wi-Fi / HTTP JSON
                                                                              ▼
                                                               Data Preprocessing  (src/preprocessing.py)
                                                                              ▼
                                                               Feature Extraction  (src/features.py)
                                                                              ▼
                                              AI/ML Predictive Intelligence Engine (src/train.py, src/predict.py)
                                                                              ▼
                                  Motor Health Assessment ─► Failure Risk Assessment ─► Maintenance Decision Support
                                  (HEALTHY/WARNING/CRITICAL)   (Low/Medium/High)
                                                                              ▼
                                                           Monitoring Dashboard & Alerts (app/dashboard.py)
                                                                              ▼
                                                  Maintenance Action ─► feedback to Industrial Motor
```

| Block | Implementation |
|---|---|
| Sensors + ESP32 | `firmware/esp32_motor_monitor/esp32_motor_monitor.ino` |
| Ingest API | `api/ingest_server.py` (Flask: `POST /ingest`, `GET /latest`) |
| Preprocessing | Timestamp sync, interpolation of missing values, 3-point median filter; scaling via `StandardScaler` |
| Feature extraction | mean, RMS, std, variance, peak, peak-to-peak, max, min, rate of change × 6 sensors = 54 features |
| AI/ML engine | Random Forest health classifier + fault classifier + Isolation-Forest anomaly detector (model is swappable: `--model gradient_boosting / logistic_regression`) |
| Health assessment | HEALTHY / WARNING / CRITICAL |
| Failure risk | `risk_score = 0.5·P(WARNING) + P(CRITICAL)` → Low (<0.25) / Medium (<0.60) / High |
| Decision support | Recommendation, priority, corrective action per fault type (`src/predict.py`) |
| Dashboard & alerts | Streamlit; simulation mode with maintenance feedback loop |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --samples 600        # train models -> models/pm_model.joblib, metrics.json
python -m src.demo --fault bearing_wear  # console demo of a degrading motor
streamlit run app/dashboard.py           # dashboard (trains automatically on first run if needed)
python -m pytest -q                      # tests
```

### Dashboard modes
* **Simulation** – choose a fault, press *Next reading* or *Auto-run*; severity grows each step so you can watch
  Healthy → Warning → Critical. *Perform maintenance* resets the motor (the feedback loop).
* **Upload CSV** – columns `vibration,temperature,current,voltage,speed,pressure` (optional `timestamp`);
  a sample file can be downloaded from the sidebar.
* **ESP32 live** – start `python -m api.ingest_server`, flash the firmware, set the API URL in the sidebar.

## ESP32 hardware setup
1. Open `firmware/esp32_motor_monitor/esp32_motor_monitor.ino` in Arduino IDE (ESP32 board package installed).
2. Set `WIFI_SSID`, `WIFI_PASS`, `SERVER_URL` (your PC's LAN IP, port 5000).
3. Wire sensors to the ADC1 pins listed in the file header and **calibrate** the constants
   (`VIB_MMS_PER_V`, `CUR_A_PER_V`, …) against reference instruments.
4. Mains voltage/current sensing must use isolated modules and be installed by a qualified person.

## Using real data
1. Log windows from the ESP32 (or save the CSV from the dashboard) while the motor is healthy and, when safely
   possible, under known faults/seeded faults.
2. Build a labelled feature table with `clean_window` → `extract_features` and train with the same
   `Pipeline` code in `src/train.py` (replace `generate_dataset`).
3. Evaluate with a **time-/machine-wise split** (not random windows) to avoid optimistic results.

## Project layout
```
src/            config, data_generator, preprocessing, features, train, predict, demo
app/            Streamlit dashboard
api/            Flask ingest API for the ESP32
firmware/       ESP32 Arduino sketch
tests/          pytest suite (pipeline, models, API)
models/         trained model bundle, metrics.json, confusion_matrix.png (generated)
```

## Limitations and future work
* Synthetic training data; real-world validation required.
* Predicts the current health class/risk, not a time-to-failure (RUL is out of scope).
* Vibration is summarised as one RMS value per reading; raw high-rate FFT analysis would improve bearing diagnostics.
* Future: real dataset collection, hardware calibration, MQTT/cloud transport, notification channels (SMS/email).
