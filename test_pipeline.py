import numpy as np
import pandas as pd
import pytest

from src.config import SENSORS, WINDOW_SIZE
from src.data_generator import generate_window
from src.features import FEATURE_COLUMNS, extract_features
from src.predict import PredictiveMaintenanceEngine, risk_level
from src.preprocessing import clean_window, synchronize_streams
from src.train import train


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    train(150, model_path=path, verbose=False)
    return PredictiveMaintenanceEngine(path)


def test_window_shape():
    df = generate_window(0.5, "overload", np.random.default_rng(0))
    assert len(df) == WINDOW_SIZE and set(SENSORS) <= set(df.columns)


def test_cleaning_removes_nans():
    df = generate_window(0.0, rng=np.random.default_rng(1))
    df.loc[3:5, "temperature"] = np.nan
    assert clean_window(df)[SENSORS].isna().sum().sum() == 0


def test_features_complete():
    feats = extract_features(clean_window(generate_window(0.0, rng=np.random.default_rng(2))))
    assert list(feats) == FEATURE_COLUMNS and not any(np.isnan(list(feats.values())))


def test_synchronize_streams():
    a = pd.DataFrame({"timestamp": [0, 200, 400], "value": [1, 2, 3]})
    b = pd.DataFrame({"timestamp": [10, 190, 410], "value": [9, 8, 7]})
    out = synchronize_streams({"vibration": a, "current": b})
    assert out["current"].tolist() == [9, 8, 7]


def test_risk_levels():
    assert (risk_level(0.1), risk_level(0.4), risk_level(0.9)) == ("Low", "Medium", "High")


def test_healthy_and_critical(engine):
    rng = np.random.default_rng(5)
    healthy = [engine.assess(generate_window(0.0, rng=rng))["health_state"] for _ in range(20)]
    crit = [engine.assess(generate_window(1.2, "bearing_wear", rng))["health_state"] for _ in range(20)]
    assert healthy.count("HEALTHY") >= 17 and crit.count("CRITICAL") >= 17


def test_missing_sensor_is_tolerated(engine):
    df = generate_window(0.0, rng=np.random.default_rng(3)).drop(columns=["pressure"])
    assert engine.assess(df)["health_state"] in {"HEALTHY", "WARNING", "CRITICAL"}


def test_api_roundtrip(engine, monkeypatch):
    import api.ingest_server as srv
    monkeypatch.setattr(srv, "_engine", engine)
    client = srv.app.test_client()
    win = generate_window(1.1, "overheating", np.random.default_rng(4))
    r = client.post("/ingest", json={"device_id": "t", "samples": win.fillna(0).to_dict("records")})
    assert r.status_code == 200 and r.get_json()["alert"] is True
    assert client.get("/latest").status_code == 200
    assert client.post("/ingest", json={"samples": []}).status_code == 400
