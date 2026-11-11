"""Console demo: a motor degrades from healthy to critical, then is serviced.

Usage:  python -m src.demo [--fault bearing_wear]
"""
import argparse

import numpy as np

from .config import FAULTS, MODEL_PATH
from .data_generator import generate_window, severity_to_state
from .predict import PredictiveMaintenanceEngine
from .train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="bearing_wear", choices=FAULTS)
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()

    if not MODEL_PATH.exists():
        train(400)
    engine = PredictiveMaintenanceEngine()
    rng = np.random.default_rng(7)
    print(f"{'step':>4} {'sev':>5} {'truth':<9} {'predicted':<9} {'risk':<7} {'score':>5}  fault / action")
    for i in range(a.steps):
        sev = max(0.0, (i - 2) * 0.1)
        res = engine.assess(generate_window(sev, a.fault, rng))
        print(f"{i:>4} {sev:>5.2f} {severity_to_state(sev):<9} {res['health_state']:<9} "
              f"{res['risk_level']:<7} {res['risk_score']:>5.2f}  {res['fault_type'] or '-'} | {res['priority']}")


if __name__ == "__main__":
    main()
