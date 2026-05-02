# backend/run_simulation.py
import random
from backend.simulator import PagingSimulator
from backend.ml_model import MLPagePredictor

def generate_reference_string(length=100, max_page=10, distribution="uniform"):
    if distribution == "uniform":
        return [random.randint(1, max_page) for _ in range(length)]
    if distribution == "zipf":
        # simple zipf-like: many small numbers
        import numpy as np
        arr = np.random.zipf(2.0, size=length)
        # normalize and clip
        arr = [min(max(1, int(x)), max_page) for x in arr]
        return arr
    return [random.randint(1, max_page) for _ in range(length)]

def main():
    frames = 3
    length = 200
    ref = generate_reference_string(length=length, max_page=10, distribution="uniform")
    sim = PagingSimulator(frames)
    results = sim.run_all(ref)
    print("Reference string (first 40):", ref[:40], "...")
    print("Classical results (page faults):")
    for k,v in results.items():
        print(f"  {k}: {v}")

    # ML
    ml = MLPagePredictor()
    try:
        acc = ml.train(ref, frames)
        print(f"ML model test accuracy (predicting faults): {acc:.3f}")
    except Exception as e:
        print("ML training error:", e)

if __name__ == "__main__":
    main()
