# backend/multiprocess.py
import random
from typing import Dict, List
from collections import defaultdict
from backend.simulator import PagingSimulator

def generate_process_traces(num_processes=3, length_per_proc=120, max_page=20, distribution="zipf_smooth"):
    import numpy as np
    traces = {}
    for pid in range(1, num_processes + 1):
        arr = np.random.zipf(1.5, size=length_per_proc)
        arr = [min(max(1, int(x)), max_page) for x in arr]

        # apply smoothing: replace back-to-back repeats
        for i in range(2, len(arr)):
            if arr[i] == arr[i-1] == arr[i-2]:
                arr[i] = random.randint(1, max_page)

        traces[f"P{pid}"] = arr
    return traces


def equal_allocation(total_frames: int, processes: List[str]) -> Dict[str, int]:
    base = total_frames // len(processes)
    alloc = {p: base for p in processes}
    r = total_frames - base * len(processes)
    for p in processes[:r]:
        alloc[p] += 1
    return alloc

def proportional_allocation(total_frames: int, traces: Dict[str, List[int]]) -> Dict[str, int]:
    sizes = {p: len(set(t)) for p, t in traces.items()}
    total = sum(sizes.values())
    if total == 0:
        return equal_allocation(total_frames, list(traces.keys()))
    alloc = {p: max(1, (sizes[p] * total_frames) // total) for p in traces}
    diff = total_frames - sum(alloc.values())
    keys = list(alloc.keys())
    i = 0
    while diff != 0:
        if diff > 0:
            alloc[keys[i % len(keys)]] += 1
            diff -= 1
        else:
            if alloc[keys[i % len(keys)]] > 1:
                alloc[keys[i % len(keys)]] -= 1
                diff += 1
        i += 1
    return alloc


# 🔥 UPDATED — MULTI-ALGORITHM SUPPORT (FIFO, LRU, OPT)
def simulate_multiprocess(traces: Dict[str, List[int]], frame_alloc: Dict[str, int], algorithm="LRU") -> Dict[str, int]:
    faults = {}
    for pid, ref in traces.items():
        frames = frame_alloc.get(pid, 1)
        sim = PagingSimulator(frames)

        if algorithm.upper() == "FIFO":
            faults[pid] = sim.fifo(ref)
        elif algorithm.upper() == "OPT":
            faults[pid] = sim.optimal(ref)
        else:
            faults[pid] = sim.lru(ref)

    return faults

def compare_algorithms(traces: Dict[str, List[int]], frame_alloc: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """
    Run simulate_multiprocess for multiple algorithms and return a dict:
    {
      "FIFO": {"P1": faults, "P2": faults, ...},
      "LRU": {...},
      ...
    }
    """
    comparison = {}
    for algo in ["FIFO", "LRU", "LFU", "MRU", "OPTIMAL"]:
        res = simulate_multiprocess(traces, frame_alloc, algorithm=algo)
        comparison[algo] = res
    return comparison


def aggregate_faults_over_time(traces: Dict[str, List[int]], frame_alloc: Dict[str, int], window=20):
    series = defaultdict(list)
    for pid, ref in traces.items():
        frames = frame_alloc.get(pid, 1)
        sim = PagingSimulator(frames)
        for i in range(0, len(ref), window):
            chunk = ref[i:i+window]
            series[pid].append(sim.lru(chunk))
    return dict(series)
