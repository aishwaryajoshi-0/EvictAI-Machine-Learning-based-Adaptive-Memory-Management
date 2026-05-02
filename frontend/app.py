import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backend.simulator import PagingSimulator
from backend.ml_model import MLPagePredictor
from backend.address_translation import AddressTranslator
from backend.multiprocess import (
    generate_process_traces, equal_allocation, proportional_allocation,
    simulate_multiprocess, compare_algorithms
)
from backend.virtual_memory import VirtualMemorySimulator


# Page config
st.set_page_config(page_title="EvictAI — Adaptive Memory Management", layout="wide")
st.title("⚙️ EvictAI — Adaptive Memory Management with ML")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Paging Simulator",
    "🤖 ML Prediction",
    "📍 TLB (Translation Lookaside Buffer)",
    "🧩 Multi-Process Allocation",
    "🔥 Thrashing Monitor"
])


# ================================
# Utility functions for Thrashing
# ================================

def fault_windows(reference_string, frames, window=50):
    sim = PagingSimulator(frames)
    faults_list = []
    for i in range(0, len(reference_string), window):
        chunk = reference_string[i:i+window]
        faults_list.append(sim.lru(chunk))  # use LRU for window analysis
    return faults_list


def detect_spikes(fault_windows, spike_factor=1.5):
    if len(fault_windows) < 3:
        return False

    median_val = np.median(fault_windows)
    threshold = spike_factor * median_val

    consecutive = 0
    for f in fault_windows:
        if f > threshold:
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
    return False


def detect_thrashing_multi(stats):
    thrashing_pids = []
    for pid, st in stats.items():
        total = st["faults"] + st["hits"]
        if total == 0:
            continue
        rate = st["faults"] / total
        if rate > 0.60:
            thrashing_pids.append(pid)
    return thrashing_pids


def system_thrashing(stats):
    total_faults = sum(st["faults"] for st in stats.values())
    total_hits = sum(st["hits"] for st in stats.values())
    total = total_faults + total_hits
    if total == 0:
        return 0
    return total_faults / total

# =========================
# TAB 1 — Paging Simulator
# =========================
# =========================
# TAB 1 — Paging Simulator
# =========================
# =========================
# TAB 1 — Paging Simulator
# =========================
with tab1:
    st.header("📌 Classical Page Replacement Algorithms")

    c1, c2, c3 = st.columns(3)
    frames = c1.slider("Frames", 1, 15, 3)
    length = c2.slider("Reference Length", 1, 700, 200, step=1)
    max_page = c3.slider("Max Page ID", 1, 100, 10)

    # ⭐ NEW — Pattern Type for Better Understanding
    pattern_type = st.selectbox(
        "Reference Pattern Type",
        ["random", "locality", "looping", "bursty-locality"]
    )

    dist = st.selectbox("Distribution Type", ["uniform", "zipf", "bursty"])

    run = st.button("Run Simulation")

    # -----------------------------------------------------
    # ⭐ GENERATORS (locality, looping, bursty-locality)
    # -----------------------------------------------------
    def generate_locality_trace(length, maxp):
        ref = []
        hot = list(range(1, min(6, maxp)))
        cold = list(range(6, maxp+1))
        for i in range(length):
            if random.random() < 0.8:
                ref.append(random.choice(hot))
            else:
                ref.append(random.choice(cold))
            if i % 150 == 0 and i > 0:
                base = random.randint(1, maxp-5)
                hot = list(range(base, min(base+5, maxp)))
        return ref

    def generate_looping_trace(length):
        loop = [1,2,3,4,5,4,3,2]
        return [loop[i % len(loop)] for i in range(length)]

    def generate_bursty_locality(length, maxp):
        ref = []
        for i in range(0, length, 40):
            hot = random.randint(1, maxp // 2)
            burst = [hot] * 20 + [random.randint(1, maxp) for _ in range(20)]
            ref.extend(burst)
        return ref[:length]

    # -----------------------------------------------------
    # ⭐ Generate Reference String
    # -----------------------------------------------------
    if run:
        if pattern_type == "locality":
            ref = generate_locality_trace(length, max_page)

        elif pattern_type == "looping":
            ref = generate_looping_trace(length)

        elif pattern_type == "bursty-locality":
            ref = generate_bursty_locality(length, max_page)

        else:   # random (your old logic)
            if dist == "zipf":
                import numpy as np
                arr = np.random.zipf(2.0, size=length)
                ref = [min(max(1, int(x)), max_page) for x in arr]

            elif dist == "bursty":
                hot_range = max(1, max_page // 4)
                cold_start = hot_range + 1
                hot = [random.randint(1, hot_range) for _ in range(length // 2)]
                cold = [random.randint(cold_start, max_page) for _ in range(length // 2)]
                ref = []
                for i in range(0, len(hot), 5):
                    ref.extend(hot[i:i+5])
                    ref.extend(cold[i:i+5])
                while len(ref) < length:
                    ref.append(random.randint(1, max_page))
                ref = ref[:length]

            else:
                ref = [random.randint(1, max_page) for _ in range(length)]

        st.markdown("### Reference String (first 50 values):")
        st.code(ref[:50])

        # -----------------------------------
        # RUN CLASSICAL ALGORITHMS + MRU
        # -----------------------------------
        sim = PagingSimulator(frames)
        results = sim.run_all(ref)

        st.markdown("### Page Fault Comparison")
        cA, cB, cC, cD, cE = st.columns(5)
        cA.metric("FIFO Faults", results["FIFO"])
        cB.metric("LRU Faults", results["LRU"])
        cC.metric("LFU Faults", results["LFU"])
        cD.metric("MRU Faults", results["MRU"])
        cE.metric("Optimal Faults", results["Optimal"])

        # ---------- BEST ALGORITHM ----------
        best_algo = min(results, key=results.get)
        st.success(f"🏆 Best Performing Algorithm: **{best_algo}** (Faults = {results[best_algo]})")

        # -----------------------------------
        # 🤖 ML ALGORITHM SELECTION
        # -----------------------------------
        st.markdown("### 🤖 ML-Based Page Fault Prediction")

        ml_mode = st.selectbox(
            "Choose ML Mode",
            ["A1-next_page (RF)", "A2-evict", "A3-hitmiss"]
        )

        ml = MLPagePredictor()
        ml_faults = None

        try:
            # A3 — HIT/MISS MODEL
            if ml_mode == "A3-hitmiss":
                acc, ytrue, ypred = ml.train_hitmiss(ref, frames)
                ml_faults = ml.simulate_with_hitmiss_model(ref, frames)
                st.metric("ML Predicted Faults (Hit/Miss)", ml_faults)
                st.info(f"Accuracy = {acc:.3f}")

            # A2 — EVICT MODEL
            elif ml_mode == "A2-evict":
                acc, ytrue, ypred = ml.train_evict(ref, frames)
                if ml.model_evict:
                    ml_faults = ml.simulate_with_evict_model(ref, frames)
                    st.metric("ML Predicted Faults (Evict)", ml_faults)
                    st.info(f"Accuracy = {acc:.3f}")
                else:
                    st.error("Not enough samples to train evict model.")

            # A1 — NEXT PAGE PREDICTOR (**FIXED UNPACKING**)
            elif ml_mode == "A1-next_page (RF)":
                acc, ytrue, ypred, _ = ml.train_next_page(ref, frames)
                ml_faults = ml.simulate_with_nextpage_model(ref, frames)
                st.metric("ML Predicted Faults (Next-Page RF)", ml_faults)
                st.info(f"Accuracy = {acc:.3f}")

        except Exception as e:
            st.error(f"ML Error: {e}")

        # ---------- COMPARISON GRAPH ----------
        algo_names = list(results.keys())
        fault_values = list(results.values())

        if ml_faults is not None:
            algo_names.append("ML Predictor")
            fault_values.append(ml_faults)

        fig_compare, ax_compare = plt.subplots()
        ax_compare.bar(algo_names, fault_values)
        ax_compare.set_ylabel("Page Faults")
        ax_compare.set_title("Algorithms vs ML Predictor")
        st.pyplot(fig_compare)

# ---------------- TAB 2 ----------------
with tab2:
    st.header("🤖 ML Training Playground")
    st.markdown("""
    Train ML models independently from the main simulator.
    Compare Random Forest (RF), LSTM predictors, and evict/hit-miss models.
    """)
    st.markdown("---")

    # ------------------------------------------------------------
    #  NEW: Reference Pattern Type (like TAB 1)
    # ------------------------------------------------------------
    st.subheader("📌 Reference Pattern Type")

    ml_pattern = st.selectbox(
        "Choose ML Reference Pattern",
        ["random", "locality", "looping", "bursty-locality"]
    )

    # ------------------------------------------------------------
    #  Pattern Generators (SAME AS TAB 1)
    # ------------------------------------------------------------
    def generate_locality_trace(length, maxp):
        ref = []
        hot = list(range(1, min(6, maxp)))
        cold = list(range(6, maxp+1))
        for i in range(length):
            if random.random() < 0.8:
                ref.append(random.choice(hot))
            else:
                ref.append(random.choice(cold))
            if i % 150 == 0 and i > 0:
                base = random.randint(1, maxp-5)
                hot = list(range(base, min(base+5, maxp)))
        return ref

    def generate_looping_trace(length):
        loop = [1, 2, 3, 4, 5, 4, 3, 2]
        return [loop[i % len(loop)] for i in range(length)]

    def generate_bursty_locality(length, maxp):
        ref = []
        for i in range(0, length, 40):
            hot = random.randint(1, maxp // 2)
            burst = [hot] * 20 + [random.randint(1, maxp) for _ in range(20)]
            ref.extend(burst)
        return ref[:length]

    # ------------------------------------------------------------
    # MODE SELECTION
    # ------------------------------------------------------------
    ml_mode = st.selectbox(
        "Choose ML Mode",
        [
            "A1-next_page (RF)",
            "A1-next_page (LSTM)",
            "A2-evict",
            "A3-hitmiss"
        ],
        key="tab2_mode"
    )

    # ------------------------------------------------------------
    # CONFIG PANEL
    # ------------------------------------------------------------
    st.subheader("⚙️ Training Configuration")

    c1, c2, c3 = st.columns(3)
    frames_ml = c1.slider("Frames", 1, 20, 3)
    length_ml = c2.slider("Trace Length", 10, 2000, 400)
    max_page_ml = c3.slider("Max Page ID", 2, 200, 20)

    # LSTM Settings
    if ml_mode == "A1-next_page (LSTM)":
        st.subheader("🔮 LSTM Settings")
        context_lstm = st.slider("LSTM Context Window", 1, 50, 10)
        lstm_epochs = st.slider("Training Epochs", 1, 30, 10)
        top_k = st.slider("Top-K Predictions", 1, 10, 3)

    st.markdown("---")

    # ------------------------------------------------------------
    # TRAIN BUTTON
    # ------------------------------------------------------------
    if st.button("🚀 Generate Reference & Train", use_container_width=True):

        # ------------------------------------------------------------
        #  Generate Reference According to Pattern
        # ------------------------------------------------------------
        if ml_pattern == "random":
            ref_ml = [random.randint(1, max_page_ml) for _ in range(length_ml)]

        elif ml_pattern == "locality":
            ref_ml = generate_locality_trace(length_ml, max_page_ml)

        elif ml_pattern == "looping":
            ref_ml = generate_looping_trace(length_ml)

        elif ml_pattern == "bursty-locality":
            ref_ml = generate_bursty_locality(length_ml, max_page_ml)

        # ------------------------------------------------------------
        #  Show Reference Preview
        # ------------------------------------------------------------
        st.subheader("📌 Generated Reference Preview (first 80)")
        st.code(ref_ml[:80])

        ml = MLPagePredictor()
        st.markdown("### 🏋️ Training Status")

        try:
            # ========================================================
            # A3 — HIT/MISS
            # ========================================================
            if ml_mode == "A3-hitmiss":
                acc, y_true, y_pred = ml.train_hitmiss(ref_ml, frames_ml)
                st.success(f"🎯 Hit/Miss Accuracy: **{acc:.3f}**")

            # ========================================================
            # A2 — EVICT
            # ========================================================
            elif ml_mode == "A2-evict":
                acc, y_true, y_pred = ml.train_evict(ref_ml, frames_ml)
                st.success(f"🛑 Evict-Model Accuracy: **{acc:.3f}**")

            # ========================================================
            # A1 — NEXT-PAGE (RF)
            # ========================================================
            elif ml_mode == "A1-next_page (RF)":
                acc, y_true, y_pred, _ = ml.train_next_page(
                    ref_ml, frames_ml,
                    context=5,
                    use_lstm=False
                )
                st.success(f"📘 Next-Page RF Accuracy: **{acc:.3f}**")

            # ========================================================
            # A1 — NEXT-PAGE (LSTM)
            # ========================================================
            elif ml_mode == "A1-next_page (LSTM)":
                st.warning("⏳ Training LSTM model... please wait...")

                acc, y_true, y_pred, lstm_info = ml.train_next_page(
                    ref_ml,
                    frames_ml,
                    context=context_lstm,
                    use_lstm=True,
                    lstm_epochs=lstm_epochs
                )

                if lstm_info is None or lstm_info[0] != "trained":
                    st.error("❌ LSTM training failed (TensorFlow missing?)")

                else:
                    st.success("🔥 LSTM Model Trained Successfully!")

                    faults_lstm, details_lstm = ml.simulate_with_lstm_nextpage_model(
                        ref_ml,
                        frames_ml,
                        context=context_lstm,
                        top_k=top_k
                    )

                    st.metric("Predicted Faults (LSTM)", faults_lstm)

                    import pandas as pd
                    st.markdown("### 🔍 LSTM Step-wise Decisions (first 50)")
                    st.dataframe(pd.DataFrame(details_lstm).head(50))

        except Exception as e:
            st.error(f"❌ ML Training Error: {e}")

        # ------------------------------------------------------------
        # Prediction Samples (RF/A2/A3 Only)
        # ------------------------------------------------------------
        if ml_mode != "A1-next_page (LSTM)":
            try:
                if len(y_true) > 0:
                    import pandas as pd
                    st.markdown("### 🔬 Prediction Samples (first 25)")
                    df_sample = pd.DataFrame({
                        "True": y_true[:25],
                        "Predicted": y_pred[:25]
                    })
                    st.dataframe(df_sample, height=280)
            except:
                pass


# ---------------- TAB 3 ----------------
with tab3:
    st.header(" TLB (Translation Lookaside Buffer) Demo")
    c1, c2, c3 = st.columns(3)
    page_size = c1.number_input("Page Size (bytes)", 1, 4096, 256)
    num_pages = c2.number_input("Total Pages", 1, 256, 32)
    tlb_cap = c3.number_input("TLB Capacity", 1, 64, 8)

    at = AddressTranslator(page_size, num_pages, tlb_cap)
    demo_map = {i: i % 16 for i in range(min(32, int(num_pages)))}
    at.preload_mapping(demo_map)

    logical = st.number_input("Enter Logical Address", 1, 100000, 1000)
    if st.button("Translate Address (demo)"):
        try:
            res = at.translate(logical)
            st.info(f"Page: **{res.page}**, Offset: **{res.offset}**")
            st.write(f"TLB Hit: {res.tlb_hit}, Page Table Hit: {res.page_table_hit}")
            if res.frame is None:
                st.error("PAGE FAULT – Page not loaded in memory.")
            else:
                st.success(f"Physical Address = `{res.physical_address}`  (Frame {res.frame})")
        except Exception as e:
            st.error(e)
    st.dataframe(pd.DataFrame(at.page_table.as_rows()))

# ---------------- TAB 4 ----------------
with tab4:
    st.header("🧩 Multi-Process Frame Allocation")

    # 🔽 Algorithm selection (ADD HERE)
    algo_mp = st.selectbox(
        "Choose Page Replacement Algorithm",
        ["FIFO", "LRU", "LFU", "MRU", "OPTIMAL"],
        index=1
    )

    c1, c2 = st.columns(2)
    nproc = c1.slider("Number of Processes", 1, 8, 3)
    total_frames = c2.slider("Total Frames", 1, 128, 24)
    policy = st.selectbox("Allocation Policy", ["Equal", "Proportional"])
    max_page_mp = st.slider("Max Page ID", 1, 200, 50)
    trace_length = st.slider("Trace Length per Process", 10, 2000, 200)


    traces = generate_process_traces(nproc, trace_length, max_page_mp)
    st.json({p: t[:12] for p, t in traces.items()})

    if policy == "Equal":
        alloc = equal_allocation(total_frames, list(traces.keys()))
    else:
        alloc = proportional_allocation(total_frames, traces)

    st.json(alloc)

    faults = simulate_multiprocess(traces, alloc, algorithm=algo_mp)
    st.json(faults)

    # 🔥 ADD THIS BUTTON FOR COMPARISON MODE
    if st.button("Compare All Algorithms"):
        comp = compare_algorithms(traces, alloc)
        st.write("Comparison Results:", comp)

# ---------------- TAB 5 ----------------
with tab5:
    st.header(" Thrashing Monitor")

    # ------------------------------
    # Controls
    # ------------------------------
    frames_th = st.slider("Frames", 1, 50, 6)
    length_th = st.slider("Reference Length", 1, 3000, 800)
    max_page_th = st.slider("Max Page ID", 1, 200, 50, key="thrash_max_page")
    window = st.slider("Window Size", 1, 300, 50)
    num_proc = st.slider("Number of Processes", 1, 5, 2)

    # ------------------------------
    # Generate Process Traces
    # ------------------------------
    reference_strings = {}
    for pid in range(1, num_proc + 1):
        ref = [random.randint(1, max_page_th) for _ in range(length_th)]
        reference_strings[f"P{pid}"] = ref

    # ------------------------------
    # Run Simulator
    # ------------------------------
    sim = VirtualMemorySimulator(num_processes=num_proc, frames=frames_th)
    stats = sim.run(reference_strings)

    # ------------------------------
    # PER-PROCESS THRASHING STATUS
    # ------------------------------
    st.markdown("## 🧍 Per-Process Thrashing Status")
    for pid in reference_strings:
        status = sim.detect_thrashing(pid)
        if status == "Stable":
            st.success(f"{pid}: 🟢 Stable")
        elif status == "Warning Zone":
            st.warning(f"{pid}: 🟡 Warning Zone")
        elif status == "Beginning of Thrashing":
            st.error(f"{pid}: 🟠 Beginning of Thrashing")
        elif status == "Thrashing":
            st.error(f"{pid}: 🔥 THRASHING")
        else:
            st.info(f"{pid}: No Data")

    # ------------------------------
    # MOVING WINDOW PAGE FAULT GRAPH
    # ------------------------------
    st.markdown("## 📊 Page Faults Over Time (Sliding Window)")

    pid_show = list(reference_strings.keys())[0]  # show P1 by default
    fw = fault_windows(reference_strings[pid_show], frames_th, window)

    fig, ax = plt.subplots()
    ax.plot(fw, marker="o")
    ax.set_title(f"Page Fault Windows for {pid_show}")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Faults")
    st.pyplot(fig)

    # Spike detection
    spike = detect_spikes(fw)
    if spike:
        st.error("⚠️ Spike detected → Beginning of Thrashing!")
    else:
        st.success("No spike detected — Stable locality.")

    # ------------------------------
    # MULTI-PROCESS THRASHING DETECTION
    # ------------------------------
    st.markdown("## 👥 Multi-Process Thrashing Detection")

    thrashing_pids = detect_thrashing_multi(stats)
    if thrashing_pids:
        st.error(f"🔥 Processes Thrashing: {thrashing_pids}")
    else:
        st.success("No thrashing across processes.")

    # ------------------------------
    # SYSTEM-LEVEL THRASHING
    # ------------------------------
    st.markdown("## 🏛️ System-level Thrashing Indicator")

    sys_rate = system_thrashing(stats)
    st.write(f"System Fault Rate: **{sys_rate:.2f}**")

    if sys_rate < 0.30:
        st.success("🟢 System Stable")
    elif sys_rate < 0.45:
        st.warning("🟡 System Warning Zone")
    elif sys_rate < 0.60:
        st.error("🟠 System Beginning of Thrashing")
    else:
        st.error("🔥 SYSTEM THRASHING — performance collapse!")
