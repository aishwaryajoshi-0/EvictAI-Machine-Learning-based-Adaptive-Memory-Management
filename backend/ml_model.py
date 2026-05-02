# backend/ml_model.py
import numpy as np
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict
import random
import warnings

# Try TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


class MLPagePredictor:

    def __init__(self, random_state: int = 42):
        self.rs = random_state

        # RF models
        self.model_hitmiss: Optional[RandomForestClassifier] = None
        self.model_next: Optional[RandomForestClassifier] = None
        self.model_evict: Optional[RandomForestClassifier] = None

        # LSTM model
        self.model_next_lstm = None
        self.page_to_idx = None
        self.idx_to_page = None
        self.lstm_vocab_size = None
        self.lstm_context = None

        self.hitmiss_window = 5

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def _ensure_rng(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ----------------------------------------------------------
    # A3 – HIT/MISS MODEL
    # ----------------------------------------------------------
    def train_hitmiss(self, ref: List[int], frames: int,
                      test_size=0.2, window=5, seed=None):

        self._ensure_rng(seed)
        self.hitmiss_window = window

        X = []
        labels = []
        recent = deque(maxlen=window)
        freq = Counter()

        # compute true labels using LRU
        mem = []
        true_labels = []
        for p in ref:
            if p in mem:
                true_labels.append(0)
                mem.remove(p)
                mem.append(p)
            else:
                true_labels.append(1)
                if len(mem) < frames:
                    mem.append(p)
                else:
                    mem.pop(0)
                    mem.append(p)

        # feature building
        for idx, p in enumerate(ref):
            distinct_recent = len(set(recent))
            freq_page = freq[p]
            try:
                recency = list(recent)[::-1].index(p) + 1
            except ValueError:
                recency = window + 1

            seen_recent = 1 if p in recent else 0
            access_mod = idx % (window * 2 + 1)

            feat = [
                p, freq_page, recency, distinct_recent,
                frames, seen_recent, access_mod
            ]
            X.append(feat)

            recent.append(p)
            freq[p] += 1

        X = np.array(X)
        y = np.array(true_labels)

        if len(np.unique(y)) < 2:
            self.model_hitmiss = None
            return 0.0, [], []

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=self.rs
        )

        clf = RandomForestClassifier(n_estimators=150, random_state=self.rs)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)

        self.model_hitmiss = clf
        preds = clf.predict(Xte).tolist()

        return float(acc), yte.tolist(), preds

    # ----------------------------------------------------------
    # A1 – RF or LSTM NEXT-PAGE MODEL
    # ALWAYS RETURNS: (acc_rf, ytrue_rf, ypred_rf, lstm_info)
    # ----------------------------------------------------------
    def train_next_page(
        self,
        ref: List[int],
        frames: int,
        context=5,
        test_size=0.2,
        use_lstm=False,
        lstm_epochs=8,
        lstm_batch=64,
        seed=None
    ):

        self._ensure_rng(seed)

        # build RF dataset
        X = []
        y = []

        for i in range(len(ref) - 1):
            start = max(0, i - context + 1)
            ctx = ref[start: i + 1]
            pad = [0] * (context - len(ctx))
            X.append(pad + ctx)
            y.append(ref[i + 1])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 5:
            self.model_next = None
            return 0.0, [], [], None

        # RF training
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size,
            random_state=self.rs, shuffle=True
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=self.rs)
        clf.fit(Xtr, ytr)
        acc_rf = float(clf.score(Xte, yte))

        self.model_next = clf
        y_true_rf = yte.tolist()
        y_pred_rf = clf.predict(Xte).tolist()

        # no LSTM requested
        if not use_lstm:
            return acc_rf, y_true_rf, y_pred_rf, None

        # LSTM but TF missing
        if not TF_AVAILABLE:
            return acc_rf, y_true_rf, y_pred_rf, ("tensorflow_missing", None)

        # ----------------------- LSTM training -----------------------

        pages = sorted(list(set(ref)))
        self.page_to_idx = {p: i + 1 for i, p in enumerate(pages)}
        self.idx_to_page = {i + 1: p for i, p in enumerate(pages)}

        vocab_size = len(pages) + 1
        self.lstm_vocab_size = vocab_size
        self.lstm_context = context

        seq_X, seq_y = [], []
        for i in range(len(ref) - 1):
            start = max(0, i - context + 1)
            ctx = ref[start: i + 1]
            pad = [0] * (context - len(ctx))
            mapped = [self.page_to_idx.get(x, 0) for x in (pad + ctx)]
            seq_X.append(mapped)
            seq_y.append(self.page_to_idx.get(ref[i + 1], 0))

        seq_X = np.array(seq_X)
        seq_y = np.array(seq_y)

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=32,
                            input_length=context, mask_zero=True))
        model.add(LSTM(64))
        model.add(Dense(vocab_size, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(seq_X, seq_y, epochs=lstm_epochs,
                  batch_size=lstm_batch, verbose=0)

        self.model_next_lstm = model

        lstm_info = ("trained", vocab_size, len(pages))

        return acc_rf, y_true_rf, y_pred_rf, lstm_info

    # ----------------------------------------------------------
    # Predict top-K next pages using LSTM
    # ----------------------------------------------------------
    def predict_next_with_lstm(self, context_seq: List[int], top_k=3):

        if self.model_next_lstm is None:
            raise ValueError("LSTM model not trained")

        ctx = context_seq[-self.lstm_context:]
        pad = [0] * (self.lstm_context - len(ctx))
        mapped = [self.page_to_idx.get(x, 0) for x in (pad + ctx)]

        preds = self.model_next_lstm.predict(
            np.array([mapped]), verbose=0
        )[0]

        top_idx = preds.argsort()[-top_k:][::-1]

        results = []
        for idx in top_idx:
            if idx == 0:
                continue
            page = self.idx_to_page.get(int(idx))
            if page is not None:
                results.append((page, float(preds[idx])))

        return results

    # ----------------------------------------------------------
    # A2 – EVICT MODEL
    # ----------------------------------------------------------
    def train_evict(self, ref: List[int], frames: int, window=5, test_size=0.2):

        X, y = [], []
        resident = []

        for i, p in enumerate(ref):

            if p in resident:
                continue

            if len(resident) < frames:
                resident.append(p)
                continue

            # optimal victim
            future = ref[i + 1:]
            def next_pos(x):
                try:
                    return future.index(x)
                except ValueError:
                    return 10**9

            victim = max(resident, key=next_pos)
            victim_idx = resident.index(victim)

            feat = [p] + resident.copy()
            if len(feat) < frames + 1:
                feat += [0] * ((frames + 1) - len(feat))

            X.append(feat)
            y.append(victim_idx)

            resident[victim_idx] = p

        if len(X) < 10:
            self.model_evict = None
            return 0.0, [], []

        X = np.array(X)
        y = np.array(y)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size,
            random_state=self.rs, shuffle=True
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=self.rs)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)

        self.model_evict = clf
        preds = clf.predict(Xte).tolist()

        return float(acc), yte.tolist(), preds

    def predict_evict_index(self, resident_pages, current_page):
        if self.model_evict is None:
            return 0
        feat = [current_page] + resident_pages.copy()
        if len(feat) < self.model_evict.n_features_in_:
            feat += [0] * (self.model_evict.n_features_in_ - len(feat))
        return int(self.model_evict.predict([feat])[0])

    # ----------------------------------------------------------
    # SIMULATION USING MODELS
    # ----------------------------------------------------------
    def simulate_with_lstm_nextpage_model(self, ref: List[int], frames: int,
                                          context=10, top_k=3):

        if self.model_next_lstm is None:
            raise ValueError("LSTM next-page model not trained")

        mem = []
        faults = 0
        details = []

        for i, p in enumerate(ref):

            # HIT
            if p in mem:
                mem.remove(p)
                mem.append(p)
                details.append({
                    "idx": i, "page": p, "hit": True,
                    "evicted": None, "pred_topk": None
                })
                continue

            # MISS
            faults += 1

            if len(mem) < frames:
                mem.append(p)
                details.append({
                    "idx": i, "page": p, "hit": False,
                    "evicted": None, "pred_topk": None
                })
                continue

            # full → LSTM eviction
            start = max(0, i - context)
            ctx = ref[start:i]

            preds = self.predict_next_with_lstm(ctx, top_k=top_k)

            score_map = {r: 0.0 for r in mem}
            for pg, pr in preds:
                if pg in score_map:
                    score_map[pg] += pr

            min_score = min(score_map.values())
            candidates = [r for r in mem if score_map[r] == min_score]

            if len(candidates) == 1:
                victim = candidates[0]
            else:
                future = ref[i + 1:]
                def next_pos(x):
                    try:
                        return future.index(x)
                    except ValueError:
                        return 10**9
                victim = max(candidates, key=next_pos)

            victim_idx = mem.index(victim)
            mem[victim_idx] = p

            details.append({
                "idx": i, "page": p, "hit": False,
                "evicted": victim, "pred_topk": preds
            })

        return faults, details

    def simulate_with_nextpage_model(self, ref: List[int], frames: int, context=5):

        if self.model_next is None:
            raise ValueError("RF next-page model not trained")

        mem = []
        faults = 0

        for i, p in enumerate(ref):

            if p in mem:
                mem.remove(p)
                mem.append(p)
                continue

            faults += 1

            if len(mem) < frames:
                mem.append(p)
            else:
                start = max(0, i - context)
                ctx = ref[start:i]
                pad = [0] * (context - len(ctx))
                feat = pad + ctx

                try:
                    pred_next = int(self.model_next.predict([feat])[0])
                except Exception:
                    pred_next = None

                future = ref[i + 1:]
                def next_pos(x):
                    try:
                        return future.index(x)
                    except ValueError:
                        return 10**9

                victim = max(mem, key=next_pos)
                mem[mem.index(victim)] = p

        return faults

    def simulate_with_evict_model(self, ref: List[int], frames: int):

        if self.model_evict is None:
            raise ValueError("Evict model not trained")

        mem = []
        faults = 0

        for i, p in enumerate(ref):

            if p in mem:
                continue

            faults += 1

            if len(mem) < frames:
                mem.append(p)
            else:
                idx = self.predict_evict_index(mem.copy(), p)
                idx = idx % len(mem)
                mem[idx] = p

        return faults
