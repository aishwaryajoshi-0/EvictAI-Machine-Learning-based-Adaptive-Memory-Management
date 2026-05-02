import math
from collections import deque, OrderedDict, Counter

class PagingSimulator:
    def __init__(self, frames: int):
        self.frames = int(frames)


    def fifo(self, reference_string):
        memory = deque()
        page_faults = 0
        for page in reference_string:
            if page not in memory:
                page_faults += 1
                if len(memory) == self.frames:
                    memory.popleft()
                memory.append(page)
        return page_faults


    def lru(self, reference_string):
        # OrderedDict: most recently used are at the end
        memory = OrderedDict()
        page_faults = 0
        for page in reference_string:
            if page in memory:
                # mark as recently used
                memory.move_to_end(page)
            else:
                page_faults += 1
                if len(memory) == self.frames:
                    # remove least recently used (first item)
                    memory.popitem(last=False)
                memory[page] = True
        return page_faults


    def lfu(self, reference_string):
        memory = set()
        freq = Counter()
        last_used = {}
        page_faults = 0
        t = 0
        for page in reference_string:
            t += 1
            freq[page] += 1
            last_used[page] = t
            if page in memory:
                continue
            page_faults += 1
            if len(memory) < self.frames:
                memory.add(page)
            else:
                # victim = page in memory with smallest frequency; ties -> oldest last_used
                victim = min(memory, key=lambda x: (freq[x], last_used.get(x, 0)))
                memory.remove(victim)
                memory.add(page)
        return page_faults


    def optimal(self, reference_string):
        memory = set()
        page_faults = 0
        n = len(reference_string)
        for i, page in enumerate(reference_string):
            if page in memory:
                continue
            page_faults += 1
            if len(memory) < self.frames:
                memory.add(page)
            else:
                future = reference_string[i+1:]
                # next position of each page in memory (inf if not used again)
                def next_pos(x):
                    try:
                        return future.index(x)
                    except ValueError:
                        return math.inf
                # pick page with farthest next use
                victim = max(memory, key=next_pos)
                memory.remove(victim)
                memory.add(page)
        return page_faults


    def mru(self, reference_string):
        """
        Most Recently Used (MRU) page replacement algorithm.
        Removes the most recently accessed page when a fault occurs
        and memory is full.
        """
        memory = []
        last_used = {}
        page_faults = 0
        t = 0

        for page in reference_string:
            t += 1
            last_used[page] = t

            if page in memory:
                # already in memory, no fault
                continue

            page_faults += 1

            if len(memory) < self.frames:
                memory.append(page)
            else:
                # remove the page that was used most recently (highest last_used)
                victim = max(memory, key=lambda x: last_used[x])
                memory.remove(victim)
                memory.append(page)
        
        return page_faults

    def run_all(self, reference_string):
        return {
            "FIFO": self.fifo(reference_string),
            "LRU": self.lru(reference_string),
            "LFU": self.lfu(reference_string),
            "MRU": self.mru(reference_string),
            "Optimal": self.optimal(reference_string),
        }
    

    # --- NEW UTILS BELOW (keep inside PagingSimulator class or refactor as free funcs if you prefer) ---

    def faults_over_time(self, reference_string, window=20, algo="LRU"):
        """Return list of page faults measured in fixed-size windows."""
        faults = []
        for i in range(0, len(reference_string), window):
            chunk = reference_string[i:i+window]
            if algo.upper() == "FIFO":
                faults.append(self.fifo(chunk))
            elif algo.upper() == "LFU":
                faults.append(self.lfu(chunk))
            elif algo.upper() == "OPTIMAL":
                faults.append(self.optimal(chunk))
            else:  # LRU default
                faults.append(self.lru(chunk))
        return faults

    def detect_thrashing(self, reference_string, window=20, algo="LRU", spike_factor=1.5, min_windows=4):
        """
        Simple thrashing heuristic:
          - compute faults per window
          - if we see a sustained spike (current > spike_factor * median) for >= 2 consecutive windows → thrashing
        Returns dict with 'is_thrashing', 'fault_windows', 'threshold'
        """
        fw = self.faults_over_time(reference_string, window=window, algo=algo)
        if len(fw) < min_windows:
            return {"is_thrashing": False, "fault_windows": fw, "threshold": None}

        import statistics
        med = statistics.median(fw)
        threshold = spike_factor * med if med > 0 else max(fw)  # avoid zero
        consec = 0
        is_thrashing = False
        for val in fw:
            if val > threshold:
                consec += 1
                if consec >= 2:
                    is_thrashing = True
                    break
            else:
                consec = 0
        return {"is_thrashing": is_thrashing, "fault_windows": fw, "threshold": threshold}

