from backend.page_table import PageTable
from backend.tlb import TLB
from backend.allocation import FrameAllocator
from collections import defaultdict
import random

class VirtualMemorySimulator:
    def __init__(self, num_processes=3, frames=12, tlb_size=4):
        self.num_processes = num_processes
        self.frames = frames

        # Assign frames per process (equal distribution)
        self.frames_per_process = max(1, frames // num_processes)

        self.tlb = TLB(size=tlb_size)
        self.page_tables = [PageTable() for _ in range(num_processes)]

        # Updated allocator with safe logic
        self.frame_allocator = FrameAllocator(self.frames_per_process)

        self.stats = defaultdict(lambda: {"faults": 0, "hits": 0})

    def run(self, reference_strings):
        """
        reference_strings: {"P1": [...], "P2": [...], ...}
        """
        for pid, ref in reference_strings.items():
            
            # Convert "P1" -> 0, "P2" -> 1 → index for page table
            idx = int(pid[1:]) - 1

            for page in ref:

                # ---------- TLB Hit ----------
                if self.tlb.lookup(pid, page):
                    self.stats[pid]["hits"] += 1
                    continue

                # ---------- Page Table Miss ----------
                if not self.page_tables[idx].is_valid(page):

                    # Page Fault
                    self.stats[pid]["faults"] += 1

                    # Allocate frame safely
                    self.frame_allocator.allocate(pid, page)

                    # Update page table entry
                    self.page_tables[idx].update(page, valid=True)

                # ---------- Add to TLB ----------
                self.tlb.add(pid, page)

        return self.stats

    def detect_thrashing(self, pid):
        faults = self.stats[pid]["faults"]
        hits = self.stats[pid]["hits"]

        total = faults + hits
        if total == 0:
            return "NO DATA"

        rate = faults / total

        if rate < 0.30:
            return "Stable"
        elif rate < 0.45:
            return "Warning Zone"
        elif rate < 0.60:
            return "Beginning of Thrashing"
        else:
            return "Thrashing"
