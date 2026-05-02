from collections import defaultdict

class FrameAllocator:
    def __init__(self, frames_per_process):
        # avoid zero frames
        self.frames_per_process = max(1, int(frames_per_process))
        self.allocations = defaultdict(list)

    def allocate(self, pid, page):
        pages = self.allocations[pid]

        # I  f there is space → directly allocate
        if len(pages) < self.frames_per_process:
            pages.append(page)
            return

        # If no space → safe eviction (only if list is NOT empty)
        if pages:
            pages.pop(0)      # FIFO eviction
        else:
            # This should never happen, but added as extra safety
            pass

        pages.append(page)

