from collections import deque

class TLB:
    def __init__(self, size=4):
        self.size = size
        self.entries = deque()

    def lookup(self, pid, page):
        if (pid, page) in self.entries:
            self.entries.remove((pid, page))
            self.entries.append((pid, page))
            return True
        return False

    def add(self, pid, page):
        if (pid, page) in self.entries:
            self.entries.remove((pid, page))
        elif len(self.entries) == self.size:
            self.entries.popleft()
        self.entries.append((pid, page))
