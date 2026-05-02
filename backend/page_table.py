class PageTable:
    def __init__(self):
        self.entries = {}

    def is_valid(self, page):
        return self.entries.get(page, {}).get("valid", False)

    def update(self, page, frame=None, valid=False):
        self.entries[page] = {"frame": frame, "valid": valid}

    def __repr__(self):
        return str(self.entries)
