# backend/address_translation.py
from collections import OrderedDict
from dataclasses import dataclass

#A data container that stores the result of address translation.
@dataclass
class TranslationResult:
    page: int
    offset: int
    tlb_hit: bool
    page_table_hit: bool
    frame: int | None
    physical_address: int | None

#Page Table Definition
class PageTable:
    def __init__(self, num_pages: int):
        # -1 means not loaded (invalid)
        self.table = [-1] * num_pages  # index = page, value = frame or -1

# Map a page to a frame in the page table
    def map_page(self, page: int, frame: int):
        self.table[page] = frame

    def lookup(self, page: int) -> int | None:
        f = self.table[page]
        return None if f == -1 else f

    def as_rows(self):
        return [{"Page": i, "Frame": ("—" if f == -1 else f), "Valid": f != -1} for i, f in enumerate(self.table)]


# Translation Lookaside Buffer (TLB) Definition
class TLB:
    def __init__(self, capacity: int = 8):
        self.cap = capacity
        self.cache = OrderedDict()  # page -> frame (LRU)

    def get(self, page: int) -> int | None:
        if page in self.cache:
            self.cache.move_to_end(page)
            return self.cache[page]
        return None

    def put(self, page: int, frame: int):
        if page in self.cache:
            self.cache.move_to_end(page)
        else:
            if len(self.cache) == self.cap:
                self.cache.popitem(last=False)
            self.cache[page] = frame

class AddressTranslator:
    """
    Simple address translator:
      - logical address split into (page, offset)
      - TLB lookup -> Page Table fallback
      - constructs physical address as frame * page_size + offset
    """
    def __init__(self, page_size: int, num_pages: int, tlb_capacity: int = 8):
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        self.page_size = page_size
        self.num_pages = num_pages
        self.page_table = PageTable(num_pages)
        self.tlb = TLB(tlb_capacity)

    def preload_mapping(self, mapping: dict[int, int]):
        """mapping: {page -> frame} to mark as resident/valid."""
        for p, f in mapping.items():
            if 0 <= p < self.num_pages:
                self.page_table.map_page(p, f)
                self.tlb.put(p, f)

    def split(self, logical_address: int) -> tuple[int, int]:
        page = logical_address // self.page_size
        offset = logical_address % self.page_size
        if page < 0 or page >= self.num_pages:
            raise ValueError("Logical address out of range for configured num_pages/page_size")
        return page, offset

    def translate(self, logical_address: int) -> TranslationResult:
        page, offset = self.split(logical_address)

        # 1) TLB lookup
        f = self.tlb.get(page)
        if f is not None:
            return TranslationResult(page, offset, True, True, f, f * self.page_size + offset)

        # 2) Page table lookup
        frame = self.page_table.lookup(page)
        if frame is not None:
            # populate TLB on PT hit
            self.tlb.put(page, frame)
            return TranslationResult(page, offset, False, True, frame, frame * self.page_size + offset)

        # 3) Not in memory (page fault): no physical address
        return TranslationResult(page, offset, False, False, None, None)
