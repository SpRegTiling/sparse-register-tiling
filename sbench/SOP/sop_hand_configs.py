import gmpy
import torch

class SOPOrig:
    def __init__(self, merge_patterns, split_patterns, split_cost=1.1, merge_cost=1, min_merge=2, recursive=False):
        self.merge_patterns = merge_patterns
        self.split_patterns = split_patterns
        self.split_cost = split_cost
        self.merge_cost = merge_cost
        self.min_merge = min_merge
        self.recursive = recursive

    def all_patterns(self):
        return self.merge_patterns + self.split_patterns

    def map_pattern(self, pat_code, cant_overlap=0):
        if pat_code == 0: return [0]

        pat_code = int(pat_code)
        if gmpy.popcount(pat_code) < self.min_merge:
            return [pat_code]

        best_pattern = None
        min_cost = torch.inf

        patterns = []

        for pattern in self.merge_patterns:
            if pattern & cant_overlap > 0: continue

            padding = gmpy.popcount(pattern) - gmpy.popcount(pattern & pat_code)
            split = gmpy.popcount(~pattern & pat_code)
            cost = self.split_cost * split + self.merge_cost * padding
            if cost < min_cost:
                best_pattern = pattern
                min_cost = cost

        assert best_pattern is not None

        patterns.append(best_pattern)

        split_pattern = ~best_pattern & pat_code

        if split_pattern:
            if self.recursive:
                #print(split_pattern, self.map_pattern(split_pattern))
                patterns += self.map_pattern(split_pattern, cant_overlap=best_pattern | cant_overlap)
            else:
                idx = 0
                while split_pattern:
                    if split_pattern & 1:
                        patterns.append(1 << idx)

                    idx += 1
                    split_pattern >>= 1

        #if split_pattern

        # if len(patterns) > 2:
        #     print("multisplit", self, self.recursive, patterns)

        return patterns


def gen_all_combos(vec_height, nnz):
    if nnz == 1:
        return [1 << i for i in range(vec_height)]

    combos = set()
    for pat in gen_all_combos(vec_height, nnz-1):
        for pat1 in gen_all_combos(vec_height, 1):
            if gmpy.popcount(pat | pat1) == nnz:
                combos.add(pat | pat1)

    return list(combos)


sop4_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 2**4)),
        split_cost=10000,
        min_merge=5
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(4, 2) + [0b00001111],
        split_patterns=[],
        split_cost=10000,
        min_merge=1
    ),
    SOPOrig(
        merge_patterns=[0b00001111],
        split_patterns=[1 << i for i in range(4)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(4, 3) + [0b00001111],
        split_patterns=[1 << i for i in range(4)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(4, 2) + [0b00001111],
        split_patterns=[1 << i for i in range(4)],
        split_cost=10000
    )
]

sop6_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 2**6)),
        split_cost=10000,
        min_merge=7
    ),
    SOPOrig(
        merge_patterns=[
            0b010101, 0b101010,
            0b111000, 0b000111,
            0b111111
        ],
        split_patterns=[1 << i for i in range(6)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b000011, 0b001100, 0b110000, 0b101010, 0b010101,
            0b111111
        ],
        split_patterns=[1 << i for i in range(6)],
        split_cost=2.1,
        recursive=True
    )
]


sop8_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 256)),
        split_cost=10000,
        min_merge=9
    ),
    SOPOrig(
        merge_patterns=[0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011,
            0b00111100,
            0b11010111, 0b10111101,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)]
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011,
            0b00111100, 0b00001111, 0b11110000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 4) + gen_all_combos(8, 6) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 3) + gen_all_combos(8, 5) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 2) + gen_all_combos(8, 5) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=2.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[0b11111111],
        split_patterns=[1 << i for i in range(8)]
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b00111100, 0b00001111, 0b11110000,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=2.1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b00011111, 0b11111000,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 2) + [
            0b01010101, 0b10101010,
            #0b00001111, 0b00111100, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=3.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=3.1,
        recursive=True
    )
]
