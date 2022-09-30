import torch
import gmpy
from dataclasses import dataclass
from typing import List

@dataclass
class Acc:
    M: int = 4
    N: int = 4


@dataclass
class Codelet:
    nnz: int = 4
    cols: int = 4


class CodeletDict(dict):
    def __missing__(self, key):
        ret = self[key] = Codelet(nnz=key, cols=0)
        return ret


def pattern_code(vec):
    vec = vec.to(dtype=torch.int)
    pat = 0
    for idx, i in enumerate(vec):
        pat |= i.item() << idx
    return pat


def get_codelets_from_panel(panel: torch.Tensor) -> List[Codelet]:
    codelets = CodeletDict()
    for row in panel.t():
        nnz = gmpy.popcount(pattern_code(row))
        if nnz == 0: continue
        codelets[nnz].cols += 1

    return list(codelets.values())


def tile_matrix(mtx, tile_shape):
    return [(i, j, tile)
            for i, x in enumerate(torch.split(mtx, tile_shape[0], 0))
            for j, tile in enumerate(torch.split(x, tile_shape[1], 1))]
