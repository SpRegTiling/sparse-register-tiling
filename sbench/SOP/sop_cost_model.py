from sop_utils import Codelet, Acc
from typing import List, Tuple, Optional
import numpy as np


class SOPCostModel:
    U_BASE = 0
    N_CHECK = 1
    N_BASE = 2
    N_CNZ = 3

    def __init__(self, model="ACC_GENERIC", acc: Optional[Acc] = None):
        self.uBase: float = 5.0
        self.nCheck: float = 2.0
        self.nBase: float = 3.0
        self.nCnz: float = 4.0
        if model == "ACC_GENERIC":
            assert True

        self._acc = acc
        self._model = model
        assert model in ["ACC_GENERIC", "ACC_SPECIFIC"]

    def gen_sample_configs(self):
        acc = self._acc if self._acc is not None else Acc(4, 2)
        samples = [
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=20)]),
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=3, cols=10)]),
            (acc, [Codelet(nnz=4, cols=20)]),

            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=10), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=10)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=10)]),
            (acc, [Codelet(nnz=4, cols=10)]),

            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=5), Codelet(nnz=2, cols=5), Codelet(nnz=1, cols=5)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=5), Codelet(nnz=2, cols=5)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=5)]),
            (acc, [Codelet(nnz=4, cols=5)]),

            (acc, [Codelet(nnz=4, cols=20)]),
            (acc, [Codelet(nnz=3, cols=20)]),
            (acc, [Codelet(nnz=2, cols=20)]),
            (acc, [Codelet(nnz=1, cols=20)]),

            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=2, cols=20)]),
            (acc, [Codelet(nnz=1, cols=20), Codelet(nnz=3, cols=20)]),
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=2, cols=5)]),
            (acc, [Codelet(nnz=1, cols=20), Codelet(nnz=3, cols=5)]),

            (acc, [Codelet(nnz=4, cols=1), Codelet(nnz=3, cols=1), Codelet(nnz=2, cols=1), Codelet(nnz=1, cols=1)]),
            (acc, [Codelet(nnz=4, cols=3), Codelet(nnz=3, cols=3), Codelet(nnz=2, cols=3), Codelet(nnz=1, cols=3)]),

            (acc, [Codelet(nnz=1, cols=5)]),
            (acc, [Codelet(nnz=1, cols=1)]),

            (acc, [Codelet(nnz=1, cols=32)]),
            (acc, [Codelet(nnz=1, cols=64)]),

            (acc, [Codelet(nnz=2, cols=32)]),
            (acc, [Codelet(nnz=2, cols=64)]),

            (acc, [Codelet(nnz=3, cols=32)]),
            (acc, [Codelet(nnz=3, cols=64)]),

            (acc, [Codelet(nnz=4, cols=32)]),
            (acc, [Codelet(nnz=4, cols=64)]),

            # Configs with empty patterns
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=0)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=0)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=20), Codelet(nnz=1, cols=10)]),
            (acc, [Codelet(nnz=4, cols=20), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=0)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=40), Codelet(nnz=1, cols=0)]),
            (acc, [Codelet(nnz=4, cols=32), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=0)]),
            (acc, [Codelet(nnz=4, cols=16), Codelet(nnz=3, cols=10), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=32)]),
            (acc, [Codelet(nnz=4, cols=16), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=10), Codelet(nnz=1, cols=32)]),
            (acc, [Codelet(nnz=4, cols=16), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0), Codelet(nnz=1, cols=32)]),

            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=5), Codelet(nnz=2, cols=0)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=5)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=5), Codelet(nnz=2, cols=5)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=5), Codelet(nnz=2, cols=0)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=5)]),
            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0)]),
            (acc, [Codelet(nnz=4, cols=32), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0)]),
            (acc, [Codelet(nnz=4, cols=16), Codelet(nnz=3, cols=0), Codelet(nnz=2, cols=0)]),

            (acc, [Codelet(nnz=4, cols=5), Codelet(nnz=3, cols=0)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=10)]),
            (acc, [Codelet(nnz=4, cols=0), Codelet(nnz=3, cols=64)]),
            (acc, [Codelet(nnz=4, cols=32), Codelet(nnz=3, cols=0)]),
            (acc, [Codelet(nnz=4, cols=64), Codelet(nnz=3, cols=0)]),
        ]

        acc = Acc(8, 1)
        if self._model == "ACC_GENERIC":
            assert False

        return samples

    def cost_codelet(self, c: Codelet, acc: Acc=None):
        acc = acc if acc is not None else self._acc
        if self._model == "ACC_SPECIFIC": assert acc == self._acc

        if self._model == "ACC_GENERIC":
            assert True
        elif self._model == "ACC_SPECIFIC":
            cost = self.nCheck + c.cols * (self.nBase + self.nCnz * c.nnz)

        return cost

    def cost_panel(self, codelets: List[Codelet], acc: Acc=None):
        acc = acc if acc is not None else self._acc
        if self._model == "ACC_SPECIFIC": assert acc == self._acc

        if self._model == "ACC_GENERIC":
            assert True
        elif self._model == "ACC_SPECIFIC":
            cost = self.uBase

        for c in codelets:
            cost += self.cost_codelet(c, acc)
        return cost

    def random_samples_with_cost(self):
        random_samples = []
        for acc, codelets in self.gen_sample_configs():
            random_samples.append((self.cost_panel(codelets, acc), acc, codelets))
        return random_samples

    @property
    def number_of_parameters(self):
        return 7 if self._model == "ACC_GENERIC" else 4

    def matrix_row(self, codelets: List[Codelet], acc: Acc):
        acc = acc if acc is not None else self._acc
        if self._model == "ACC_SPECIFIC": assert acc == self._acc

        row = np.zeros(self.number_of_parameters)
        if self._model == "ACC_GENERIC":
            assert False
        elif self._model == "ACC_SPECIFIC":
            row[self.U_BASE] = 1
            row[self.N_CHECK] = len(codelets)
            row[self.N_BASE] = np.sum([c.cols for c in codelets])
            row[self.N_CNZ] = np.sum([c.cols * c.nnz for c in codelets])

        return row

    def solve_least_squares(self, samples: List[Tuple[float, Acc, List[Codelet]]]):
        b = np.zeros(len(samples))
        A = np.zeros((len(samples), self.number_of_parameters))

        for i, sample in enumerate(samples):
            cost, acc, codelets = sample
            b[i] = cost
            A[i] = self.matrix_row(codelets, acc)

        assert np.linalg.matrix_rank(A) == self.number_of_parameters
        results, *rest = np.linalg.lstsq(A, b, rcond=None)

        if self._model == "ACC_SPECIFIC":
            self.uBase = results[self.U_BASE]
            self.nCheck = results[self.N_CHECK]
            self.nBase = results[self.N_BASE]
            self.nCnz = results[self.N_CNZ]
        else:
            assert True

        return results

    def __repr__(self):
        repr = f'uBase={self.uBase}, nCheck={self.nCheck}, nBase={self.nBase}, nCnz={self.nCnz}'
        if self._model == "ACC_GENERIC":
            assert True

        return repr
