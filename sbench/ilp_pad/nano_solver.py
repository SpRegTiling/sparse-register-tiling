import matplotlib.pyplot as plt
import mosek
import numpy as np
import scipy.sparse
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import math

from utils import *
from mosek_test import *
from functools import partial
from collections import defaultdict
from typing import List, Set, Dict

from dataclasses import dataclass

from sbench.SOP.distribution import extract_distribution
from sbench.SOP.mapping import save_mapping
from sbench.loaders.filelist import FilelistPathIterator

#
#   Utils
#


def popcount(x):
    return bin(x).count("1")


def row_indices_from_bit_pattern(bit_pattern, max_bit=None):
    max_bit = math.ceil(math.log2(bit_pattern)) + 1 if max_bit is None else max_bit
    return frozenset([i for i in range(max_bit) if (1 << i) & bit_pattern])


#
#   Set Elements, (for sets K, Q, R)
#


@dataclass(frozen=True)
class QElement:
    row_indices: frozenset
    bit_pattern: int

    @property
    def nnz(self):
        return len(self.row_indices)

    @staticmethod
    def new_from_bitmap(bit_pattern: int) -> 'QElement':
        return QElement(row_indices=row_indices_from_bit_pattern(bit_pattern),
                        bit_pattern=bit_pattern)


@dataclass(frozen=True)
class KElement:
    row_indices: frozenset
    bit_pattern: int

    @staticmethod
    def new_from_bitmap(bit_pattern: int) -> 'KElement':
        return KElement(row_indices=row_indices_from_bit_pattern(bit_pattern),
                        bit_pattern=bit_pattern)


@dataclass(frozen=True)
class RElement:
    q_elem: QElement
    row_idx: int


#
#   Cost Model
#

# From Niagara
nCheck = 2.0
nBase = 1.15
nCnz = 0.2
K_c = 128       # predicted K_c tile size


# fQ based on a probabilistic model, Random Matrix
def p_fQ(q: QElement, M_r: int, d: float):
    "p(q | d) = (1-d)^(M_r - q.nnz) * d^q.nnz"
    p_q = (1-d)**(M_r - q.nnz) * (d)**(q.nnz)
    return p_q * K_c


# fQ based on a extracted distribution
def d_fQ(q: QElement, distribution: Dict[int, float]):
    return distribution[q.bit_pattern] * K_c


def fS(s: Set[RElement], fQ: callable):
    Q_hat = set()
    for r in s:
        Q_hat.add(r.q_elem)
    return sum(fQ(q) for q in Q_hat)


# k(s) = union of all row indices in s
def k(s: Set[RElement]) -> KElement:
    bit_pattern = 0
    for r in s:
        bit_pattern |= 1 << r.row_idx
    return KElement.new_from_bitmap(bit_pattern)


def ncost(k: KElement):
    return nBase + nCnz * len(k.row_indices)


def subset_cost(s: Set[RElement], fS: callable):
    return nCheck + fS(s) * ncost(k(s))


#
#   Set cover model
#

def generate_Q_for(M_r: int):
    Q = set()
    for bit_pattern in range(1, 2**M_r):
        Q.add(QElement.new_from_bitmap(bit_pattern))

    return Q


def generate_R_for(Q: set):
    R = set()
    for q in Q:
        for row_idx in q.row_indices:
            R.add(RElement(q_elem=q, row_idx=row_idx))
    return R


def create_Q_to_R_map(R: set):
    Q_to_R = defaultdict(set)
    for r in R:
        Q_to_R[r.q_elem].add(r)
    return Q_to_R


def powerset(iterable, convert_tuple_to_set=False):
    """
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        NOTE: ignoring empty set
    """
    s = list(iterable)
    if convert_tuple_to_set:
        return [set().union(*c) for r in range(1, len(s)+1) for c in combinations(s, r)]
    else:
        return [combinations(s, r) for r in range(1, len(s)+1)]


def create_universal_set(R: set, Q: set, approx_algo="MERGE_ONLY"):
    Q_to_R = create_Q_to_R_map(R)

    # MERGE_ONLY i.e Q only
    if approx_algo == "MERGE_ONLY":
        QR = [Q_to_R[q] for q in Q]
        return powerset(QR, convert_tuple_to_set=True)
    else:
        raise Exception("Unknown approximation algorithm")


def compute_costs(S: List[Set[RElement]], fS: callable):
    cost = np.zeros((1, len(S)))
    for i, s in enumerate(S):
        cost[0, i] = subset_cost(s, fS)
    return cost


def set_cover(set_to_cover: Set[RElement], universal_set: List[Set[RElement]], cost: List[float],
              max_num_subset: int, force_num_subsets=False):
    n = len(universal_set)  # 2^16
    with Model() as M:
        M.setLogHandler(sys.stdout)
        x = M.variable([1, n], Domain.binary())

        # Constrain that every element is covered by at least one subset
        R_id = {}
        for i, r in enumerate(set_to_cover):
            R_id[r] = i

        rows, cols, values = [], [], []
        for i, ele in enumerate(universal_set):
            for r in ele:
                rows.append(i)
                cols.append(R_id[r])
                values.append(1.0)

        M_constraint = Matrix.sparse(n, len(set_to_cover), rows, cols, values)
        M.constraint(Expr.mul(x, M_constraint), Domain.equalsTo(1.0))

        # Constrain the number of subsets
        if force_num_subsets:
            M.constraint(Expr.sum(x, 1), Domain.equalsTo(max_num_subset))
        else:
            M.constraint(Expr.sum(x, 1), Domain.lessThan(max_num_subset))

        # Set the objective (min sum of costs)
        M.objective(ObjectiveSense.Minimize, Expr.dot(cost, x))

        M.solve()
        if M.getProblemStatus() == ProblemStatus.Unknown or M.getPrimalSolutionStatus() == SolutionStatus.Unknown:
            return {}, {}, -1, M.getProblemStatus()

        subsets_chosen_indices = np.where(x.level() > 0.9)[0]
        subsets_chosen = [universal_set[i] for i in subsets_chosen_indices]

        K = {k(s).bit_pattern for s in subsets_chosen}
        mapping = defaultdict(set)
        for s in subsets_chosen:
            for r in s:
                mapping[r.q_elem.bit_pattern].add(k(s).bit_pattern)

        return K, mapping, M.primalObjValue(), M.getProblemStatus()


#
#   Visualization
#


def visualize(K: Set[int], mapping: Dict[int, Set[int]], M_r: int, fQ: callable):
    fig, axi = plt.subplots(2, figsize=(20, 6))
    x_ticks_every = 1

    # https://stackoverflow.com/a/1168328
    PHI = (1 + math.sqrt(5))/2
    colors = ['#%06X' %  math.floor((i * PHI - math.floor(i * PHI)) * 0xFFFFFF) for i in range(1, len(K) + 1)]
    K_colors = {k: colors[i] for i, k in enumerate(K)}

    patterns = sorted(list(range(1, 2**M_r)), key=lambda x: popcount(x))

    for x, pattern in enumerate(patterns):
        for k in mapping[pattern]:
            for y in row_indices_from_bit_pattern(k & pattern):
                axi[0].scatter(x + 1, M_r - y, color=K_colors[k])
            for y in row_indices_from_bit_pattern(k & ~pattern, max_bit=M_r):
                axi[0].scatter(x + 1, M_r - y, facecolors='none', edgecolors=K_colors[k])

    axi[0].axis(xmin=0, xmax=2**M_r)
    axi[0].set_xticks([])

    def compute_fQ(x):
        return fQ(QElement.new_from_bitmap(x))

    compute_fQ = np.vectorize(compute_fQ)

    axi[1].bar(np.arange(1, 2**M_r), compute_fQ(np.array(patterns)), color='black')
    axi[1].set_xticks(np.arange(1, 2**M_r, x_ticks_every))
    axi[1].set_xticklabels([f'{x:08b}'.replace('0', ' ').replace('1', '*')
                            for x in patterns[::x_ticks_every]],
                            rotation=90)
    axi[1].set_ylabel('fQ')

    ax = plt.gca()  # get the axis
    ax.xaxis.tick_bottom()  # and move the X-Axis
    ax.yaxis.tick_left()  # remove right y-Ticks
    return plt


#
#   Main
#


if __name__ == '__main__':
    M_r = 4
    max_num_subset = 16
    sparsity = 0.7


    paths = FilelistPathIterator('../../tools/filelists/transformer_rr_nano4.txt')
    speacialized_mappings = []

    for path in paths:
        distribution = extract_distribution(M_r, path)
        fQ = partial(d_fQ, distribution=distribution)

        print(f'Speacializing for {path}')

        print("  Generating Q...")
        Q = generate_Q_for(M_r)
        print("  Generating R...")
        R = generate_R_for(Q)

        print("  Generating approximate powerset...")
        U = create_universal_set(R, Q)

        print("  Computing costs...")
        cost = compute_costs(U, partial(fS, fQ=fQ))

        print("  Solving setcover...")
        K, mapping, final_cost, status = set_cover(R, U, cost, max_num_subset)
        assert status == ProblemStatus.PrimalFeasible

        print("  Saving mapping...")
        id = save_mapping(M_r, lambda x: mapping[x], subdir='speacialized_mappings')
        print("  Saved mapping to", id)

        speacialized_mappings.append((path, id))
        # plt = visualize(K, mapping, M_r, fQ)
        # plt.show(block=True)

    with open('speacialized_mappings.txt', 'w') as f:
        for path, id in speacialized_mappings:
            f.write(f'{path},{id}\n')