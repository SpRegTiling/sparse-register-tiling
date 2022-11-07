from sbench.ilp_pad.nano_solver import *

# M_r = 4
# sparsity = 0.7
#
# speacialized_mappings = []
#
# fQ = partial(p_fQ, M_r=M_r, d=1-sparsity)
#
# print("  Generating Q...")
# Q = generate_Q_for(M_r)
#
# print("  Generating R...")
# R = generate_R_for(Q)
#
# print("  Generating approximate powerset...")
# U = create_universal_set(R, Q, approx_algo="MERGE_ONLY")
#
# print("  Computing costs...")
# cost = compute_costs(U, partial(fS, fQ=fQ))
#
#
# print("  Solving setcover...")
# K, mapping, final_cost, status = set_cover(R, U, cost, 2**M_r, force_num_subsets=False)
# assert status == ProblemStatus.PrimalFeasible

for M_r in [2, 4, 6, 8]:
    print("  Saving mapping...")
    id = save_mapping(M_r, lambda x: [x], subdir=f'identity_mappings/{M_r}')
    print("  Saved mapping to", id)