from sbench.ilp_pad.nano_solver import *

M_r = 8
sparsity = 0.7

speacialized_mappings = []

fQ = partial(p_fQ, M_r=M_r, d=1-sparsity)

print("  Generating Q...")
Q = generate_Q_for(M_r)

print("  Generating R...")
R = generate_R_for(Q)

print("  Generating approximate powerset...")
U = create_universal_set(R, Q, approx_algo="MUST_OVERLAP")

print("  Computing costs...")
cost = compute_costs(U, partial(fS, fQ=fQ))

for max_num_subset in range(6, 2**M_r):
    print(f'Speacializing for {max_num_subset}')

    print("  Solving setcover...")
    K, mapping, final_cost, status = set_cover(R, U, cost, max_num_subset, force_num_subsets=True)
    assert status == ProblemStatus.PrimalFeasible

    print("  Saving mapping...")
    id = save_mapping(M_r, lambda x: mapping[x], subdir=f'sweep_mappings/ilp/{M_r}')
    print("  Saved mapping to", id)

    speacialized_mappings.append(id)
    # plt = visualize(K, mapping, M_r, fQ)
    # plt.show(block=True)

with open(f'sweep_{M_r}.txt', 'w') as f:
    for id in speacialized_mappings:
        f.write(f'sweep_mappings/ilp/{M_r}/mapping_{id}_{int(100*sparsity)}.txt\n')
