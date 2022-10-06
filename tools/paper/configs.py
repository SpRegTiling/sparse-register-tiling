
vec_width = {
    "AVX2": 256,
    "AVX512": 512,
    "NEON": 128
}


def nano(arch, size, nr, mapping, packed=False, load_balance=False, tune=None, tlb_comp=None, sparse_a=False, beta=1.0):
    name = f"M{size}N{nr}"
    beta_10x = int(round(beta * 10))

    default_arch_vec_width = {
        "AVX2": 256,
        "AVX512": 512,
        "NEON": 128
    }

    mapping_name = mapping
    mapping = {
        4: {"identity": "61fee", "orig": "da01e"},
        8: {"orig": "400fa", "alt": "747f9"},
    }.get(size, {}).get(mapping, mapping)

    assert (not sparse_a and beta_10x == 10 and not tlb_comp) or tune is None

    if packed: name += "_packed"
    if load_balance: name += "_LB"
    if tune is not None: name += "_tuned"
    if tlb_comp is not None: name += f"_TLB{tlb_comp}"
    if sparse_a: name += "_SA"
    if beta != 1.0: name += f"_B{beta_10x}"
    name += f'_{mapping_name}'

    method = {
        "method_id": "nano",
        "name": name,
        "options": {
            "packed": packed,
            "load_balance": load_balance,
            "mapping_id": mapping,
            "arch": arch,
            "vec_width_bits": default_arch_vec_width[arch],
            "nr": nr,
        }
    }

    if tune is not None:
        method["tune"] = { "grid": tune }
    else:
        method["config"] = {
            "tiling_strategy": 1,
            "sparse_a": 1 if sparse_a else 0,
            "beta_10x": beta_10x,
        }

        if tlb_comp is not None:
            method["config"]["max_tlb_entries"] = tlb_comp
            method["config"]["tiling_strategy"] = 2

    return method


def csb(arch, storage, nr=32, tune=None, tlb_comp=None, sparse_a=False, beta=1.0):
    name = f"CSB_{storage}"
    beta_10x = int(round(beta * 10))

    assert (not sparse_a and beta_10x == 10 and not tlb_comp) or tune is None

    name += f"_{nr}"
    if tune is not None: name += "_tuned"
    if tlb_comp is not None: name += f"_TLB{tlb_comp}"
    if sparse_a: name += "_SA"
    if beta != 1.0: name += f"_B{beta_10x}"

    method = {
        "method_id": "gecsb",
        "name": name,
        "options": {
            "storage": storage,
            "vec_width": vec_width[arch],
        }
    }

    if tune is not None:
        method["tune"] = { "grid": tune }
    else:
        method["config"] = {
            "tiling_strategy": 1,
            "sparse_a": 1 if sparse_a else 0,
            "beta_10x": beta_10x,
        }

        if tlb_comp is not None:
            method["config"]["max_tlb_entries"] = tlb_comp
            method["config"]["tiling_strategy"] = 2
    return method
