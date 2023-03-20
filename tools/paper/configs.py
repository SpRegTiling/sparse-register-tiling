import re

vec_width = {
    "AVX2": 256,
    "AVX512": 512,
    "NEON": 128
}

schedules ={
    "KNM": 0,
    "NKM": 1,
    "MNK": 2,
    "NMK": 3,
    "KMN": 4,
    "MKN": 5,
    "KM": 6,
    "MK": 7,
    "KN": 8,
    "NK": 9,
    "MN": 10,
    "NM": 11,
    "K": 12,
    "N": 13,
    "M": 14,
}

def neon_nano_from_name(s, config):
    assert "tuned" not in s

    mapping = s.split("_")[2 if "split" in s else -1]
    packed = "packed" in s
    load_balance = "load_balanced" in s
    sparse_a = False
    mr_nr = re.search(r'M(\d)N(\d)', s)
    if mr_nr is None:
        mr_nr = re.search(r'(\d)x(\d)', s)
    print(s, mapping)
    size = int(mr_nr.group(1))
    nr = int(mr_nr.group(2))
    outer_schedule = s.split("_")[4 if "split" in s else 2]
    tlb_comp = 64
    beta = float(re.search(r'_B(\d+)', s).group(1)) / 10 if '_B' in s else 1.0

    extra_config = {
        "k_tile": int(re.search(r'k_tile:(\d+)', config).group(1)),
        "m_tile": int(re.search(r'm_tile:(\d+)', config).group(1)),
        "n_tile": int(re.search(r'n_tile:(\d+)', config).group(1)),
        "tiling_strategy": 0
    }
    
    return nano("NEON", size, nr, mapping, outer_schedule, packed, load_balance, None, tlb_comp, sparse_a, beta, extra_config=extra_config)


def nano_from_name(arch, s):
    assert "tuned" not in s

    mapping = s.split("_")[-1]
    packed = "packed" in s
    load_balance = "LB" in s
    sparse_a = "SA" in s
    mr_nr = re.search(r'M(\d)N(\d)', s)
    size = int(mr_nr.group(1))
    nr = int(mr_nr.group(2))
    outer_schedule = s.split("_")[2 if "NANO" in s else 1]
    tlb_comp = int(re.search(r'TLB(\d+)', s).group(1)) if 'TLB' in s else None
    beta = float(re.search(r'_B(\d+)', s).group(1)) / 10 if '_B' in s else 1.0

    return nano(arch, size, nr, mapping, outer_schedule, packed, load_balance, None, tlb_comp, sparse_a, beta)


def nano(arch, size, nr, mapping, outer_schedule,
         packed=False, load_balance=False, tune=None, tlb_comp=None, 
         sparse_a=False, beta=1.0, extra_config=None):
    name = f"M{size}N{nr}_{outer_schedule}"
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
            "outer_schedule": schedules[outer_schedule],
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
    
    if extra_config is not None:
        method["config"].update(extra_config)

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
