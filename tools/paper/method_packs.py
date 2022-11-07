from tools.paper.configs import nano, csb, nano_from_name


method_packs = {
    "NEON": {
        "xnn": [
            {
                "name": "XNN",
                "method_id": "xnn",
            },
        ],
        "nano4_part1": [
            nano("NEON", 4, 4, "orig", "N", load_balance=True),
            nano("NEON", 4, 6, "orig", "N", load_balance=True),
        ],
    },
    "AVX512": {
        "aspt": [
            {
                "name": "ASpT",
                "method_id": "aspt",
                "options": {
                    "vec_width": "not-supported"
                }
            }
        ],
        # "taco": [
        #     {
        #         "name": "TACO_4",
        #         "method_id": "taco",
        #         "options": {
        #             "width": 4
        #         }
        #     },
        #     {
        #         "name": "TACO_16",
        #         "method_id": "taco",
        #         "options": {
        #             "width": 16
        #         }
        #     },
        # ],
        # "mkl": [
        #     {
        #         "name": "MKL_Dense",
        #         "method_id": "mkl_dense"
        #     },
        #     {
        #         "name": "MKL_Sparse",
        #         "method_id": "mkl",
        #         "options": {
        #             "inspector": False
        #         }
        #     },
        #     {
        #         "name": "MKL_Sparse_IE",
        #         "method_id": "mkl",
        #         "options": {
        #             "inspector": True
        #         }
        #     }
        # ],
        # "mkl_bsr": [
        #     {
        #         "name": "MKL_BSR_B2",
        #         "method_id": "mkl_bsr",
        #         "options": {
        #             "block_size": 2
        #         }
        #     },
        #     {
        #         "name": "MKL_BSR_B4",
        #         "method_id": "mkl_bsr",
        #         "options": {
        #             "block_size": 4
        #         }
        #     },
        #     {
        #         "name": "MKL_BSR_B8",
        #         "method_id": "mkl_bsr",
        #         "options": {
        #             "block_size": 8
        #         }
        #     }
        # ],
        # "nano4_bests_part1": [
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_LB_TLB128_SA_identity"),
        #     nano_from_name("AVX512", "NANO_M4N4_KNM_LB_TLB128_SA_identity"),
        #     nano_from_name("AVX512", "NANO_M4N4_KNM_LB_SA_identity"),
        #     nano_from_name("AVX512", "NANO_M4N4_KNM_identity"),
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_identity"),
        # ],
        # "nano4_bests_part2": [
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_LB_orig"),
        #     nano_from_name("AVX512", "NANO_M4N4_KNM_LB_orig"),
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_LB_SA_identity"),
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_LB_TLB64_SA_orig"),
        #     nano_from_name("AVX512", "NANO_M4N4_NKM_LB_TLB64_SA_identity"),
        # ],
        # "nano8_bests_part1": [
        #     nano_from_name("AVX512", "NANO_M8N2_KNM_alt"),
        #     nano_from_name("AVX512", "NANO_M8N2_NKM_alt"),
        #     nano_from_name("AVX512", "NANO_M8N3_NKM_LB_TLB128_SA_orig"),
        #     nano_from_name("AVX512", "NANO_M8N2_KNM_orig"),
        #     nano_from_name("AVX512", "NANO_M8N2_NKM_orig"),
        # ],
        # "nano8_bests_part2": [
        #     nano_from_name("AVX512", "NANO_M8N3_KNM_LB_TLB128_SA_orig"),
        #     nano_from_name("AVX512", "NANO_M8N2_KNM_LB_TLB64_SA_alt"),
        #     nano_from_name("AVX512", "NANO_M8N2_NKM_LB_TLB64_SA_alt"),
        #     nano_from_name("AVX512", "NANO_M8N3_KNM_LB_orig"),
        #     nano_from_name("AVX512", "NANO_M8N2_KNM_LB_TLB128_SA_orig"),
        # ],
        # "nano4x6_identity_NKM": [
        #     nano("AVX512", 4, 6, "identity", "NKM"),
        #     nano("AVX512", 4, 6, "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=48),
        #     nano("AVX512", 4, 6, "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=64),
        #     nano("AVX512", 4, 6, "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=128),
        # ],
        # "nano4x6_orig_NKM": [
        #     nano("AVX512", 4, 6, "orig", "NKM",     load_balance=True),
        #     nano("AVX512", 4, 6, "orig", "NKM",     load_balance=True, sparse_a=True, tlb_comp=64),
        # ],
        # "nano4x6_identity_KNM": [
        #     nano("AVX512", 4, 6, "identity", "KNM"),
        #     nano("AVX512", 4, 6, "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=48),
        #     nano("AVX512", 4, 6, "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=64),
        #     nano("AVX512", 4, 6, "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=128),
        # ],
        # "nano4x6_orig_KNM": [
        #     nano("AVX512", 4, 6, "orig", "KNM",     load_balance=True),
        #     nano("AVX512", 4, 6, "orig", "KNM",     load_balance=True, sparse_a=True, tlb_comp=64),
        # ]
        # "nano4_orig_NKM": [
        #     nano(arch, 4, 4, "orig", "NKM",     load_balance=True),
        #     nano(arch, 4, 4, "orig", "NKM",     load_balance=True, sparse_a=True, tlb_comp=64),
        #     nano(arch, 4, 6, "orig", "NKM",     load_balance=True),
        #     nano(arch, 4, 6, "orig", "NKM",     load_balance=True, sparse_a=True, tlb_comp=64),
        # ],
        # "csb": [
        #     csb(arch, "CSR", 32),
        #     csb(arch, "CSR", 32, sparse_a=True, tlb_comp=64),
        #     csb(arch, "CSR", 64),
        #     csb(arch, "CSR", 64, sparse_a=True, tlb_comp=64),
        # ]
    }
}