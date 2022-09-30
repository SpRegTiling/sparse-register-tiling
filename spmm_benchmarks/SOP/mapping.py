import hashlib
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def save_mapping(M_r, mapping: callable, subdir="mappings"):
    mapping_dict = {}
    for i in range(1, 2**M_r):
        mapping_dict[i] = mapping(i)

    id = hashlib.md5(str(mapping_dict).encode('utf-8')).hexdigest()[-5:]

    with open(SCRIPT_DIR + "/{}/mapping_{}.txt".format(subdir, id), "w+") as f:
        f.write("{}\n".format(M_r))
        for k, v in mapping_dict.items():
            f.write("{}: {}\n".format(k, list(v)))

    return id