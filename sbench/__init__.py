import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

def _register_extensions():
    import importlib
    import os
    import torch
    import sys

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    cwd = os.getcwd()
    importlib.invalidate_caches()

    # Force find the installed version incase we are in the working tree
    sbench_installed = importlib.machinery.PathFinder.find_spec("sbench", [path for path in sys.path if path != cwd] +
                                                                ['/sdb/codegen/spmm-nano-bench/'] +
                                                                [f'{SCRIPT_DIR}/../'])
    extfinder = importlib.machinery.FileFinder(sbench_installed.origin.replace("__init__.py", ""), loader_details)

    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)


_register_extensions()