import os
import torch
import zlib
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import random as rnd
from .load import load_csr

DLMC_ROOT = os.environ.get("DLMC_ROOT")


def _scan_for_subdirectories(dir):
    return [entry.name for entry in os.scandir(dir) if entry.is_dir()]


def _scan_for_files(dir, suffix):
    return [entry.name for entry in os.scandir(dir) if entry.is_file() and entry.name[-len(suffix):] == suffix]


class DLMCPathIterator:
    def __init__(self, models=None, pruning_methods=None, sparsities=None, file_list=None, random=None):
        self.__file_list = []

        if DLMC_ROOT is None:
            print("Please set the `DLMC_ROOT` environment variable to point to the expanded data from:")
            print("   https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz")
            import sys; sys.exit()

        _models = models if models is not None else _scan_for_subdirectories(DLMC_ROOT)

        if file_list is not None:
            with open(file_list) as f:
                for line in f.readlines():
                    if len(line) < 3: continue
                    self.__file_list.append(f'{DLMC_ROOT.replace("dlmc", "").rstrip("/")}/{line.rstrip()}')
        else:
            for model in _models:
                _pruning_methods = pruning_methods if pruning_methods is not None else \
                    _scan_for_subdirectories(f'{DLMC_ROOT}/{model}')

                for pruning_method in _pruning_methods:
                    _sparsities = sparsities if sparsities is not None else \
                        _scan_for_subdirectories(f'{DLMC_ROOT}/{model}/{pruning_method}')

                    for sparsity in _sparsities:
                        for file in _scan_for_files(f'{DLMC_ROOT}/{model}/{pruning_method}/{sparsity}', 'smtx'):
                            self.__file_list.append(f'{model}/{pruning_method}/{sparsity}/{file}')

        if random is not None:
            rnd.shuffle(self.__file_list)
            self.__file_list = self.__file_list[:random]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self.__file_list): raise StopIteration
        path = self.__file_list[self.__idx]
        self.__idx += 1
        if path[0] != "/":
            path = DLMC_ROOT + "/" + path
        return path

    def deterministic_hash(self):
        return zlib.adler32(str(self.__file_list).encode('utf-8'))


class DLMCLoader:
    def __init__(self, models=None, pruning_methods=None, sparsities=None, file_list=None, random=None, loader=load_csr):
        self._path_iterator_base = DLMCPathIterator(
            file_list=file_list,
            models=models,
            pruning_methods=pruning_methods,
            sparsities=sparsities,
            random=random)
        self._path_iterator = None
        self.loader = loader

    def __iter__(self):
        self._path_iterator = self._path_iterator_base.__iter__()
        return self

    def __next__(self):
        filepath = next(self._path_iterator)
        return self.loader(filepath), os.path.abspath(filepath)
