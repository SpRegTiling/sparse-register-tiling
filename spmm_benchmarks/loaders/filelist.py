import os
import torch
import zlib
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from .load import load_csr


DATASET_DIR = os.environ.get("DATASET_DIR")

class FilelistPathIterator:
    def __init__(self, file_list):
        self.__file_list = []

        if DATASET_DIR is None:
            print("Please set the `DATASET_DIR` environment variable")
            import sys; sys.exit()

        with open(file_list) as f:
            for line in f.readlines():
                if len(line) < 3: continue
                self.__file_list.append(f'{DATASET_DIR.rstrip("/")}/{line.rstrip()}')

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self.__file_list): raise StopIteration
        path = self.__file_list[self.__idx]
        self.__idx += 1
        if path[0] != "/":
            path = DATASET_DIR + "/" + path
        return path

    def deterministic_hash(self):
        return zlib.adler32(str(self.__file_list).encode('utf-8'))


class FilelistLoader:
    def __init__(self, file_list, loader=load_csr):
        self._path_iterator_base = FilelistPathIterator(file_list)
        self._path_iterator = None
        self.loader = loader

    def __iter__(self):
        self._path_iterator = self._path_iterator_base.__iter__()
        return self

    def __next__(self):
        filepath = next(self._path_iterator)
        return self.loader(filepath), os.path.abspath(filepath)
