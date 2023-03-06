import os
import torch
import zlib
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from .load import load_csr


DATASET_DIR = os.environ.get("DATASET_DIR")

class FilelistPathIterator:
    def __init__(self, file_list, percentage=1.0, random=False, datatset_dir=None, range=None):
        self.__file_list = []

        if datatset_dir is None and DATASET_DIR is None:
            print("Please set the `DATASET_DIR` environment variable")
            import sys; sys.exit()

        if datatset_dir is None:
            datatset_dir = DATASET_DIR

        with open(file_list) as f:
            for line in f.readlines():
                if len(line) < 3: continue
                self.__file_list.append(f'{datatset_dir.rstrip("/")}/{line.rstrip()}')
        
        if percentage < 1.0:
            assert range is None
            assert percentage > 0
            files_to_keep = int(len(self.__file_list) * percentage)
            if random: 
                import random
                random.seed(42)
                random.shuffle(files_to_keep)
            self.__file_list = self.__file_list[:files_to_keep]
        
        if range is not None:
            self.__file_list = self.__file_list[range[0]:range[1]]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self.__file_list): raise StopIteration
        path = self.__file_list[self.__idx]
        self.__idx += 1
        if path[0] != "/":
            path = datatset_dir + "/" + path
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
