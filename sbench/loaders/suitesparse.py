import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from .load import load_csr


SS_STORAGE_PATH = '/sdb/datasets/ss/'
DEFAULT_MATRIX_IDS = [
    2374, 1281,  937, 1421, 1278, 1403, 1269, 1290, 1254, 1266,  369, 1258, 1580,
     947, 1644, 2664, 2547,  939, 2543, 1267,  407,  887,  440,  760, 1406,  759,
    1610, 1847,  354, 1203
]

class SuiteSparsePathIterator:
    def __init__(self, matrix_ids=None, paths=None, skip_list=None):
        self.__matrix_ids = matrix_ids
        self.__file_list = []

        if matrix_ids is not None:
            import ssgetpy

            matrix_list = ssgetpy.matrix.MatrixList()
            def _matrix_localfile(mtx: ssgetpy.matrix.Matrix):
                return f'{mtx.localpath(destpath=SS_STORAGE_PATH, extract=True)[0]}/{mtx.name}.mtx'

            for id in matrix_ids:
                matrix_list += ssgetpy.fetch(id, location=SS_STORAGE_PATH, dry_run=True)
                if skip_list:
                    skip = False
                    for skip_name in skip_list:
                        if skip_name in matrix_list[-1].name:
                            skip = True
                    if skip:
                        continue

                if not os.path.exists(_matrix_localfile(matrix_list[-1])):
                    matrix_list[-1].download(destpath=SS_STORAGE_PATH, extract=True)

                self.__file_list.append(_matrix_localfile(matrix_list[-1]))
        else:
            self.__file_list =[SS_STORAGE_PATH + path for path in paths if path.split('/')[-1].split('.')[-2] not in skip_list or []]

        print([path.replace("/sdb/datasets/ss/", "") for path in self.__file_list])

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self.__file_list): raise StopIteration
        path = self.__file_list[self.__idx]
        self.__idx += 1
        return path


class SuiteSparseLoader:
    def __init__(self, matrix_ids=None, paths=None, loader=load_csr, skip_list=None):
        self._path_iterator_base = SuiteSparsePathIterator(matrix_ids, paths, skip_list=skip_list)
        self._path_iterator = None
        self.loader = loader

    def __iter__(self):
        self._path_iterator = self._path_iterator_base.__iter__()
        return self

    def __next__(self):
        filepath = next(self._path_iterator)
        return self.loader(filepath), os.path.abspath(filepath)


