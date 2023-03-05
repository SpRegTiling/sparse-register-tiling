#!/usr/bin/env python3

import distutils.command.clean
import glob
import os
import re
import shutil
import sys

import setuptools
import torch
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path):
    with open(version_file_path) as version_file:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M
        )
        # The following is used to build release packages.
        # Users should never use it.
        suffix = os.getenv("SBENCH_VERSION_SUFFIX", "")
        if version_match:
            return version_match.group(1) + suffix
        raise RuntimeError("Unable to find version string.")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    main_file = []
    source_utils = glob.glob(os.path.join(this_dir, "sbench", "cpp", "*.cpp"))

    sources = main_file + source_utils
    extension = CppExtension

    define_macros = []

    debug = False

    extra_compile_args = {"cxx": ["-O3"] if not debug else ["-g", "-O1"]}
    if sys.platform == "win32":
        extra_compile_args["cxx"].append("/MP")
    elif "OpenMP not found" not in torch.__config__.parallel_info():
        extra_compile_args["cxx"].append("-fopenmp")
    extra_compile_args["cxx"].append("-mavx")
    extra_compile_args["cxx"].append("-march=native")
    extra_compile_args["cxx"].append("-std=c++17")
    include_dirs = ["/opt/intel/mkl/include/",
                    os.path.join(this_dir, "sbench", "third_party", "vectorclass"),
                    os.path.join(this_dir, "sbench", "include")]

    ext_modules = [
        extension(
            "sbench._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):  # type: ignore
    def run(self):
        if os.path.exists(".gitignore"):
            with open(".gitignore", "r") as f:
                ignores = f.read()
                for wildcard in filter(None, ignores.split("\n")):
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    setuptools.setup(
        name="sbench",
        description="Utils for benchmarking sparse matrix multiplication",
        setup_requires=[],
        #install_requires=fetch_requirements(),
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        url="https://github.com/LucasWilkinson/spmm-nano-bench",
        python_requires=">=3.6",
        author="Lucas Wilkinson",
        author_email="wilkinson.lucas@gmail.com",
        long_description="SpMM Nano Benchmarks",
        long_description_content_type="text/markdown",
        zip_safe=False,
    )
