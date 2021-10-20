from __future__ import annotations

import glob
import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError
from distutils.errors import DistutilsExecError
from distutils.errors import DistutilsPlatformError

import numpy
from Cython.Build import cythonize
from setuptools import Extension

# Cython Extensions
cython_file: str
cython_files: list[tuple[str, str]] = [
    # First position has the filename using . and without extension, e.g.,
    #  recsys_framework.Recommenders.SLIM.Cython.SLIM_BPR
    # Second position is the cython file path, e.g.,
    #  recsys_framework/Recommenders/SLIM/Cython/SLIM_BPR.pyx
    (cython_file[:-4].replace("/", "."), cython_file)
    for cython_file in glob.glob('recsys_framework/**/Cython/*.pyx', recursive=True)
]

extensions = [
    Extension(
        name=cython_file_name,
        sources=[cython_file_source],
        include_dirs=[numpy.get_include()]
    )
    for cython_file_name, cython_file_source in cython_files
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    # This class allows C extension building to fail.
    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError) as exception:
            raise exception

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError) as exception:
            raise exception


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    # Importing Distribution at the top of the file throws an
    # error due to inconsistencies between setuptools and distutils: https://stackoverflow.com/a/65711633/13385583
    from distutils.dist import Distribution
    distribution = Distribution({
        "name": "recsys_framework",
        "ext_modules": cythonize(
            extensions,
            quiet=False,
            exclude_failures=False
        )
    })
    distribution.package_dir = "recsys_framework"

    cmd = ExtBuilder(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    outputs = cmd.get_outputs()
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)

        if not os.path.exists(output):
            continue

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build({})
