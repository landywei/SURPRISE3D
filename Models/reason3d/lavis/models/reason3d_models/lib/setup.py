from setuptools import find_packages, setup

import os
import os.path as osp
import torch
from glob import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_sources(module, surfix='*.c*'):
    src_dir = osp.join(*module.split('.'), 'src')
    cuda_dir = osp.join(src_dir, 'cuda')
    cpu_dir = osp.join(src_dir, 'cpu')
    return glob(osp.join(src_dir, surfix)) + glob(osp.join(cuda_dir, surfix)) + glob(osp.join(cpu_dir, surfix))


def _default_system_include() -> str:
    """Conda's nvcc uses a sysrooted host compiler; -I/usr/include can mix with host glibc
    headers and break the compile (e.g. bits/timesize.h). Prefer the conda sysroot if present.
    """
    p = os.environ.get("CONDA_PREFIX", "")
    if p:
        for sub in (
            ("x86_64-conda-linux-gnu", "sysroot", "usr", "include"),
            ("x86_64-conda_cos7-linux-gnu", "sysroot", "usr", "include"),
        ):
            cand = osp.join(p, *sub)
            if osp.isfile(osp.join(cand, "stdio.h")):
                return cand
    return "/usr/include"


def _has_google_dense_hash_map(parent: str) -> bool:
    p = osp.join(parent, "google", "dense_hash_map")
    return osp.isfile(p) or osp.isfile(p + ".h")


def _sparsehash_include_dirs(sysinc: str) -> list:
    """#include <google/dense_hash_map> needs -I<PARENT> where PARENT/google/dense_hash_map exists.
    GPU nodes may lack /usr/include/{google,sparsehash}; prefer conda: `conda install -c conda-forge sparsehash`.
    Optional override: POINTGROUP_SPARSEHASH_ROOT=/path/to/parent_of_google
    """
    extra = os.environ.get("POINTGROUP_SPARSEHASH_ROOT", "").strip()
    if extra and _has_google_dense_hash_map(extra):
        return [extra]
    cands = []
    p = os.environ.get("CONDA_PREFIX", "")
    if p:
        cands.extend(
            [
                osp.join(p, "include", "sparsehash"),
                osp.join(p, "include"),
            ]
        )
    cands.extend(
        [
            osp.join(sysinc, "sparsehash"),
            sysinc,
            "/usr/include/sparsehash",
            "/usr/include",
        ]
    )
    seen = set()
    for c in cands:
        c = osp.normpath(c)
        if c in seen:
            continue
        seen.add(c)
        if _has_google_dense_hash_map(c):
            return [c]
    return []


def get_include_dir(module):
    include_dir = osp.join(*module.split('.'), 'include')
    sysinc = _default_system_include()
    sh = _sparsehash_include_dirs(sysinc)
    default_includes = [sysinc] + sh
    if osp.exists(include_dir):
        return [osp.abspath(include_dir)] + default_includes
    else:
        return default_includes


def make_extension(name, module):
    if not torch.cuda.is_available(): return
    extra = os.environ.get("POINTGROUP_CUDA_INCLUDE_DIRS", "")
    extra_includes = [p for p in extra.split(os.pathsep) if p]
    extersion = CUDAExtension
    return extersion(
        name='.'.join([module, name]),
        sources=get_sources(module),
        include_dirs=get_include_dir(module) + extra_includes,
        extra_compile_args={
            'cxx': ['-g'],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ],
        },
        define_macros=[('WITH_CUDA', None)])


setup(
    name='pointgroup_ops',
    ext_modules=[make_extension(name='pointgroup_ops_ext', module='pointgroup_ops')],
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension})
