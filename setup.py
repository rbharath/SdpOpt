from setuptools import setup, find_packages
import sys
from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import glob
import copy
import shutil
import textwrap
import tempfile
import subprocess
from distutils.ccompiler import new_compiler
from distutils.spawn import find_executable
import numpy as np
from numpy.distutils import system_info

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
try:
    import Cython
    from Cython.Distutils import build_ext
    if Cython.__version__ < '0.18':
        raise ImportError()
except ImportError:
    print('Cython version 0.18 or later is required. Try "easy_install cython"')
    sys.exit(1)

##########################
VERSION = '0.1'
ISRELEASED = False
__version__ = VERSION
##########################

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Programming Language :: C++
Programming Language :: Python
Development Status :: 3 - Alpha
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.5
"""
def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

def detect_openmp():
    "Does this compiler support OpenMP parallelization?"
    compiler = new_compiler()
    print('\n\033[95mAttempting to autodetect OpenMP support...\033[0m')
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print('\033[92mCompiler supports OpenMP\033[0m\n')
    else:
        print('\033[91mDid not detect OpenMP support; parallel support disabled\033[0m\n')
    return hasopenmp, needs_gomp



class custom_build_ext(build_ext):
    def build_extensions(self):
        # Here come the cython hacks
        from distutils.command.build_ext import build_ext as _build_ext
        # first, AVOID calling cython.build_ext's build_extensions
        # method, because it cythonizes all of the pyx files to cpp
        # here, which we do *not* want to do. Instead, we want to do
        # them one at a time during build_extension so that we can
        # regenerate them on every extension. This is necessary for getting
        # the single/mixed precision builds to work correctly, because we
        # use the same pyx file, with different macros, and make differently
        # named extensions. Since each extension needs to have a unique init
        # method in the cpp code, the cpp needs to be translated fresh from
        # pyx.
        _build_ext.build_extensions(self)

    def build_extension(self, ext):
        build_ext.cython_gdb = True
        # Clean all cython files for each extension
        # and force the cpp files to be rebuilt from pyx.
        cplus = self.cython_cplus or getattr(ext, 'cython_cplus', 0) or \
                (ext.language and ext.language.lower() == 'c++')
        if len(ext.define_macros) > 0:
            for f in ext.sources:
                if f.endswith('.pyx'):
                    if cplus:
                        compiled = f[:-4] + '.cpp'
                    else:
                        compiled = f[:-4] + '.c'
                    if os.path.exists(compiled):
                        os.unlink(compiled)
        ext.sources = self.cython_sources(ext.sources, ext)
        build_ext.build_extension(self, ext)

openmp_enabled, needs_gomp = detect_openmp()
extra_compile_args = ['-msse3']
if openmp_enabled:
    extra_compile_args.append('-fopenmp')
libraries = ['gomp'] if needs_gomp else []
extensions = []
lapack_info = get_lapack()

extensions.append(
    Extension('sdpopt._reversibility',
              sources=['src/reversibility.pyx'],
              libraries=['m'],
              include_dirs=[np.get_include()]))

extensions.append(
    Extension('mixtape._mslds',
              language='c++',
              sources=['platforms/cpu/wrappers/MetastableSLDSCPUImpl.pyx'] +
                        glob.glob('platforms/cpu/kernels/*.c') +
                        glob.glob('platforms/cpu/kernels/*.cpp'),
              libraries=libraries + lapack_info['libraries'],
              extra_compile_args=extra_compile_args,
              extra_link_args=lapack_info['extra_link_args'],
              include_dirs=[np.get_include(), 'platforms/cpu/kernels/include/',
                            'platforms/cpu/kernels/']))


def main():
    if 'develop' not in sys.argv:
        raise NotImplementedError("Use python setup.py develop.")
    setup(
        name="sdpopt",
        author="Bharath Ramsundar",
        author_email="rbharath@stanford.edu",
        url='https://github.com/rbharath/sdpopt',
        description='A first order library for SDP solution',
        packages=find_packages(),
        cmdclass={'build_ext': custom_build_ext})
    )

if __name__ == '__main__':
    main()
