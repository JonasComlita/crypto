#!/usr/bin/env python3
"""
Setup script for enhanced blockchain implementation.
Handles installation of Python package and C++ acceleration libraries.
"""

import os
import sys
import platform
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import setuptools

# Check for Python version
if sys.version_info < (3, 8):
    sys.exit('Python >= 3.8 is required')

# Get package version
VERSION = '1.0.0'

# Define package metadata
DESCRIPTION = 'Enhanced blockchain implementation with PostgreSQL, msgpack, and C++ acceleration'
LONG_DESCRIPTION = '''
Enhanced blockchain implementation featuring:
- C++ acceleration for compute-intensive operations
- GPU mining support with CUDA
- PostgreSQL storage for high scalability
- MessagePack serialization for efficiency
- Multiprocessing for mining operations
'''

# Define dependencies
REQUIRES = [
    'asyncio>=3.4.3',
    'asyncpg>=0.27.0',  # PostgreSQL async driver
    'msgpack>=1.0.4',   # MessagePack serialization
    'pybind11>=2.10.0', # C++ binding
    'ecdsa>=0.18.0',    # Elliptic curve cryptography
    'cryptography>=39.0.0',  # For key encryption
    'aiohttp>=3.8.0',   # For HTTP APIs
    'uvloop>=0.17.0',   # Faster asyncio event loop
    'psutil>=5.9.0',    # System monitoring
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.20.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'isort>=5.12.0',
        'mypy>=1.0.0',
    ],
    'gpu': [
        'cuda-python>=12.0.0; platform_system=="Linux" or platform_system=="Windows"',
    ]
}

# Custom build extension for CMake
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the C++ extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for universal 2 platform on macOS
        if platform.system() == "Darwin":
            cmake_args = [
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
                '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64',
            ]
        else:
            cmake_args = [
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
            ]
        
        # Check if CUDA is available
        try:
            subprocess.check_output(['nvcc', '--version'])
            cmake_args.append('-DWITH_CUDA=ON')
            print("CUDA detected, enabling GPU support")
        except Exception:
            print("CUDA not found, building without GPU support")
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        if platform.system() == "Windows":
            # Use Visual Studio compiler
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            # Unix-like systems
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j4']
        
        # Create build directory
        os.makedirs(self.build_temp, exist_ok=True)
        
        # Run CMake and build
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, 
            cwd=self.build_temp
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, 
            cwd=self.build_temp
        )
        
        print()  # Add an empty line for cleaner output

# Setup configuration
setup(
    name='enhanced_blockchain',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Developer',
    author_email='developer@example.com',
    url='https://github.com/developer/enhanced_blockchain',
    packages=find_packages(),
    install_requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Distributed Computing',
    ],
    ext_modules=[CMakeExtension('blockchain_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'blockchain-node=enhanced_blockchain.main:main',
        ],
    },
)
