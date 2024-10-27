from setuptools import setup, find_packages, Extension
import pybind11

ext_modules = [
    Extension(
        'pupok.python.pupok_core',
        sources=[
            'pupok/python/bindings.cpp',
            'pupok/src/cosine_similarity.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            'pupok/src',
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],
    ),
]

setup(
    name='pupok',
    version='0.1',
    description='Cosine similarity utilities with Python bindings',
    packages=find_packages(include=['pupok', 'pupok.*']),
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False,
)
