from setuptools import setup, find_packages
from glob import glob

so_files = glob("pupok/python/pupok_core*.so")

setup(
    name="pupok",
    version="0.1",
    description="Cosine similarity utilities with Python bindings",
    packages=find_packages(),
    package_data={
        "pupok": ["python/*.so"],
    },
)