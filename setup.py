# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from setuptools import setup
from pathlib import Path

version_file = Path(__file__).parent / "xobjects/_version.py"
dd = {}
with open(version_file.absolute(), "r") as fp:
    exec(fp.read(), dd)
__version__ = dd["__version__"]


setup(
    name="xobjects",
    version=__version__,
    description="In-memory serialization and code generator for CPU and GPU",
    long_description="In-memory serialization and code generator for CPU and GPU",
    author="Riccardo De Maria",
    author_email="riccardo.de.maria@cern.ch",
    url="https://xsuite.readthedocs.io/",
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=["numpy", "cffi", "scipy"],
    packages=["xobjects"],
    license="Apache 2.0",
    download_url="https://pypi.python.org/pypi/xobjects",
    project_urls={
        "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
        "Documentation": "https://xsuite.readthedocs.io/",
        "Source Code": "https://github.com/xsuite/xobjects",
    },
    extras_require={
        "tests": ["pytest", "pytest-mock"],
    },
)
