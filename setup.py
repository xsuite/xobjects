# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from setuptools import setup

setup(
    name="xobjects",
    version="0.1.15",
    description="In-memory serialization and code generator for CPU and GPU",
    author="Riccardo De Maria",
    author_email="riccardo.de.maria@cern.ch",
    url="https://github.com/rdemaria/cobjects",
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=["numpy", "cffi"],
    packages=["xobjects"],
)
