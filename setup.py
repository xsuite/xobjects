# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from setuptools import setup

setup(
    name="xobjects",
    version="0.1.21",
    description="In-memory serialization and code generator for CPU and GPU",
    long_description="In-memory serialization and code generator for CPU and GPU",
    author="Riccardo De Maria",
    author_email="riccardo.de.maria@cern.ch",
    url='https://xsuite.readthedocs.io/',
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=["numpy", "cffi"],
    packages=["xobjects"],
    license='Apache 2.0',
    download_url="https://pypi.python.org/pypi/xobjects",
    project_urls={
            "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
            "Documentation": 'https://xsuite.readthedocs.io/',
            "Source Code": "https://github.com/xsuite/xobjects",
        },
)
