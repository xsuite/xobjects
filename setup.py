from setuptools import setup

setup(
        name='xobjects',
        version='0.0.0',
        description='Manage dynamic data in devide buffers',
        author='Riccardo De Maria',
        author_email='riccardo.de.maria@cern.ch',
        url='https://github.com/rdemaria/cobjects',
        python_requires='>=3.7',
        setup_requires=[],
        install_requires=['numpy','pyopencl'],
        packages=['xobjects'],
)
