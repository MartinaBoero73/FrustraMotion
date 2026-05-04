from setuptools import setup, find_packages

setup(
    name='frustramotion',
    version='0.1.0',
    description='A toolkit for extracting and analyzing dynamic frustration patterns from molecular dynamics trajectories.',
    author='martinaboero73',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
    ],
    entry_points={
        'console_scripts': [
            'frustramotion=frustramotion.cli:main',
        ],
    },
)