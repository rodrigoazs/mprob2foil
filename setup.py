#! /usr/bin/env python

from __future__ import print_function
import sys
import os


version_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'probfoil', 'version.py')

version = {}
with open(version_file) as fp:
    exec(fp.read(), version)
version = version['version']

if __name__ == '__main__':
    from setuptools import setup, find_packages

    package_data = {}

    setup(
        name='probfoil',
        version=version,
        description='Prob2FOIL: rule learner for probabilistic logic',
        url='https://dtai.cs.kuleuven.be/software/probfoil',
        author='Anton Dries',
        author_email='anton.dries@cs.kuleuven.be',
        license='Apache Software License',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Prolog',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        keywords='probabilistic logic learning',
        packages=find_packages(),
        entry_points={
            'console_scripts': ['probfoil=probfoil.probfoil:main']
        },
        package_data=package_data,
        install_requires=['problog']
    )


def increment_release(v):
    v = v.split('.')
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1)]
    else:
        v = v[:4]
    return '.'.join(v)


def increment_dev(v):
    v = v.split('.')
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1), 'dev1']
    else:
        v = v[:4] + ['dev' + str(int(v[4][3:]) + 1)]
    return '.'.join(v)


def increment_version_dev():
    v = increment_dev(version)
    os.path.dirname(__file__)
    with open(version_file, 'w') as f:
        f.write("version = '%s'\n" % v)


def increment_version_release():
    v = increment_release(version)
    with open(version_file, 'w') as f:
        f.write("version = '%s'\n" % v)
