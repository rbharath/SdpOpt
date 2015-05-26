from setuptools import setup, find_packages
import sys


def main():
    if 'develop' not in sys.argv:
        raise NotImplementedError("Use python setup.py develop.")
    setup(
        name="sdpopt",
        url='https://github.com/rbharath/sdpopt',
        description='A first order library for SDP solution',
        packages=find_packages(),
    )

if __name__ == '__main__':
    main()
