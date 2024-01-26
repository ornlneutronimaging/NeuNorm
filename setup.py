#!/usr/bin/env python
from setuptools import setup, find_packages, Command
import os
import sys
from shutil import rmtree

import NeuNorm

# to create library
# > python setup.py upload

NAME = 'NeuNorm'
DESCRIPTION = "neutron normalization data"
LONGDESCRIPTION = "See the README.md file on GitHub for more information"
URL = "https://github.com/ornlneutronimaging/NeuNorm"
EMAIL = "bilheuxjm@ornl.gov"
AUTHOR = "Jean Bilheux"
VERSION = NeuNorm.__version__
KEYWORDS = "neutron normalization imaging"

# what packages are required for this module to be executed
REQUIRED = ['numpy',
            'pillow',
            'pathlib',
            'astropy',
            'scipy',
            ]

here = os.path.abspath('./')


class UploadCommand(Command):
    """Support setup.py upload."""
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        """Initialization options."""
        pass

    def finalize_options(self):
        """Finalize options."""
        pass

    def run(self):
        """Remove previous builds."""
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel distribution...')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        sys.exit()


setup(
        name=NAME,
        description=DESCRIPTION,
        long_description=LONGDESCRIPTION,
        url=URL,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        packages=find_packages(exclude=['tests', 'notebooks']),
        include_package_data=True,
        test_suite='tests',
        install_requires=REQUIRED,
        dependency_links=[
        ],
        license='BSD',
        keywords=KEYWORDS,
        classifiers=['Development Status :: 3 - Alpha',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.7'],
        cmdclass={
            'upload': UploadCommand,
        },
)

# End of file
