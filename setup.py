from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Stellarator plotting routines'
LONG_DESCRIPTION = 'Python package for plotting stellarator codes output'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pystellplot", 
        version=VERSION,
        author="Antoine Baillod",
        author_email="ab5667@columbia.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'stellarator', 'vizualization'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
