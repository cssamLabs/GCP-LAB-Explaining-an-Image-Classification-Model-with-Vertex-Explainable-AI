from setuptools import find_packages
from setuptools import setup

setup(
    name='flowers_trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['fire==0.4.0', 'tensorflow-hub==0.12.0'],
    description='Flowers image classifier training application.'
)
