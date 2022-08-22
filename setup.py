import os
import setuptools

setuptools.setup(
    name = 'BAT',
    version='1.0.0',
    description='BORN FOR AUTO-TAGGING: FASTER AND BETTER WITH NEW OBJECTIVE FUNCTIONS',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.md'
        )
    ).read(),
    author='awoo AI Lab',
    author_email='awoobrain@awoo.com.tw',
    packages=setuptools.find_packages(),
    license='MIT',
)