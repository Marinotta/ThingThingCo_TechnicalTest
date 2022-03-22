from setuptools import setup

setup(
    name='TechnicalTest',
    version='0.2.2',
    author='Marina Lacambra',
    author_email='marina.lacambra@gmail.com',
    packages=['TechnicalTest'],
    scripts=['./TechnicalTest/technical_test.py'],
    url='http://pypi.python.org/pypi/TechnicalTest/',
    description='Technical Test',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=["preprocessor", "nltk", "pandas", "sklearn"]
)