from setuptools import setup

setup(
    name='TechnicalTest',
    version='0.3.3',
    author='Marina Lacambra',
    author_email='marina.lacambra@gmail.com',
    packages=['TechnicalTest'],
    scripts=['./TechnicalTest/technical_test.py'],
    url='http://pypi.python.org/pypi/TechnicalTest/',
    description='Technical Test',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
"tweet-preprocessor==0.6.0",
"pandas==1.3.4",
"nltk==3.6.5",
"sklearn"]
)