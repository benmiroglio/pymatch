from setuptools import setup, find_packages
dependencies =[
  'seaborn',
  'statsmodels',
  'scipy',
  'patsy',
  'matplotlib',
  'pandas', 
  'numpy'
  ]

VERSION = "0.1.3"

setup(
  name = 'pymatch',
  packages = find_packages('pymatch'),
  version = VERSION,
  description = 'Matching techniques for Observational Studies',
  author = 'Ben Miroglio',
  author_email = 'benmiroglio@gmail.com',
  url = 'https://github.com/benmiroglio/pymatch',
  download_url = 'https://github.com/benmiroglio/pymatch/archive/{}.tar.gz'.format(VERSION), 
  keywords = ['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference'],
  classifiers = [],
  include_package_data=True,
  package_data={'': ['*.py']},
  requires=dependencies,
  provides=['utils', 'sys']
)