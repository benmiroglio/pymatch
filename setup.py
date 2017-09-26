from distutils.core import setup

dependencies =[
  'seaborn',
  'statsmodels',
  'scipy',
  'patsy',
  'matplotlib',
  'pandas', 
  'numpy'
  ]

VERSION = "0.0.9"

setup(
  name = 'pymatch',
  packages = ['pymatch'],
  version = VERSION,
  description = 'Matching techniques for Observational Studies',
  author = 'Ben Miroglio',
  author_email = 'benmiroglio@gmail.com',
  url = 'https://github.com/benmiroglio/pymatch',
  download_url = 'https://github.com/benmiroglio/pymatch/archive/{}.tar.gz'.format(VERSION), 
  keywords = ['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference'],
  classifiers = [],
  requires=dependencies,
  provides=['utils', 'sys']
)