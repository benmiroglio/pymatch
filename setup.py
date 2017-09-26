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

setup(
  name = 'pymatch',
  packages = ['pymatch'],
  version = '0.0.7',
  description = 'Matching techniques for Observational Studies',
  author = 'Ben Miroglio',
  author_email = 'benmiroglio@gmail.com',
  url = 'https://github.com/benmiroglio/pymatch', 
  download_url = 'https://github.com/benmiroglio/pymatch/archive/0.0.7.tar.gz', 
  keywords = ['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference'],
  classifiers = [],
  requires=dependencies,
  provides=['utils', 'sys']
)