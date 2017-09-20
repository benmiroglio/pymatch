from distutils.core import setup
setup(
  name = 'matcher',
  packages = ['matcher'],
  version = '0.1',
  description = 'Matching techniques for Observational Studies',
  author = 'Ben Miroglio',
  author_email = 'benmiroglio@gmail.com',
  url = 'https://github.com/benmiroglio/matcher', 
  download_url = 'https://github.com/benmiroglio/matcher/archive/0.1.tar.gz', 
  keywords = ['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference'],
  classifiers = [],
  install_requires=[
  'seaborn',
  'statsmodels'.
  'scipy',
  'patsy',
  'matplotlib',
  'pandas', 
  'numpy'
  ]
)