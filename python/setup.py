from setuptools import setup

setup(name='meas',
      version='0.0.1',
      url='http://github.com/JasonTam/meas',
      packages=['meas'],
      install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'scikit-learn',
            'ml_metrics',
      ],
      tests_require=[
            'pytest'
      ],
      )
