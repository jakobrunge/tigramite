from distutils.core import setup

setup(
    name='tigramite',
    version='3.0',
    packages=['tigramite',],
    license='GNU General Public License v3.0',
    description='Tigramite causal discovery for time series',
    author='Jakob Runge',
    author_email='jakobrunge@posteo.de',
    url='https://github.com/jakobrunge/tigramite_v3/',
    long_description=open('README.md').read(),
    keywords = ['causality', 'time series'],

)