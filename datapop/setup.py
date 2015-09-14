from setuptools import setup, find_packages

setup(
    name='datapop',
    version='0.2.0',
    packages=['datapop'],
    package_dir={'datapop': 'datapop'},
    url='https://github.com/hushchyn-mikhail/DataPopularity',
    license='',
    author='Mikhail Hushchyn',
    author_email='mikahil91@yandex-team.ru',
    description='',
    install_requires=[
        'numpy >= 1.9.2',
        'scipy >= 0.15.1',
        'ipython == 3.0.0',
        'pyzmq >= 14.5.0',
        'matplotlib >= 1.4.3',
        'openpyxl >= 1.8.6',
        'pandas >= 0.14.0',
        'pycurl >= 7.19.5.1',
        'Jinja2 >= 2.7.3',
        'numexpr >= 2.4',
        'plotly >= 1.2.3',
        'scikit-learn >= 0.16.1',
        'cffi',
        'rep>=0.6.0',
        'kernel_regression==1.0',
        'xlrd >= 0.9.3'
    ],
)
