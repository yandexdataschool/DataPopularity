from setuptools import setup, find_packages

setup(
    name='data-popularity',
    version='0.2.0',
    packages=['data-popularity'],
    package_dir={'data-popularity': 'data-popularity'},
    url='https://github.com/hushchyn-mikhail/DataPopularity',
    license='',
    author='Mikhail Hushchyn',
    author_email='mikahil91@yandex-team.ru',
    description='',
    install_requires=[
        'numpy >= 1.8.1',
        'scipy >= 0.14.0',
        'ipython >= 2.3.0',
        'pyzmq >= 14.3.1',
        'matplotlib >= 1.3.1',
        'openpyxl < 2.0.0',
        'pandas == 0.14.0',
        'pycurl >= 7.19.3',
        'Jinja2 >= 2.7.3',
        'numexpr >= 2.4',
        'plotly == 1.2.3',
        'scikit-learn >= 0.15.2'
        'rep==0.5.0',
        'kernel_regression==1.0'
    ],
)
