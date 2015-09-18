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
        'rep>=0.6.0',
        'kernel_regression==1.0',
        'xlrd >= 0.9.3'
    ],
)
