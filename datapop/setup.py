from setuptools import setup, find_packages

setup(
    name='datapop',
    version='3.0.0',
    packages=['datapop'],
    package_dir={'datapop': 'datapop'},
    url='https://github.com/yandexdataschool/DataPopularity',
    license='',
    author='Mikhail Hushchyn',
    author_email='mikahil91@yandex-team.ru',
    description='Data Popularity for the LHCb data grid.',
    install_requires=[
        'numpy==1.9.2',
        'scipy==0.15.1',
        'pandas==0.14.0',
        'scikit-learn==0.17'
    ],
)
