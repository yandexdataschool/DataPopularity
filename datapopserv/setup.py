from setuptools import setup, find_packages

setup(
    name='datapopserv',
    version='3.0.0',
    url='https://github.com/yandexdataschool/DataPopularity',
    license='',
    packages=find_packages(),
    author='mikhail91',
    author_email='mikhail91@yandex-team.ru',
    description='Api service for the datapop library.',
    install_requires=[
        'flask == 0.10.1',
        'flask-restful == 0.3.2',
        'datapop == 3.0.0'
    ],
)
