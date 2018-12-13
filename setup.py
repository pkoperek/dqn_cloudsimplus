from setuptools import setup

setup(
    name='dqn_cloudsimplus',
    version='0.0.1',
    install_requires=[
        'gym',
        'gym_cloudsimplus',
        'torch',
        'torchvision',
        'psycopg2-binary',
    ]
)
