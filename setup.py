from setuptools import setup

setup(
    name='lstm_problems',
    version='0.0.0',
    description='Problems to test if a model can learn long-term dependencies',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/lstm_problems',
    packages=['lstm_problems'],
    long_description="""
    Suite of toy problems which can test whether a model can learn long-term
    dependencies.
    """,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords='lstm',
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
    ],
)
