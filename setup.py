# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='tyche',  # Replace with the actual name of your package
    version='1.0',          # Start with a version number
    author='Dylan Ray',       # Replace with your name
    author_email='ddray1993@gmail.com', # Replace with your email
    description='A package for querying CFR tables for poker and other related stuff', # Provide a brief description
    long_description=open('README.md').read(), # Optionally, read from a README file
    long_description_content_type='text/markdown', # Specify the content type if using Markdown
    url='https://github.com/kilmoretrout/tyche', # Replace with your project's URL (e.g., GitHub repo)
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # Include all .so files from the 'lib' directory within any package
        'tyche': ['lib/*.so', '*.config'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11', # Specify the minimum Python version required
    install_requires=[
        'numpy',
        'streamlit',
        'pokerkit',
        'pandas',
        'scipy',
        'seaborn',
        'scikit-learn',
        'diskcache',
        # List your project's dependencies here.
        # For example:
        # 'numpy>=1.18.0',
        # 'pandas',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'your_script_name = your_package_name.module:main_function',
    #     ],
    # },
)
