from setuptools import setup, find_packages

setup(
    name="football_outcome_predictor",
    version="0.1.0",
    description="A machine learning model to predict football match outcomes (win/draw/loss)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/football-outcome-predictor",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'football-predictor=main:main',
        ],
    },
)
