from setuptools import setup, find_packages

setup(
    name="hegemonikon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "nes-run=run_simulation:main",
        ],
    },
    author="Jesse Wright",
    license="MIT",
)
