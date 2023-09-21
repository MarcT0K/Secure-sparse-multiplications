from setuptools import setup, find_packages

setup(
    name="Secure sparse computations",
    version="1.0",
    description="Codebase used in the paper Damie et al. 'Secure sparse matrix multiplications and their applications to PPML'",
    author="Marc DAMIE",
    author_email="marc.damie@inria.fr",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "git+https://github.com/lschoe/mpyc/",
        "scipy",
        "numpy",
        "gmpy2",
        "matplotlib",
        "sklearn",
        "pandas",
        "psutil",
        "colorlog",
    ],
    entry_points={
        "console_scripts": [
            "run_all_experiments = securecomputations.launch_experiment:main",
            "generate_all_figures = securecomputations.generate_figures:main",
            "benchmark = securecomputations.benchmark:__main__",
        ],
    },
)
