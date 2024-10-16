from setuptools import setup, find_packages

setup(
    name="securesparsecomputations",
    version="1.0",
    description="Codebase used in the paper'Secure sparse matrix multiplications and their applications to PPML'",
    author="[anonymous]",
    author_email="[anonymous]",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mpyc @ git+https://github.com/lschoe/mpyc/",
        "scipy",
        "numpy",
        "gmpy2",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "psutil",
        "colorlog",
    ],
    entry_points={
        "console_scripts": [
            "run_all_experiments = securesparsecomputations.launch_experiment:main",
            "generate_all_figures = securesparsecomputations.generate_figures:main",
            "benchmark = securesparsecomputations.benchmark:run",
            "benchmark_spam_detection = securesparsecomputations.applications.spam_detection:run",
            "benchmark_recommender = securesparsecomputations.applications.recommender_system:run",
            "benchmark_access_control = securesparsecomputations.applications.access_control:run",
        ],
    },
)
