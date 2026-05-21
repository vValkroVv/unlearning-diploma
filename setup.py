from pathlib import Path

from setuptools import find_packages, setup

# Read dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="dualcf-open-unlearning",
    version="0.1.0",
    author="Valerii Kropotin; upstream OpenUnlearning authors",
    description="Diploma research fork of OpenUnlearning with DualCF tooling.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/vValkroVv/unlearning-diploma",
    project_urls={
        "Upstream OpenUnlearning": "https://github.com/locuslab/open-unlearning",
    },
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install .[lm-eval]`
        "lm_eval": [
            "lm-eval==0.4.8",
        ],  # Backward-compatible alias.
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.10",
)
