from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

# Read long description from README.md if present
README = HERE / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else "AI-IDS package."

setup(
    name="ai-ids",
    version="0.1.0",
    description="AI-driven Intrusion Detection System utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://example.com/ai-ids",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19",
        "pandas>=1.1",
        "scikit-learn>=0.24",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"]
    },
    entry_points={
        "console_scripts": [
            # Replace ai_ids.__main__:main with your package entry point
            "ai-ids=ai_ids.__main__:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Security :: Intrusion Detection",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)