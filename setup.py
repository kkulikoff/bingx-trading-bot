#!/usr/bin/env python3
"""
Setup script for BingX Trading Bot package.
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Check Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or later is required.")

def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "BingX Trading Bot - Advanced algorithmic trading system for BingX exchange"

def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def get_version():
    version_path = Path(__file__).parent / "src" / "init.py"
    with open(version_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"

setup(
    name="bingx-trading-bot",
    version=get_version(),
    description="Advanced algorithmic trading bot for BingX exchange",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="BingX Trading Bot Team",
    author_email="support@bingx-trading-bot.com",
    url="https://github.com/your-username/bingx-trading-bot",
    project_urls={
        "Documentation": "https://github.com/your-username/bingx-trading-bot/docs",
        "Source": "https://github.com/your-username/bingx-trading-bot",
        "Tracker": "https://github.com/your-username/bingx-trading-bot/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords="trading, bot, algorithm, bingx, cryptocurrency, finance",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "tensorflow>=2.10.0; platform_system != 'Darwin' or platform_machine != 'arm64'",
            "tensorflow-macos>=2.10.0; platform_system == 'Darwin' and platform_machine == 'arm64'",
            "xgboost>=1.7.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "bingx-bot=main:main",
            "bingx-trading-bot=main:main",
        ],
    },
    package_data={
        "bingx_trading_bot": [
            "config/*.py",
            "config/*.conf",
            "config/*.encrypted",
            "web/templates/*.html",
            "web/static/css/*.css",
            "web/static/js/*.js",
            "web/static/img/*.png",
            "web/static/img/*.jpg",
            "web/static/img/*.svg",
        ],
    },
    data_files=[
        ("share/bingx-trading-bot/config", ["config/settings.py", "config/security.py"]),
        ("share/bingx-trading-bot", [".env.example", "requirements.txt"]),
        ("share/bingx-trading-bot/scripts", ["scripts/install.sh", "scripts/start.sh"]),
        ("share/doc/bingx-trading-bot", ["README.md", "LICENSE"]),
    ],
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    platforms=["any"],
)

if __name__ == "__main__":
    # Additional setup validation
    if not Path("requirements.txt").exists():
        print("Warning: requirements.txt not found. Dependencies may not install correctly.")
    
    if not Path("src").exists():
        print("Error: src directory not found. Please run from project root.")
        sys.exit(1)
    
    print("BingX Trading Bot setup configuration loaded successfully.")