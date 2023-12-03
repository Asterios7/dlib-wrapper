from setuptools import setup, find_packages

with open('README.md', "r") as f:
    long_description = f.read()

setup(
    name='dlib_wrapper',
    version='0.0.1',
    description="A dlib wrapper for face detection and encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Asterios7/dlib-wrapper.git",
    author="Asterios7",
    license="None",
    packages=find_packages(),
    install_requires=[
        'dlib-bin==19.24.2',
        'gdown==4.6.4',
        'requests>=2.30.0',
        'Pillow>=9.0.0',
        'numpy>=1.22.1'
    ],
    extras_require = {
        "dev": ["pytest>=7", "pytest-cov>=4.1.0"]
    },
    python_requires=">=3.8"
)