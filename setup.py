import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="goes",
    version="0.0.1",
    author="Scott Staniewicz",
    author_email="scott.stanie@gmail.com",
    description="Tools for downloading, cropping, plotting GOES images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottstanie/goes",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "boto3",
    ],
    zip_safe=False,
)
