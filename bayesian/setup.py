import setuptools
long_description = """
# TD Tools
"""
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="td_tools",
    version="0.1.0",
    author="Equinor ASA",
    author_email="apenh@equinor.com",
    description="Bayesian calibration of velicty to CKS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dischler/td_tools",
    packages=setuptools.find_packages(exclude=[]),
    install_requires=[
        "numpy","pandas","scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)