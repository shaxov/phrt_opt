import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phrt_opt",
    version="0.0.1",
    author="Maksym Shpakovych",
    author_email="maksym.shpakovych@unilim.fr",
    description="Phase retrieval algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.xlim.fr/shpakovych/phrt-opt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
