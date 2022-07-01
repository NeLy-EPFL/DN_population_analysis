from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="DN_population_analysis",
    version="0.1",
    packages=["DN_population_analysis",],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Code for preprocessing and analysis of the data published in Aymanns et al. 2022",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/DN_population_analysis.git",
)
