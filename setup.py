import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="invertpy",
    version="1.0.0",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="Python package of computational models relevant to invertebrate processing, from the environment to"
                "sensor responses and to deeper neural responses in the invertebrate brain.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgkanias/InvertBrain",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: OSI Approved :: GPL-3.0 Licence",
        "Operating System :: OS Independent"
    ],
    packages=["invertbrain", "invertsense", "invertio"],
    package_dir={
        "invertbrain": "src/invertbrain",
        "invertsense": "src/invertsense",
        "invertio": "src/invertio"},
    python_requires=">=3.8",
)